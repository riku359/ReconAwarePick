#!/usr/bin/env python3
"""compare_official_vs_triangular.py -- compare the released predictMask (uniform average
of the overlapping windows + 8-rotation TTA + fixJumpInBorders) against the triangular-window
blend extract_blended (stride-factor=2, no fixJump, single orientation) at **model scale**.
It computes the difference, a straight-line (stripe) score and the border difference, and
saves the arrays for the figures.

This is the material behind Fig. S2 of the paper.

Targets: every micrograph in the triangular mask store ($RAPICK_WORK/masks), whose npz meta
names the entry, the micrograph and the box size.
Output: <out>/comparison.csv and <out>/arrays/<id>__<mic>.npz (off_m, tri_m, mic_pre as f16).
Run in the micrograph_cleaner environment, on a GPU.
"""
import argparse, os, sys, glob, json, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env  # noqa: E402


def stripe_peak(m):
    """How far the single worst line (row/column) stands out above the median. Large = a
    straight-line discontinuity."""
    rj = np.abs(np.diff(m, axis=0)).mean(1)
    cj = np.abs(np.diff(m, axis=1)).mean(0)
    rr = float(rj.max() / (np.median(rj) + 1e-6))
    cc = float(cj.max() / (np.median(cj) + 1e-6))
    return max(rr, cc)


def ring_masks(shape, frac=0.08):
    h, w = shape
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    ring = np.zeros(shape, bool)
    ring[:bh, :] = ring[-bh:, :] = ring[:, :bw] = ring[:, -bw:] = True
    return ring, ~ring


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-root", default=None,
                    help="triangular mask store naming the micrographs to compare "
                         "(default: $RAPICK_WORK/masks)")
    ap.add_argument("--mic-root", default=None,
                    help="parent of <id>/micrographs (default: $RAPICK_DATA/cryoppp)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: $RAPICK_WORK/mask_compare)")
    ap.add_argument("--ids", default="", help="comma-separated ids to restrict to (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="cap on micrographs per entry")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=None,
                    help="model .h5 (default: $RAPICK_DATA/checkpoints/%s)" % env.MODEL_BASENAME)
    args = ap.parse_args()

    model = env.resolve_model(args.model)
    mask_root = args.mask_root or env.mask_store_root()
    mic_root = args.mic_root or os.path.join(env.data_root(), "cryoppp")
    out_dir = args.out_dir or os.path.join(env.work_root(), "mask_compare")

    env._ensure_modern_ptxas()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import cv2, mrcfile, keras
    import micrograph_cleaner_em as mce
    from triangular_mask import build_extractor, extract_blended
    from micrograph_cleaner_em.preprocessMic import preprocessMic

    arr_dir = os.path.join(out_dir, "arrays")
    os.makedirs(arr_dir, exist_ok=True)

    full = keras.models.load_model(model,
                                   custom_objects={"LeakyReLU": keras.layers.LeakyReLU}, compile=False)
    extractor = build_extractor(full)
    mp = mce.MaskPredictor(174, deepLearningModelFname=model, gpus=[0])

    only = set(x for x in args.ids.split(",") if x)
    npzs = sorted(glob.glob(os.path.join(mask_root, "*", "*_tri.npz")))
    if only:
        npzs = [p for p in npzs if os.path.basename(os.path.dirname(p)) in only]
    if args.limit:
        per_id = {}
        kept = []
        for p in npzs:
            eid = os.path.basename(os.path.dirname(p))
            per_id[eid] = per_id.get(eid, 0) + 1
            if per_id[eid] <= args.limit:
                kept.append(p)
        npzs = kept
    print("[cfg] %d micrographs to compare" % len(npzs))

    fields = ["dist", "empiar_id", "micrograph", "box", "h", "w", "mad", "maxd", "signed",
              "off_stripe", "tri_stripe", "border_ratio", "off_max", "tri_max",
              "off_fracpos", "tri_fracpos", "status"]
    fcsv = open(os.path.join(out_dir, "comparison.csv"), "w", newline="")
    wr = csv.DictWriter(fcsv, fieldnames=fields)
    wr.writeheader()

    t0 = time.time()
    for i, f in enumerate(npzs):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        eid, box = str(meta["empiar_id"]), int(meta["box_size"])
        mic = os.path.splitext(meta["micrograph"])[0]      # the store keeps the name with .mrc
        dist = meta.get("dist_class") or env.dist_class_of(eid)
        row = {k: "" for k in fields}
        row.update({"dist": dist, "empiar_id": eid, "micrograph": mic, "box": box})
        try:
            mrc_path = os.path.join(mic_root, eid, "micrographs", mic + ".mrc")
            with mrcfile.open(mrc_path, permissive=True) as m:
                arr = np.squeeze(np.asarray(m.data, dtype=np.float32))
            if arr.ndim == 3:
                arr = arr.mean(0)
            mic_pre = preprocessMic(arr, box)                       # model scale
            _, tri_m, _, _ = extract_blended(extractor, mic_pre, 2, 1)   # pool=1 -> mask at model scale
            mp.boxSize = box
            off_full = mp.predictMask(arr)                          # full resolution
            off_m = cv2.resize(off_full, (mic_pre.shape[1], mic_pre.shape[0]),
                               interpolation=cv2.INTER_AREA)
            diff = off_m - tri_m
            ring, interior = ring_masks(diff.shape)
            bmad = float(np.abs(diff[ring]).mean())
            imad = float(np.abs(diff[interior]).mean()) + 1e-6
            row.update({
                "h": diff.shape[0], "w": diff.shape[1],
                "mad": "%.4f" % np.abs(diff).mean(), "maxd": "%.4f" % np.abs(diff).max(),
                "signed": "%.4f" % diff.mean(),
                "off_stripe": "%.1f" % stripe_peak(off_m), "tri_stripe": "%.1f" % stripe_peak(tri_m),
                "border_ratio": "%.2f" % (bmad / imad),
                "off_max": "%.3f" % off_m.max(), "tri_max": "%.3f" % tri_m.max(),
                "off_fracpos": "%.4f" % np.mean(off_m >= 0.5), "tri_fracpos": "%.4f" % np.mean(tri_m >= 0.5),
                "status": "ok"})
            np.savez_compressed(os.path.join(arr_dir, "%s__%s.npz" % (eid, mic)),
                                off=off_m.astype(np.float16), tri=tri_m.astype(np.float16),
                                pre=mic_pre.astype(np.float16), meta=json.dumps(meta))
        except Exception as e:  # noqa: BLE001
            row["status"] = ("ERR: %s" % e)[:120]
        wr.writerow(row); fcsv.flush()
        if (i + 1) % 10 == 0:
            print("[%d/%d] %.0fs" % (i + 1, len(npzs), time.time() - t0))
    fcsv.close()
    print("[done] %.0fs -> %s" % (time.time() - t0, os.path.join(out_dir, "comparison.csv")))


if __name__ == "__main__":
    main()
