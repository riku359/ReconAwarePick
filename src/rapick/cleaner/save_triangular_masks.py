#!/usr/bin/env python3
"""save_triangular_masks.py -- compute the triangular-window extract_blended masks once
and store them as npz.

It builds the triangular store ($RAPICK_WORK/masks) that pairs with the store of released
predictMask masks ($RAPICK_WORK/masks_official), so that re-selecting the released
version's failures and drawing them afterwards need no GPU. The targets are the
micrographs whose released-arm meta reports a ground-truth removal count
(n_gt_in_contam) at or above a threshold.

The stored mask is
  extract_blended(extractor, preprocessMic(image, box), 2, 1)[1]
so it matches a run-time computation bit for bit. It is stored at model scale and scaled
back to image scale with cv2.resize(INTER_LINEAR) when drawing (the released store is
full-res and can be used as it is).

Output: <out-root>/<id>/<mic>_tri.npz  (tri: float16 at model scale, meta: json).
Run in the micrograph_cleaner environment. The GPU defaults to 1 (MaskPredictor restricts
CUDA_VISIBLE_DEVICES through gpus=[1], with memory growth).

Note: the candidate selection reads the store and the manifest written by the
contamination-detection driver of the research repository, which is not part of this
release. To precompute masks for an entry from micrographs alone, use
save_fullset_triangular_masks.py.
"""
import argparse, os, re, sys, glob, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env
import filter_star_by_contamination as fsc

UID = re.compile(r"^\d{12,}_")            # the uid prefix on CryoPPP ground truth


def gt_index(eid):
    _, rows = fsc.parse_star(env.ground_truth_star(eid))
    d = {}
    for (_, m, x, y) in rows:
        d.setdefault(UID.sub("", m), []).append((x, y))   # line up on the basename (with .mrc)
    return d


def candidates(official_root, min_off_rm):
    """Read the meta of the released-arm masks and collect, per entry, the micrographs whose
    released ground-truth removal count is at or above the threshold."""
    by_id = {}
    for f in glob.glob(os.path.join(official_root, "*", "*_mask.npz")):
        meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        if meta["n_gt_in_contam"] < min_off_rm:
            continue
        by_id.setdefault(meta["empiar_id"], []).append(meta)
    return by_id


def candidates_negative(manifest_path):
    """Collect the has_anomaly==0 (clean) micrographs of the manifest, per entry.
    The released arm stores no mask for a negative, so these come from the manifest rather
    than from the mask store."""
    import csv
    by_id = {}
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            if str(r["has_anomaly"]).strip() != "0":
                continue
            by_id.setdefault(r["empiar_id"], []).append({
                "empiar_id": r["empiar_id"], "dist_class": r["dist_class"],
                "micrograph": r["micrograph"],
                "n_gt_in_contam": int(float(r.get("n_gt_in_contam") or 0)),
                "has_anomaly": 0})
    return by_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-off-rm", type=int, default=30,
                    help="take the micrographs whose anomaly meta has n_gt_in_contam at or above this")
    ap.add_argument("--ids", default="", help="comma-separated ids to restrict to (default: all)")
    ap.add_argument("--include-negative", action="store_true",
                    help="also include the manifest's has_anomaly==0 (clean) micrographs")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--deep-thr", type=float, default=0.5)
    ap.add_argument("--model", default=None,
                    help="model .h5 (default: $RAPICK_DATA/checkpoints/%s)" % env.MODEL_BASENAME)
    ap.add_argument("--official-root", default=None,
                    help="store of released-arm masks (default: $RAPICK_WORK/masks_official)")
    ap.add_argument("--manifest", default=None,
                    help="manifest CSV read by --include-negative (default: $RAPICK_WORK/manifest.csv)")
    ap.add_argument("--out-root", default=None,
                    help="triangular mask store (default: $RAPICK_WORK/masks)")
    args = ap.parse_args()

    model = env.resolve_model(args.model)
    official_root = args.official_root or env.official_mask_store_root()
    out_root = args.out_root or env.mask_store_root()

    env._ensure_modern_ptxas()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import cv2
    import micrograph_cleaner_em as mce
    from triangular_mask import build_extractor, extract_blended
    from micrograph_cleaner_em.preprocessMic import preprocessMic

    by_id = candidates(official_root, args.min_off_rm)
    if args.include_negative:
        manifest = args.manifest or os.path.join(env.work_root(), "manifest.csv")
        for k, v in candidates_negative(manifest).items():
            by_id.setdefault(k, []).extend(v)
    only = set(x for x in args.ids.split(",") if x)
    ids = sorted(k for k in by_id if not only or k in only)
    total = sum(len(by_id[i]) for i in ids)
    print("[cfg] %d ids, %d mics (n_gt_in_contam>=%d)" % (len(ids), total, args.min_off_rm))

    # The extractor does not depend on the box size, so build it once (only preprocessMic's
    # box changes per entry).
    mp = mce.MaskPredictor(180, deepLearningModelFname=model, gpus=[args.gpu])
    extractor = build_extractor(mp.model)
    t0 = time.time()
    done = skip = 0
    for eid in ids:
        gt = gt_index(eid)
        box = env.box_size_of(eid)
        mic_dir = env.annotated_mic_dir(eid)
        out_dir = os.path.join(out_root, eid)
        os.makedirs(out_dir, exist_ok=True)
        for meta in by_id[eid]:
            mic = meta["micrograph"]                       # basename (with .mrc)
            dist = meta["dist_class"]
            out = os.path.join(out_dir, os.path.splitext(mic)[0] + "_tri.npz")
            if os.path.exists(out):
                skip += 1
                continue
            try:
                image = fsc.load_micrograph(os.path.join(mic_dir, mic))
                _, trim, _, _ = extract_blended(extractor, preprocessMic(image, box), 2, 1)
                trim = trim.astype(np.float32)
                picks = gt.get(mic, [])
                tri_full = cv2.resize(trim, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                n_tri = sum(1 for b in fsc.keep_flags(tri_full, picks, args.deep_thr) if not b)
                np.savez_compressed(
                    out, tri=trim.astype(np.float16),
                    meta=json.dumps({
                        "empiar_id": eid, "micrograph": mic, "dist_class": dist,
                        "box_size": box, "model_h": int(trim.shape[0]), "model_w": int(trim.shape[1]),
                        "height": int(image.shape[0]), "width": int(image.shape[1]),
                        "deep_thr": args.deep_thr, "n_gt": len(picks),
                        "n_gt_in_tri_contam": int(n_tri),
                        "n_gt_in_contam": int(meta["n_gt_in_contam"]),
                        "has_anomaly": int(meta.get("has_anomaly", 1))}))
                done += 1
            except Exception as e:  # noqa: BLE001
                print("  ERR %s/%s: %s" % (eid, mic[:30], str(e)[:80]))
            if (done + skip) % 10 == 0:
                print("[%d/%d] %.0fs (done=%d skip=%d)" % (done + skip, total, time.time() - t0, done, skip))
    mp.close()
    print("[done] %.0fs  saved=%d skipped(existing)=%d -> %s" % (time.time() - t0, done, skip, out_root))


if __name__ == "__main__":
    main()
