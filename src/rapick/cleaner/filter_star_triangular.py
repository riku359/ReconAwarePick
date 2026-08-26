#!/usr/bin/env python3
"""filter_star_triangular.py -- the triangular-window mask version of
filter_star_by_contamination. This is the production filter (Sec. 3.3).

Instead of the released `predictMask`, the contamination mask comes from
**extract_blended (triangular-window blending, seam-corrected)**, and contaminated
particles are removed by **exactly the same centre-point test as the existing filter**,
`flipud(mask)[round(y),round(x)] >= deep_thr`. The output is `<prefix>_clean<suffix>.star`
and friends. The only difference from the released version's cryotransformer_clean.star
is how the mask is made (released predictMask vs the triangular window). It can resume
(decisions<suffix>.jsonl).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env                   # box_size_of / ptxas / model path
import filter_star_by_contamination as fsc  # parse_star / keep_flags / load_micrograph
from triangular_mask import build_extractor, extract_blended


def assemble(args, header, rows, by_mic, decisions_path, box, deep):
    import csv
    keep_row = [True] * len(rows)
    per_mic = {}
    with open(decisions_path) as f:
        for line in f:
            d = json.loads(line); per_mic[d["mic"]] = d
            for ri, kf in zip(by_mic.get(d["mic"], []), d.get("keep", [])):
                keep_row[ri] = bool(kf)
    pfx, sfx = args.star_prefix, args.suffix
    clean = os.path.join(args.out_dir, "%s_clean%s.star" % (pfx, sfx))
    removed = os.path.join(args.out_dir, "%s_removed%s.star" % (pfx, sfx))
    with open(clean, "w") as fc, open(removed, "w") as fr:
        fc.write("\n".join(header) + "\n"); fr.write("\n".join(header) + "\n")
        nk = nr = 0
        for ri, (ln, *_) in enumerate(rows):
            if keep_row[ri]:
                fc.write(ln.rstrip("\n") + "\n"); nk += 1
            else:
                fr.write(ln.rstrip("\n") + "\n"); nr += 1
    stats = os.path.join(args.out_dir, "filter_stats%s.csv" % sfx)
    n_err = n_anom = 0
    with open(stats, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["micrograph", "n_picks", "n_removed", "contam_fraction", "status"])
        for mic in sorted(by_mic):
            d = per_mic.get(mic, {})
            w.writerow([mic, d.get("n_picks", len(by_mic[mic])), d.get("n_removed", 0),
                        d.get("contam_fraction", ""), d.get("status", "missing")])
            n_err += d.get("status") == "error"; n_anom += d.get("status") == "anomaly"
    summary = {"star": args.star, "empiar_id": args.empiar_id, "box_size": box, "deep_thr": deep,
               "mask_method": "triangular_extract_blended(stride_factor=2)",
               "n_micrographs": len(by_mic), "n_micrographs_anomaly": n_anom, "n_micrographs_error": n_err,
               "picks_total": len(rows), "picks_kept": nk, "picks_removed": nr,
               "removed_fraction": round(nr / len(rows), 5) if rows else 0.0,
               "clean_star": clean, "removed_star": removed}
    with open(os.path.join(args.out_dir, "summary%s.json" % sfx), "w") as f:
        json.dump(summary, f, indent=2)
    print("[done] kept=%d removed=%d (%.2f%%) anomaly_mics=%d error=%d" %
          (nk, nr, 100.0 * nr / max(1, len(rows)), n_anom, n_err))
    print("[out] %s" % clean)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star", required=True)
    ap.add_argument("--mic-dir", required=True)
    ap.add_argument("--empiar-id", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: $RAPICK_WORK/picks/<empiar-id>)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--deep-thr", type=float, default=0.5)
    ap.add_argument("--box-size", type=int, default=None)
    ap.add_argument("--model", default=None,
                    help="model .h5 (default: $RAPICK_DATA/checkpoints/%s)" % env.MODEL_BASENAME)
    ap.add_argument("--suffix", default="_tri")
    ap.add_argument("--star-prefix", default="cryotransformer")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = env.resolve_model(args.model)
    args.out_dir = args.out_dir or env.picks_dir_of(args.empiar_id)

    env._ensure_modern_ptxas()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import cv2, mrcfile, keras
    from micrograph_cleaner_em.preprocessMic import preprocessMic

    os.makedirs(args.out_dir, exist_ok=True)
    decisions_path = os.path.join(args.out_dir, "decisions%s.jsonl" % args.suffix)
    header, rows = fsc.parse_star(args.star)
    by_mic = {}
    for ri, (_, mic, _, _) in enumerate(rows):
        by_mic.setdefault(mic, []).append(ri)
    print("[cfg] star=%s picks=%d mics=%d" % (args.star, len(rows), len(by_mic)))
    box = env.box_size_of(args.empiar_id, args.box_size)
    deep = args.deep_thr

    done = {}
    if os.path.isfile(decisions_path) and not args.overwrite:
        with open(decisions_path) as f:
            for line in f:
                d = json.loads(line); done[d["mic"]] = d
        for m in [m for m, d in done.items() if d.get("status") == "error"]:
            del done[m]
        print("[resume] %d micrographs already decided" % len(done))

    t0 = time.time()
    full = keras.models.load_model(model, custom_objects={"LeakyReLU": keras.layers.LeakyReLU}, compile=False)
    extractor = build_extractor(full)
    print("[model] loaded %.1fs box=%d deep=%.2f gpu=%d method=triangular" % (time.time()-t0, box, deep, args.gpu))

    dec_out = open(decisions_path, "w" if args.overwrite else "a")
    mic_order = sorted(by_mic)
    if args.limit:
        mic_order = mic_order[:args.limit]
    t_run = time.time()
    try:
        for k, mic in enumerate(mic_order):
            if mic in done:
                continue
            idxs = by_mic[mic]
            picks = [(rows[ri][2], rows[ri][3]) for ri in idxs]
            rec = {"mic": mic, "n_picks": len(picks)}
            try:
                image = fsc.load_micrograph(os.path.join(args.mic_dir, mic))
                rec["H"], rec["W"] = int(image.shape[0]), int(image.shape[1])
                pre = preprocessMic(image, box)
                _, mask_model, _, _ = extract_blended(extractor, pre, 2, 1)   # triangular window, model scale
                mask = cv2.resize(mask_model.astype(np.float32), (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)                # back to full resolution
                flags = fsc.keep_flags(mask, picks, deep)                        # the same centre-point test as the existing filter
                rec["keep"] = [int(b) for b in flags]
                rec["n_removed"] = int(sum(1 for b in flags if not b))
                rec["contam_fraction"] = round(float(np.mean(mask >= deep)), 5)
                rec["status"] = "anomaly" if rec["contam_fraction"] >= 0.02 else "ok"
            except Exception as e:  # noqa: BLE001
                rec["keep"] = [1] * len(picks); rec["n_removed"] = 0
                rec["status"] = "error"; rec["error"] = ("%s" % e)[:200]
            dec_out.write(json.dumps(rec) + "\n"); dec_out.flush()
            if (k + 1) % 100 == 0:
                print("[%d/%d] %s (%.1fs)" % (k + 1, len(mic_order), mic, time.time()-t_run))
    finally:
        dec_out.close()
    assemble(args, header, rows, by_mic, decisions_path, box, deep)


if __name__ == "__main__":
    main()
