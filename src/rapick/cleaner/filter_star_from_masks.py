#!/usr/bin/env python3
"""filter_star_from_masks.py -- narrow a star by looking up pre-computed triangular
masks (no inference).

The decision is identical to filter_star_triangular.py; the mask is read from
save_fullset_triangular_masks.py's npz instead of being recomputed. This is for the
feedback loop, which picks the same micrographs over and over: the mask depends only on
the micrograph and not on the picks, so there is no need to run MaskPredictor once per
round.

The reuse is not an approximation. What save_fullset_triangular_masks.py stores is the second
return value of `extract_blended(extractor, preprocessMic(image, box), 2, 1)`, the same
expression filter_star_triangular.py evaluates. The only difference is the rounding from
the float16 storage, whose step near 0.5 is about 0.0005.

Needs neither TensorFlow nor a GPU nor mrcfile. The npz meta carries the full-resolution
dimensions, so the resize target is known without opening the micrograph itself.

Usage:
    python filter_star_from_masks.py \
        --star <picks>.star --empiar-id 10081 \
        --mask-dir "$RAPICK_WORK/masks/10081" \
        --out-dir <round-dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env                    # box_size_of / mask_dir_of
import filter_star_by_contamination as fsc   # parse_star / keep_flags
import filter_star_triangular as fst         # assemble (keeps the output shape identical)


def mask_path_for(mask_dir, mic):
    """<mask-dir>/<stem>_tri.npz. The star's _rlnMicrographName comes with an extension."""
    stem = os.path.splitext(os.path.basename(mic))[0]
    return os.path.join(mask_dir, stem + "_tri.npz")


def full_res_mask(npz_path):
    """Restore a stored model-scale mask to full resolution. Returns (mask, meta)."""
    import cv2

    with np.load(npz_path, allow_pickle=False) as z:
        tri = z["tri"].astype(np.float32)                 # stored as float16 -> decide in float32
        meta = json.loads(str(z["meta"]))
    return cv2.resize(tri, (meta["width"], meta["height"]),
                      interpolation=cv2.INTER_LINEAR), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star", required=True)
    ap.add_argument("--mask-dir", default=None,
                    help="directory of save_fullset_triangular_masks.py's npz "
                         "(default: $RAPICK_WORK/masks/<empiar-id>)")
    ap.add_argument("--empiar-id", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: $RAPICK_WORK/picks/<empiar-id>)")
    ap.add_argument("--deep-thr", type=float, default=0.5)
    ap.add_argument("--box-size", type=int, default=None)
    ap.add_argument("--suffix", default="_tri")
    ap.add_argument("--star-prefix", default="cryotransformer")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.mask_dir = args.mask_dir or env.mask_dir_of(args.empiar_id)
    args.out_dir = args.out_dir or env.picks_dir_of(args.empiar_id)

    os.makedirs(args.out_dir, exist_ok=True)
    decisions_path = os.path.join(args.out_dir, "decisions%s.jsonl" % args.suffix)
    header, rows = fsc.parse_star(args.star)
    by_mic = {}
    for ri, (_, mic, _, _) in enumerate(rows):
        by_mic.setdefault(mic, []).append(ri)
    box = env.box_size_of(args.empiar_id, args.box_size)
    deep = args.deep_thr
    print("[cfg] star=%s picks=%d mics=%d masks=%s" % (args.star, len(rows), len(by_mic), args.mask_dir))

    # A missing mask is recorded as status=error and keeps every pick (the same safe side
    # as the existing filter). Failing here would stop a round because of a single missing
    # mask.
    done = {}
    if os.path.isfile(decisions_path) and not args.overwrite:
        with open(decisions_path) as f:
            for line in f:
                d = json.loads(line)
                done[d["mic"]] = d
        for m in [m for m, d in done.items() if d.get("status") == "error"]:
            del done[m]
        print("[resume] %d micrographs already decided" % len(done))

    t_run = time.time()
    with open(decisions_path, "w" if args.overwrite else "a") as dec_out:
        for k, mic in enumerate(sorted(by_mic)):
            if mic in done:
                continue
            picks = [(rows[ri][2], rows[ri][3]) for ri in by_mic[mic]]
            rec = {"mic": mic, "n_picks": len(picks)}
            try:
                mask, meta = full_res_mask(mask_path_for(args.mask_dir, mic))
                rec["H"], rec["W"] = meta["height"], meta["width"]
                flags = fsc.keep_flags(mask, picks, deep)     # the same centre-point test as the existing filter
                rec["keep"] = [int(b) for b in flags]
                rec["n_removed"] = int(sum(1 for b in flags if not b))
                rec["contam_fraction"] = round(float(np.mean(mask >= deep)), 5)
                rec["status"] = "anomaly" if rec["contam_fraction"] >= 0.02 else "ok"
            except Exception as e:  # noqa: BLE001
                rec["keep"] = [1] * len(picks)
                rec["n_removed"] = 0
                rec["status"] = "error"
                rec["error"] = ("%s" % e)[:200]
            dec_out.write(json.dumps(rec) + "\n")
            if (k + 1) % 100 == 0:
                print("[%d/%d] %s (%.1fs)" % (k + 1, len(by_mic), mic, time.time() - t_run))

    fst.assemble(args, header, rows, by_mic, decisions_path, box, deep)


if __name__ == "__main__":
    main()
