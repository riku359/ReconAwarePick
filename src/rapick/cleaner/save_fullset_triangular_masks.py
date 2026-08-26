#!/usr/bin/env python3
"""save_fullset_triangular_masks.py -- the full-set version of save_triangular_masks.py.

It computes the triangular-window blend mask (extract_blended) for every full-set
micrograph of the given entries and caches it as npz. With this built in advance, a later
full-set reconstruction can produce a cleaner-filtered star straight away with
filter_star_from_masks.py (no TensorFlow, no GPU) -- there is no need to redo the
triangular-window inference in this TF2 environment every time.

The only difference from save_triangular_masks.py is how the targets are chosen.
save_triangular_masks.py narrows the candidates by the released predictMask's
ground-truth containment meta (n_gt_in_contam), but most full-set micrographs have no
ground truth (only the 300 micrographs of the CryoPPP subset do). This script therefore
never looks at the ground truth and processes every full-set micrograph of the given
entries unconditionally.

The output format and directory are identical to save_triangular_masks.py's
(<out-root>/<id>/<mic>_tri.npz, tri: float16 at model scale, meta: json). The CryoPPP
subset micrographs are a true subset of the full set (confirmed to exist under the same
file names), so they can be appended to the same npz store, and whatever was already
computed for the subset is skipped on resume.

Run in the micrograph_cleaner environment. The GPU defaults to 1.
  python save_fullset_triangular_masks.py --ids 10093,10345 --gpu 1
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env                   # noqa: E402  dist_class_of / box_size_of / ptxas
import filter_star_by_contamination as fsc  # noqa: E402  load_micrograph


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="comma-separated EMPIAR ids")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--mic-root", default=None,
                    help="parent of <id>/micrographs (default: $RAPICK_DATA/cryoppp_fullset; "
                         "pass $RAPICK_DATA/cryoppp for the 300 annotated micrographs)")
    ap.add_argument("--out-root", default=None,
                    help="the same store as save_triangular_masks.py (default: $RAPICK_WORK/masks)")
    ap.add_argument("--model", default=None,
                    help="model .h5 (default: $RAPICK_DATA/checkpoints/%s)" % env.MODEL_BASENAME)
    ap.add_argument("--limit", type=int, default=None, help="cap on micrographs per entry (for testing)")
    args = ap.parse_args()

    model = env.resolve_model(args.model)
    mic_root = args.mic_root or env.fullset_root()
    out_root = args.out_root or env.mask_store_root()

    env._ensure_modern_ptxas()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import cv2                                              # noqa: F401  early import-failure check
    import keras                                             # noqa: F401
    import micrograph_cleaner_em as mce
    from triangular_mask import build_extractor, extract_blended
    from micrograph_cleaner_em.preprocessMic import preprocessMic

    ids = [x for x in args.ids.split(",") if x]
    if not ids:
        sys.exit("--ids is empty")

    # The extractor does not depend on the box size, so build it once (only preprocessMic's
    # box changes per entry).
    mp = mce.MaskPredictor(180, deepLearningModelFname=model, gpus=[args.gpu])
    extractor = build_extractor(mp.model)

    t0 = time.time()
    done = skip = err = 0
    for eid in ids:
        dist = env.dist_class_of(eid)
        box = env.box_size_of(eid)
        mic_dir = os.path.join(mic_root, eid, "micrographs")
        mics = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mic_dir, "*.mrc")))
        if not mics:
            print("[skip] id=%s: no .mrc in %s" % (eid, mic_dir))
            continue
        if args.limit:
            mics = mics[:args.limit]
        out_dir = os.path.join(out_root, eid)
        os.makedirs(out_dir, exist_ok=True)
        print("[cfg] id=%s dist=%s box=%d mics=%d -> %s" % (eid, dist, box, len(mics), out_dir))

        for mic in mics:
            out = os.path.join(out_dir, os.path.splitext(mic)[0] + "_tri.npz")
            if os.path.exists(out):
                skip += 1
                continue
            try:
                image = fsc.load_micrograph(os.path.join(mic_dir, mic))
                _, trim, _, _ = extract_blended(extractor, preprocessMic(image, box), 2, 1)
                trim = trim.astype(np.float32)
                np.savez_compressed(
                    out, tri=trim.astype(np.float16),
                    meta=json.dumps({
                        "empiar_id": eid, "micrograph": mic, "dist_class": dist,
                        "box_size": box, "model_h": int(trim.shape[0]), "model_w": int(trim.shape[1]),
                        "height": int(image.shape[0]), "width": int(image.shape[1]),
                        "source": "fullset"}))
                done += 1
            except Exception as e:  # noqa: BLE001
                print("  ERR %s/%s: %s" % (eid, mic[:30], str(e)[:80]))
                err += 1
            if (done + skip + err) % 100 == 0:
                print("[%d] %.0fs (done=%d skip=%d err=%d)" %
                      (done + skip + err, time.time() - t0, done, skip, err))
    mp.close()
    print("[done] %.0fs saved=%d skipped(existing)=%d error=%d -> %s" %
          (time.time() - t0, done, skip, err, out_root))


if __name__ == "__main__":
    main()
