#!/usr/bin/env python3
"""filter_star_by_contamination.py -- from a picker's standardized star, write a star with
the particles that landed on MicrographCleaner's contamination (anomaly) regions removed.

This is the **released post-processing** arm: the mask comes from upstream's
`predictMask` (uniform averaging of the overlapping windows, `fixJumpInBorders` seam
repair, 8-rotation averaging). It exists for the comparison in Sec. S3 of the paper;
the production filter is filter_star_triangular.py.

For the reconstruction-aware particle picking experiments: of a picker's full-set picks,
drop the particles that sit on contamination such as carbon, ice or aggregates, and run
a 3D reconstruction on that clean stack to see whether the resolution improves.

------------------------------------------------------------------------------
What it does
------------------------------------------------------------------------------
1. Read a standardized star (top-left origin, integer coordinates at mrc scale;
   = the CryoPPP ground-truth frame), grouped by micrograph.
2. Run `MaskPredictor` on each micrograph to get a 0..1 contamination mask.
3. Decide whether each pick (x,y) falls inside contamination with
   `flipud(mask)[round(y), round(x)] >= deep_thr` (both are in the same top-left
   frame, so the same expression works; this is also consistent with `y_flip: true`
   in the dataset configuration).
4. Write `*_clean.star`, keeping only the particles that are not on contamination, with
   the same header/columns as the input. The removals go to `*_removed.star`, the
   per-micrograph statistics to `filter_stats.csv` and the totals to `summary.json`.
5. For inspection, write an overlay JPG for the first few anomaly micrographs
   (green = kept, red = removed, red area = mask, yellow contour = the thresholded
   region, background = denoised).

The heavy GPU inference checkpoints the per-micrograph decision to `decisions.jsonl`
as it goes, so it can resume. The star is assembled from the decisions at the end, so
a resumed run reproduces the star deterministically.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
  python filter_star_by_contamination.py \
      --star  "$RAPICK_WORK/picks/10532/cryotransformer.star" \
      --mic-dir "$RAPICK_DATA/cryoppp_fullset/10532/micrographs" \
      --empiar-id 10532 --gpu 0 \
      --out-dir "$RAPICK_WORK/picks/10532"
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env  # noqa: E402


def parse_star(path):
    """Split a standardized star into (header_lines, rows).

    rows is a list[(raw_line, mic_basename, x, y)] in input order. header keeps
    everything up to the first data line (data_particles / loop_ / column declarations)
    verbatim, and it is reused unmodified when writing back.
    """
    lines = open(path).read().splitlines()

    # Find the loop_ that declares _rlnCoordinateX: its column name -> index map and
    # the first data line.
    data_start, cols = None, {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j, c = i + 1, {}
            while j < len(lines) and lines[j].strip().startswith("_"):
                c[lines[j].split()[0].lstrip("_")] = len(c)
                j += 1
            if any(k.startswith("rlnCoordinateX") for k in c):
                data_start, cols = j, c
                break
            i = j
        else:
            i += 1
    if data_start is None:
        sys.exit("no loop_ containing _rlnCoordinateX in the star: %s" % path)

    idx_mic = next(cols[k] for k in cols if k.startswith("rlnMicrographName"))
    idx_x = next(cols[k] for k in cols if k.startswith("rlnCoordinateX"))
    idx_y = next(cols[k] for k in cols if k.startswith("rlnCoordinateY"))

    header = lines[:data_start]
    rows = []
    for ln in lines[data_start:]:
        s = ln.strip()
        if not s or s.startswith(("_", "#", "data_")) or s == "loop_":
            continue
        t = s.split()
        if len(t) <= max(idx_mic, idx_x, idx_y):
            continue
        # Drop lines whose coordinates are not numeric (e.g. EMPIAR-10077's ground truth
        # quotes ImageName/MicrographName fields that contain spaces, so a bare split()
        # shifts the columns). A well-formed picker star is numeric on every line, so
        # this does not affect it.
        try:
            x, y = float(t[idx_x]), float(t[idx_y])
        except ValueError:
            continue
        rows.append((ln, os.path.basename(t[idx_mic]), x, y))
    return header, rows


def load_micrograph(path):
    """Read an mrc as a 2D float32. A movie stack (frames,H,W) is collapsed to one image
    by averaging the frames (contamination is large-scale structure, so an unaligned
    average is good enough)."""
    import mrcfile
    with mrcfile.open(path, permissive=True) as mrc:
        mic = np.asarray(mrc.data, dtype=np.float32)
    mic = np.squeeze(mic)
    if mic.ndim == 3:
        mic = mic.mean(axis=0)
    if mic.ndim != 2 or 0 in mic.shape:
        raise ValueError("not a 2D micrograph: shape=%s" % (mic.shape,))
    return mic


def keep_flags(mask, picks, deep_thr):
    """For picks=[(x,y),...], return a list of bools: "not inside contamination = keep".

    Same frame as the ground-truth containment count: flip the mask with flipud to line
    it up with the ground truth / star, and read (round(y), round(x)) at full resolution.
    A coordinate outside the frame is kept, which is the safe side.
    """
    maskf = np.flipud(mask)
    h, w = maskf.shape
    flags = []
    for x, y in picks:
        xi, yi = int(round(x)), int(round(y))
        in_contam = (0 <= yi < h and 0 <= xi < w and maskf[yi, xi] >= deep_thr)
        flags.append(not in_contam)
    return flags


def render_validation(mic, mask, picks, flags, deep_thr, box, max_out_dim, alpha=0.5,
                      denoised_full=None):
    """Validation overlay: denoised background + mask (red) + threshold contour (yellow),
    with kept particles as green and removed particles as red circles.

    The background uses CryoSegNet's released full-res denoised JPG (denoised_full, in the
    flipud frame) when there is one, downscaled to the output resolution. Without one,
    mic is downscaled and denoised on the spot with `overlay_panel.denoise_flip_frame` -- that
    denoises after the downsample, so it is blurrier than the full-res JPG.
    """
    import cv2

    h, w = mic.shape
    s = min(1.0, max_out_dim / float(max(h, w))) if max_out_dim else 1.0
    out_w, out_h = max(1, int(round(w * s))), max(1, int(round(h * s)))

    if denoised_full is not None:                                   # reuse CryoSegNet's full-res denoise (sharp)
        bg = cv2.resize(denoised_full, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:                                                           # fallback: downscale then denoise (blurry)
        bg = env.denoise_flip_frame(cv2.resize(mic, (out_w, out_h), interpolation=cv2.INTER_AREA))
    bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR).astype(np.float32)

    small_mask = np.flipud(cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR))
    m = np.clip(small_mask.astype(np.float32), 0.0, 1.0)
    a = (alpha * m)[..., None]
    red = np.zeros_like(bgr); red[..., 2] = 255.0
    img = (bgr * (1.0 - a) + red * a).astype(np.uint8)

    binm = (m >= deep_thr).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)                 # yellow: the contamination boundary

    r = max(2, int(round(box * s / 2)))
    for (x, y), keep in zip(picks, flags):
        cx, cy = int(round(x * s)), int(round(y * s))
        color = (0, 220, 0) if keep else (0, 0, 255)                      # green = kept / red = removed
        cv2.circle(img, (cx, cy), r, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star", required=True, help="input standardized star")
    ap.add_argument("--mic-dir", required=True, help="directory of full-set micrographs (.mrc)")
    ap.add_argument("--empiar-id", required=True, help="EMPIAR id, used to pick the boxSize")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: $RAPICK_WORK/picks/<empiar-id>)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--deep-thr", type=float, default=0.5,
                    help="mask threshold above which a pixel counts as contamination "
                         "(0.5, the same as micrograph_cleaner's own default)")
    ap.add_argument("--box-size", type=int, default=None, help="override MaskPredictor's boxSize")
    ap.add_argument("--model", type=str, default=None,
                    help="model .h5 (default: $RAPICK_DATA/checkpoints/%s)" % env.MODEL_BASENAME)
    ap.add_argument("--overlay-limit", type=int, default=6,
                    help="how many anomaly micrographs get a validation overlay (0 disables it)")
    ap.add_argument("--denoised-root", type=str, default=None,
                    help="root of CryoSegNet's released denoised train/test set. When given or "
                         "detected, the validation overlay uses its full-res JPG as the background "
                         "(default: $RAPICK_DATA/cryosegnet_dataset when it exists)")
    ap.add_argument("--no-denoised", action="store_true",
                    help="ignore denoised-root and always denoise on the spot after downscaling "
                         "(blurrier background)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N micrographs (for smoke tests; the star is "
                         "still assembled in full, and unprocessed picks are kept)")
    ap.add_argument("--max-out-dim", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true", help="ignore the decisions and redo everything")
    args = ap.parse_args()

    env._ensure_modern_ptxas()      # put the CUDA-12 ptxas first on PATH before importing TF
    import cv2  # noqa: F401
    import micrograph_cleaner_em as mce

    model = env.resolve_model(args.model)
    if not os.path.isfile(model):
        sys.exit("no model at: %s (run download_model.sh first)" % model)

    args.out_dir = args.out_dir or env.picks_dir_of(args.empiar_id)
    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, "validation_overlays")
    decisions_path = os.path.join(args.out_dir, "decisions.jsonl")

    header, rows = parse_star(args.star)
    by_mic = {}                     # mic -> [row_index, ...] (input order)
    for ri, (_, mic, _, _) in enumerate(rows):
        by_mic.setdefault(mic, []).append(ri)
    print("[cfg] star=%s  picks=%d  mics=%d" % (args.star, len(rows), len(by_mic)))

    box = env.box_size_of(args.empiar_id, args.box_size)
    deep = args.deep_thr

    # Index CryoSegNet's full-res denoised JPGs per entry for the overlay background.
    # A micrograph without one makes render_validation fall back to denoising on the spot.
    denoised_root = None if args.no_denoised else (args.denoised_root or env.default_denoised_root())
    denoised_idx = env.build_denoised_index(denoised_root, args.empiar_id)
    print("[cfg] denoised_root=%s  (%d denoised JPGs available as overlay backgrounds)"
          % (denoised_root or "(none: denoise on the spot)", len(denoised_idx)))

    # resume: load the micrographs that were already decided.
    done = {}
    if os.path.isfile(decisions_path) and not args.overwrite:
        with open(decisions_path) as f:
            for line in f:
                d = json.loads(line)
                done[d["mic"]] = d
        # "error" is not a final decision but something to retry (it covers a transient
        # failure on a micrograph that is in fact readable). On resume it is processed
        # again, and since assemble reads the decisions last-wins, the corrected record
        # overwrites the old error.
        retry = [m for m, d in done.items() if d.get("status") == "error"]
        for m in retry:
            del done[m]
        print("[resume] %d micrographs already decided (%d error mics will be retried)"
              % (len(done) + len(retry), len(retry)))

    t0 = time.time()
    mp = mce.MaskPredictor(box, deepLearningModelFname=model, gpus=[args.gpu])
    print("[model] loaded in %.1fs  box=%d deep_thr=%.2f gpu=%d"
          % (time.time() - t0, box, deep, args.gpu))

    mode = "w" if args.overwrite else "a"
    dec_out = open(decisions_path, mode)
    n_overlay = 0
    t_run = time.time()
    mic_order = sorted(by_mic)
    if args.limit:
        mic_order = mic_order[:args.limit]
    try:
        for k, mic in enumerate(mic_order):
            if mic in done:
                continue
            idxs = by_mic[mic]
            picks = [(rows[ri][2], rows[ri][3]) for ri in idxs]
            path = os.path.join(args.mic_dir, mic)
            rec = {"mic": mic, "n_picks": len(picks)}
            try:
                image = load_micrograph(path)
                rec["H"], rec["W"] = int(image.shape[0]), int(image.shape[1])
                with env.suppress_stdout():
                    mask = mp.predictMask(image)
                flags = keep_flags(mask, picks, deep)
                rec["keep"] = [int(b) for b in flags]
                rec["n_removed"] = int(sum(1 for b in flags if not b))
                rec["contam_fraction"] = round(float(np.mean(mask >= deep)), 5)
                rec["status"] = "anomaly" if rec["contam_fraction"] >= 0.02 else "ok"

                # The validation overlay is decoration for a human. The decision
                # (keep/n_removed) is already settled above, so a failure to draw must
                # never affect which particles are kept or removed (independent try that
                # swallows it). The caption bar has been removed; summary/stats/decisions
                # carry that information.
                if n_overlay < args.overlay_limit and rec["n_removed"] > 0:
                    try:
                        os.makedirs(overlay_dir, exist_ok=True)
                        img = render_validation(image, mask, picks, flags, deep,
                                                box, args.max_out_dim,
                                                denoised_full=env.load_denoised(denoised_idx, mic))
                        cv2.imwrite(os.path.join(overlay_dir, os.path.splitext(mic)[0] + "_filter.jpg"),
                                    img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        n_overlay += 1
                    except Exception as e:  # noqa: BLE001 -- a failed overlay only warns, the decision stands
                        print("[warn] validation overlay failed for %s: %s" % (mic, e))
            except Exception as e:  # noqa: BLE001 -- one failure must not stop the run. Keep every pick, the safe side.
                rec["keep"] = [1] * len(picks)
                rec["n_removed"] = 0
                rec["status"] = "error"
                rec["error"] = ("%s" % e)[:200]

            dec_out.write(json.dumps(rec) + "\n"); dec_out.flush()
            if (k + 1) % 100 == 0:
                print("[%d/%d] %s  (%.1fs)" % (k + 1, len(mic_order), mic, time.time() - t_run))
    finally:
        mp.close()
        dec_out.close()

    assemble_and_report(args, header, rows, by_mic, decisions_path, box, deep)


def assemble_and_report(args, header, rows, by_mic, decisions_path, box, deep):
    """Write the clean/removed star, the stats and the summary from decisions.jsonl."""
    import csv

    keep_row = [True] * len(rows)     # a micrograph with no decision defaults to True (the safe side)
    per_mic = {}
    with open(decisions_path) as f:
        for line in f:
            d = json.loads(line)
            per_mic[d["mic"]] = d
            for ri, k in zip(by_mic.get(d["mic"], []), d.get("keep", [])):
                keep_row[ri] = bool(k)

    clean = os.path.join(args.out_dir, "cryotransformer_clean.star")
    removed = os.path.join(args.out_dir, "cryotransformer_removed.star")
    with open(clean, "w") as fc, open(removed, "w") as fr:
        fc.write("\n".join(header) + "\n")
        fr.write("\n".join(header) + "\n")
        n_keep = n_rm = 0
        for ri, (ln, *_ ) in enumerate(rows):
            if keep_row[ri]:
                fc.write(ln.rstrip("\n") + "\n"); n_keep += 1
            else:
                fr.write(ln.rstrip("\n") + "\n"); n_rm += 1

    # per-micrograph stats CSV.
    stats_csv = os.path.join(args.out_dir, "filter_stats.csv")
    n_err = n_anom = 0
    with open(stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["micrograph", "n_picks", "n_removed", "contam_fraction", "status"])
        for mic in sorted(by_mic):
            d = per_mic.get(mic, {})
            w.writerow([mic, d.get("n_picks", len(by_mic[mic])), d.get("n_removed", 0),
                        d.get("contam_fraction", ""), d.get("status", "missing")])
            if d.get("status") == "error":
                n_err += 1
            if d.get("status") == "anomaly":
                n_anom += 1

    summary = {
        "star": args.star, "empiar_id": args.empiar_id, "box_size": box, "deep_thr": deep,
        "n_micrographs": len(by_mic), "n_micrographs_anomaly": n_anom,
        "n_micrographs_error": n_err,
        "picks_total": len(rows), "picks_kept": n_keep, "picks_removed": n_rm,
        "removed_fraction": round(n_rm / len(rows), 5) if rows else 0.0,
        "clean_star": clean, "removed_star": removed, "stats_csv": stats_csv,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[done] picks kept=%d removed=%d (%.2f%%)  anomaly_mics=%d error_mics=%d"
          % (n_keep, n_rm, 100.0 * n_rm / max(1, len(rows)), n_anom, n_err))
    print("[out] %s" % clean)


if __name__ == "__main__":
    main()
