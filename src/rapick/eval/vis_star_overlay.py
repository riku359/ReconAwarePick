#!/usr/bin/env python3
"""vis_star_overlay.py -- draw a GT-aligned STAR on its micrograph and save the image.

An eyeball tool: it takes the very **GT-aligned STAR (top-left origin)** that
`calc_common_2d_metrics.py` scores and draws a box at each of its coordinates on the
background micrograph. STAR resolution and STAR reading are reused from the scorer, so
the picture shows the coordinates that were actually scored.

**The one thing to get right**: a native picker STAR has a bottom origin and must be
drawn as cy = H - Y. The input here is a **GT-aligned STAR (top-left origin)**, so it is
**not flipped** (cy = Y). Feeding a native STAR to this tool draws every box in the
wrong place.

The background is the full-scale jpg under $RAPICK_TEST_DATA/<id>/images/ (same
resolution as the mrc, so the mrc-scale integer coordinates land directly on it).

------------------------------------------------------------------------------
Usage (needs numpy + opencv-python)
------------------------------------------------------------------------------
  # resolve the STAR from a picker name, exactly as calc_common_2d_metrics does:
  python vis_star_overlay.py --picker cryolo --id 10081 --n 2

  # point at any STAR directly (file / per-micrograph directory / glob):
  python vis_star_overlay.py --star $RAPICK_WORK/picks/10093/topaz --id 10093 --n 3

  # visualise the CryoPPP annotations (green, tag=<id>_gt):
  python vis_star_overlay.py --gt --id 10081 --mic HCN1apo_0008_2xaligned

  # name a micrograph explicitly (extension and path ignored, matched on the stem):
  python vis_star_overlay.py --picker cryosegnet --id 10081 --mic HCN1apo_0008_2xaligned

Output: $RAPICK_WORK/overlays/<id>_<tag>_<mic>.jpg (change it with --out).
Colours come from SOURCE_COLORS: gt=green / cryolo=red / topaz=cyan /
cryotransformer=magenta / cryosegnet=yellow. --color overrides.

Environment: RAPICK_TEST_DATA (background jpgs), RAPICK_WORK (default output
directory), RAPICK_DATA (annotations, via the scorer). See docs/CONFIGURATION.md.
None of them has a default; a missing variable is an error naming it.
"""
import argparse
import glob
import os
import sys

import cv2

# Reuse the scoring code as the single source of truth (STAR resolution, reading,
# particle diameter).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calc_common_2d_metrics as ccm

# Colour name -> BGR (cv2 orders channels BGR).
COLORS = {"green": (0, 255, 0), "yellow": (0, 255, 255), "red": (0, 0, 255),
          "cyan": (255, 255, 0), "magenta": (255, 0, 255), "orange": (0, 165, 255)}

# Source (annotations / each picker) -> box colour. Annotations are green and each
# picker gets its own colour, so overlays rendered separately can be told apart at a
# glance.
SOURCE_COLORS = {
    "gt":              COLORS["green"],
    "cryolo":          COLORS["red"],
    "topaz":           COLORS["cyan"],
    "cryotransformer": COLORS["magenta"],
    "cryosegnet":      COLORS["yellow"],
}
DEFAULT_COLOR = COLORS["green"]   # default when --star names an arbitrary STAR


def test_data_root():
    """$RAPICK_TEST_DATA -- root holding <EMPIAR id>/images/ background micrographs."""
    value = os.environ.get("RAPICK_TEST_DATA")
    if not value:
        raise SystemExit(
            "RAPICK_TEST_DATA is not set; it must point at the root holding "
            "<EMPIAR id>/images/. See docs/CONFIGURATION.md.")
    return os.path.expanduser(value)


def default_out_dir():
    """$RAPICK_WORK/overlays -- where the rendered jpgs go unless --out says otherwise."""
    value = os.environ.get("RAPICK_WORK")
    if not value:
        raise SystemExit(
            "RAPICK_WORK is not set; it must point at the pipeline's output tree, or "
            "pass --out. See docs/CONFIGURATION.md.")
    return os.path.join(os.path.expanduser(value), "overlays")


def mic_key(name):
    """jpg/star file name -> comparison key; ccm.normalize_mic_name, extension-agnostic.

    ccm.normalize_mic_name only strips '.mrc', so it cannot be used on a jpg. Here the
    extension is dropped first and the same normalization applied, which puts this key
    and the STAR-side key (ccm) in the same namespace.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    return ccm.normalize_mic_name(stem + ".mrc")


def background_index(eid):
    """{mic_key: full jpg path}, scanning $RAPICK_TEST_DATA/<id>/images/*.jpg."""
    idx = {}
    for p in sorted(glob.glob(os.path.join(test_data_root(), str(eid), "images", "*.jpg"))):
        idx[mic_key(p)] = p
    return idx


def load_background(jpg_path):
    """Read the grayscale jpg as BGR (so coloured boxes can be drawn); None if missing."""
    gray = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_boxes(img, points, box_px, thickness, color):
    """Draw a square box at each center=(x, y). **No Y flip** (GT-aligned is top-left)."""
    r = box_px // 2
    for (x, y) in points:
        cx, cy = int(round(x)), int(round(y))
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness)


def render_micrograph(mic, jpg_path, points, diam, tag, out_dir, color=DEFAULT_COLOR):
    """Draw one micrograph's picks, save the image, and return the pick count."""
    bg = load_background(jpg_path)
    if bg is None:
        print(f"  skip (unreadable jpg): {jpg_path}")
        return None
    draw_boxes(bg, points, int(diam), max(2, int(diam) // 18), color)
    out_path = os.path.join(out_dir, f"{tag}_{mic}.jpg")
    cv2.imwrite(out_path, bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  {mic}: picks={len(points)}  -> {out_path}")
    return len(points)


def select_mics(requested, pred, bg_idx, n):
    """Which mic keys to draw: the named ones, else the n micrographs that have both
    predictions and a jpg, most picks first."""
    if requested:
        keys = [mic_key(m) for m in requested]
        missing = [m for m, k in zip(requested, keys) if k not in bg_idx]
        if missing:
            print(f"  warn: no background jpg for: {missing}", file=sys.stderr)
        return [k for k in keys if k in bg_idx]
    have = [k for k in pred if k in bg_idx]
    have.sort(key=lambda k: len(pred[k]), reverse=True)   # a busy micrograph shows more
    return have[:n]


def resolve_star(args):
    """Decide the STAR path and the display tag from --picker / --gt / --star."""
    if args.gt:
        return ccm.gt_path_for(args.id), f"{args.id}_gt"
    if args.picker:
        return ccm.picker_pred_path(args.picker, args.id), f"{args.id}_{args.picker}"
    stem = os.path.basename(os.path.normpath(args.star)).replace(".star", "") or "star"
    return args.star, f"{args.id}_{stem}"


def color_for(args):
    """Box colour: explicit --color > the source's default (gt / picker) > DEFAULT_COLOR."""
    if args.color:
        return COLORS[args.color]
    if args.gt:
        return SOURCE_COLORS["gt"]
    if args.picker:
        return SOURCE_COLORS[args.picker]
    return DEFAULT_COLOR


def load_pred(star_path, only_mics):
    """{mic_key: [(x, y), ...]}; with --mic on a per-micrograph directory, open only
    the matching files.

    crYOLO and Topaz write one STAR per micrograph -- 300 of them -- and opening all of
    them over a network filesystem is slow. When the micrographs to draw are already
    known, only those stems need reading (a single-file picker opens one file anyway, so
    it is delegated straight to ccm.load_pred).
    """
    if only_mics and os.path.isdir(star_path):
        wanted = set(only_mics)
        merged = {}
        for fp in sorted(glob.glob(os.path.join(star_path, "*.star"))):
            if mic_key(fp) not in wanted:
                continue
            for mic, pts in ccm.load_star_points(fp).items():
                merged.setdefault(mic, []).extend(pts)
        return merged
    return ccm.load_pred(star_path)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--picker", choices=ccm.PICKERS,
                     help="resolve from $RAPICK_WORK/picks/ (same rule as the scorer)")
    src.add_argument("--gt", action="store_true",
                     help="visualise the CryoPPP annotations (selected.star)")
    src.add_argument("--star",
                     help="a GT-aligned STAR (file / per-micrograph directory / glob)")
    ap.add_argument("--id", type=int, required=True,
                    help="EMPIAR ID (resolves the diameter and the background)")
    ap.add_argument("--diam", type=float, default=None,
                    help="particle diameter px (box size); default comes from the table")
    ap.add_argument("--mic", nargs="+", default=None,
                    help="name the micrographs to draw (matched on the stem)")
    ap.add_argument("--n", type=int, default=2,
                    help="how many micrographs to sample when none are named")
    ap.add_argument("--color", choices=sorted(COLORS), default=None,
                    help="override the box colour (default SOURCE_COLORS: gt=green, "
                         "one colour per picker)")
    ap.add_argument("--out", default=None,
                    help="output directory (default $RAPICK_WORK/overlays)")
    return ap.parse_args()


def main():
    args = parse_args()
    star_path, tag = resolve_star(args)
    diam = args.diam if args.diam is not None else ccm.DIAMETERS.get(args.id)
    if diam is None:
        raise SystemExit(f"no diameter registered for EMPIAR {args.id}; pass --diam.")

    color = color_for(args)
    only_mics = [mic_key(m) for m in args.mic] if args.mic else None
    pred = load_pred(star_path, only_mics)
    bg_idx = background_index(args.id)
    if not bg_idx:
        raise SystemExit(
            f"no background jpg found: {test_data_root()}/{args.id}/images/*.jpg")

    mics = select_mics(args.mic, pred, bg_idx, args.n)
    if not mics:
        raise SystemExit(
            "nothing to draw (no micrograph has both predictions and a background jpg).")

    out_dir = args.out or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"STAR: {star_path}\n  diam={diam}  mics={len(mics)}")
    for mic in mics:
        render_micrograph(mic, bg_idx[mic], pred.get(mic, []), diam, tag, out_dir, color)
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
