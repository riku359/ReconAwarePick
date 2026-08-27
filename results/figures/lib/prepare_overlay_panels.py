#!/usr/bin/env python3
"""Turn the raw overlay strips into the bare panels the deck builders place.

The overlays come out of the stage renderer with their descriptions burnt into a black
bar across the top, and the contamination overlays additionally carry a yellow
0.5-contour on the mask. Neither survives being shrunk to column width in a paper, so
both come off here and the description is set as deck text under the panel instead.

Four operations, all of them on the JPG, because the masks and denoised backgrounds
that produced it are pipeline outputs rather than figure inputs:

    strip_header   drop the leading rows that are the black bar
    drop_contour   inpaint the yellow contour away, leaving the red mask alpha
    split          cut a stage strip into its separate panels
    top_right      keep the upper-right quarter of a stage panel, so the circles are
                   large enough to read at the width the figure is set at

Three figures are cut from the same two kinds of strip, so they are cut here together
and each figure's README calls this with its own `--only`:

    pick_fates        Fig. 2, the five-panel stage strip of two entries
    cleaner_failure   Fig. 6, the two contamination overlays of EMPIAR-10532
    protocol          the picking and mask panels of the supplementary protocol figure,
                      which is drawn in TikZ in the manuscript over panels like these

    python prepare_overlay_panels.py --strips <dir> [--out <dir>] [--only pick_fates]

`--strips` holds the strips the renderers wrote, under the names below. `--out`
defaults to `$RAPICK_FIGURES_OUT/overlay_panels`, which is where the deck builders look.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_paths                                    # noqa: E402

# The stage strip is five equal-width panels hstacked in this order. Each one holds what
# still survives at that point in the pipeline, so the strip narrows left to right.
STAGE_PANELS = ["raw", "mask", "class2d", "select", "gt"]

# The full-set strip carries only the two panels the protocol figure needs, drawn by
# fullset_stage_overlay.py from the full-set arm's own STAR files.
FULLSET_PANELS = ["raw", "mask"]

# The protocol figure names the same two panels after the pipeline stage rather than the
# tool, and takes them from the strip of the block it draws them in.
PROTOCOL_PANELS = {"raw": "picks", "mask": "masked"}

# Which strip each figure is cut from. The names are what the renderers write.
STAGE_ENTRIES = ("10081", "10532")
CLEANER_STRIPS = ("cleaner_10532_typical", "cleaner_10532_inverted")

BAR_ROW_MEAN_MAX = 100  # bar rows sit under 90, micrograph rows over 130
BAR_ROW_RUN = 8         # bright rows in a row before the bar is called over


def top_right_quarter(image):
    """Keep the upper-right quarter of the frame, halving each side.

    The whole micrograph at one quarter of the text width leaves the picks a few points
    across, which is where the figure stops carrying information. The same corner is
    taken from every panel of a row, so the panels still show one field of view
    narrowing stage by stage. Halving both sides keeps the aspect ratio, so the deck
    layout does not change with the crop.
    """
    height, width = image.shape[:2]
    return image[: height // 2, width - width // 2:]


def strip_header(image):
    """Drop the burnt-in black title bar at the top of the frame.

    The bar is solid black with white glyphs, so its rows average far darker than any
    micrograph row. Scanning row means for the boundary finds it without hardcoding the
    bar height, which varies with the renderer's panel width. Counting near-black pixels
    instead would stop early on the rows the glyphs pass through. A single bright row is
    not enough either: on the contamination overlays the description is set large enough
    that one row of glyphs averages over the threshold, and cutting there left the rest
    of the bar as a black band along the top of the panel. The boundary is therefore the
    first row that is bright and stays bright for BAR_ROW_RUN rows.
    """
    row_means = image.mean(axis=(1, 2))
    if row_means[0] >= BAR_ROW_MEAN_MAX:
        return image
    bright = row_means >= BAR_ROW_MEAN_MAX
    runs = np.convolve(bright.astype(int), np.ones(BAR_ROW_RUN, int), mode="valid")
    first_content_row = int(np.argmax(runs == BAR_ROW_RUN))
    # A couple of rows past the boundary, so the bar's antialiased bottom edge does not
    # survive as a dark line along the top of the panel.
    return image[first_content_row + 3:]


def drop_contour(image):
    """Inpaint the yellow threshold contour out of a contamination overlay.

    The contour is drawn as pure yellow, but JPEG leaves a halo around it, so the mask
    is taken with a loose tolerance and then dilated before inpainting. Everything else
    in the frame is red mask, green or red circles, or grayscale, none of which is
    yellow.
    """
    blue, green, red = (image[..., i].astype(np.int16) for i in range(3))
    is_yellow = (green > 120) & (red > 120) & (green - blue > 55) & (red - blue > 55)
    mask = cv2.dilate(is_yellow.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(image, mask, 4, cv2.INPAINT_TELEA)


def read(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read {path}; --strips must hold the rendered strips")
    return image


def write(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  {path.name}  {image.shape[1]}x{image.shape[0]}")


def split_strip(source: Path, stages):
    """One cropped panel per stage of a strip, keyed by the stage each one shows."""
    body = strip_header(read(source))
    width = body.shape[1] // len(stages)
    return {stage: top_right_quarter(body[:, index * width:(index + 1) * width])
            for index, stage in enumerate(stages)}


def build_pick_fates(strips: Path, out: Path):
    print("stage overlays (Fig. 2)")
    for entry in STAGE_ENTRIES:
        panels = split_strip(strips / f"stage_{entry}_round1.jpg", STAGE_PANELS)
        for stage, panel in panels.items():
            write(out / f"stage_{entry}_{stage}.jpg", panel)


def build_cleaner_failure(strips: Path, out: Path):
    print("contamination overlays (Fig. 6)")
    for name in CLEANER_STRIPS:
        write(out / f"{name}.jpg", drop_contour(strip_header(read(strips / f"{name}.jpg"))))


def build_protocol(strips: Path, out: Path):
    print("protocol panels (the supplementary protocol figure, drawn in TikZ)")
    for block, source, stages in (
            ("loop", strips / "stage_10081_round1.jpg", STAGE_PANELS),
            ("full", strips / "stage_10081_fullset.jpg", FULLSET_PANELS)):
        panels = split_strip(source, stages)
        for stage, target in PROTOCOL_PANELS.items():
            write(out / "protocol" / f"{block}_{target}.jpg", panels[stage])


FIGURES = {"pick_fates": build_pick_fates,
           "cleaner_failure": build_cleaner_failure,
           "protocol": build_protocol}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strips", required=True, type=Path,
                        help="directory holding the rendered strips")
    parser.add_argument("--out", type=Path, default=None,
                        help="where the bare panels go "
                             "(default $RAPICK_FIGURES_OUT/overlay_panels)")
    parser.add_argument("--only", choices=sorted(FIGURES), action="append", default=[],
                        help="cut only this figure's panels; repeatable, default all")
    args = parser.parse_args()

    out = args.out.expanduser() if args.out else figure_paths.figures_out("overlay_panels")
    out.mkdir(parents=True, exist_ok=True)
    for name in (args.only or sorted(FIGURES)):
        FIGURES[name](args.strips.expanduser(), out)


if __name__ == "__main__":
    main()
