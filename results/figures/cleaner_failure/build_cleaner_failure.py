#!/usr/bin/env python3
"""Fig. 6: where the contamination mask fails on EMPIAR-10532.

Two mask overlays of the same entry side by side at single-column width: a typical case,
where the mask removes a handful of the candidates on the micrograph, and a failure case,
where the mask has inverted and removes nearly all of them.

The overlay renderer burns a black title bar into every frame and draws a yellow
0.5-contour on the mask. Neither survives being shrunk to column width, so
`lib/prepare_overlay_panels.py` strips the bar and inpaints the contour away, and this
places the bare panels with their descriptions set as deck text underneath.

The two counts in the captions are the removed/kept counts of the two micrographs the
panels show. They are properties of those two frames, so they do not change unless the
panels do.

    python build_cleaner_failure.py [--panels <dir>] [--out-dir <dir>]

Then export and crop, as for the pipeline figure:

    soffice --headless --convert-to pdf --outdir <dir> <dir>/cleaner_failure.pptx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402
import pptx_deck                                       # noqa: E402

PANELS = [
    ("cleaner_10532_typical", "(a) typical case: 6 of 324"),
    ("cleaner_10532_inverted", "(b) failure case: 181 of 183"),
]


def build(panels_dir, out_dir):
    slide_w = 6.5
    gap = 0.12
    margin = pptx_deck.MARGIN
    panel_w = (slide_w - 2 * margin - gap) / 2
    panel_h = panel_w * pptx_deck.panel_aspect(PANELS[0][0], panels_dir)
    slide_h = 2 * margin + panel_h + pptx_deck.CAPTION_H

    deck, slide = pptx_deck.new_deck(slide_w, slide_h)
    for index, (name, caption) in enumerate(PANELS):
        left = margin + index * (panel_w + gap)
        slide.shapes.add_picture(
            str(pptx_deck.panel_path(name, panels_dir)),
            Inches(left), Inches(margin), Inches(panel_w), Inches(panel_h),
        )
        pptx_deck.add_text(slide, caption, left, margin + panel_h + 0.05, panel_w,
                           pptx_deck.CAPTION_H, pptx_deck.CAPTION_PT,
                           color=pptx_deck.GRAY)

    pptx_deck.save(deck, out_dir, "cleaner_failure", slide_w, slide_h)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", type=Path, default=None,
                        help="the bare panels prepare_overlay_panels.py wrote "
                             "(default $RAPICK_FIGURES_OUT/overlay_panels)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the deck goes (default $RAPICK_FIGURES_OUT)")
    args = parser.parse_args()
    build(args.panels, args.out_dir or figure_paths.figures_out())


if __name__ == "__main__":
    main()
