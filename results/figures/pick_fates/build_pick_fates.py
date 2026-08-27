#!/usr/bin/env python3
"""Fig. 2: four stages across, one row per entry, laid out as a one-slide deck.

The stage renderer burns a black title bar into every overlay. At column width that
text is unreadable and the bar wastes vertical space, so `lib/prepare_overlay_panels.py`
strips it and this places the bare panels with the stage name set as deck text
underneath.

The strip holds five panels and the figure uses four of them: the `after 2D
classification` panel is left out, and the per-panel counts are not printed, so the
figure reads as the qualitative narrowing rather than as a second copy of the particle
table.

The entry names are set sideways in a column left of the panels and the stage names
once under the bottom row, so neither costs a band of its own per row. That is what
keeps the slide at 7.01 in rather than 8.59. Sideways text is `vert270` on the text
box, which python-pptx has no setter for.

The figure is set at 0.8 of the text width, so its text shrinks by a further 1.25 on the
page. The two sizes below carry that factor, which puts them back at the print size the
single-column figure gets from the shared caption size.

    python build_pick_fates.py [--panels <dir>] [--out-dir <dir>]

Then export and crop, as for the pipeline figure:

    soffice --headless --convert-to pdf --outdir <dir> <dir>/pick_fates.pptx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402
import pptx_deck                                       # noqa: E402

ENTRIES = ("10081", "10532")

# The four panels the figure carries, in the order the funnel narrows.
STAGES = ["raw", "mask", "select", "gt"]
STAGE_CAPTIONS = {
    "raw": "raw picks",
    "mask": "after cleaner mask",
    "select": "after 2D selection",
    "gt": "Ground Truth",
}

ROW_LABEL_PT = 24
STAGE_CAPTION_PT = 22.5
STAGE_CAPTION_H = 0.43  # the stage names fit on one line at one panel width


def build(panels_dir, out_dir):
    slide_w = 13.333
    gap = 0.10
    label_w = 0.42        # the column the sideways entry names occupy
    label_gap = 0.10
    row_gap = 0.14
    margin = pptx_deck.MARGIN

    panels_w = slide_w - 2 * margin - label_w - label_gap
    panel_w = (panels_w - gap * (len(STAGES) - 1)) / len(STAGES)

    rows = []
    for entry in ENTRIES:
        aspect = pptx_deck.panel_aspect(f"stage_{entry}_{STAGES[0]}", panels_dir)
        rows.append((entry, panel_w * aspect))

    slide_h = (2 * margin
               + sum(height for _, height in rows)
               + row_gap * (len(rows) - 1)
               + STAGE_CAPTION_H)
    deck, slide = pptx_deck.new_deck(slide_w, slide_h)

    panels_left = margin + label_w + label_gap
    top = margin
    for entry, panel_h in rows:
        pptx_deck.add_text(slide, f"EMPIAR-{entry}", margin, top, label_w, panel_h,
                           ROW_LABEL_PT, bold=True, anchor=MSO_ANCHOR.MIDDLE,
                           vertical=True)
        for index, stage in enumerate(STAGES):
            left = panels_left + index * (panel_w + gap)
            slide.shapes.add_picture(
                str(pptx_deck.panel_path(f"stage_{entry}_{stage}", panels_dir)),
                Inches(left), Inches(top), Inches(panel_w), Inches(panel_h),
            )
        top += panel_h + row_gap

    caption_top = top - row_gap + 0.05
    for index, stage in enumerate(STAGES):
        pptx_deck.add_text(slide, STAGE_CAPTIONS[stage],
                           panels_left + index * (panel_w + gap), caption_top,
                           panel_w, STAGE_CAPTION_H, STAGE_CAPTION_PT,
                           color=pptx_deck.GRAY)

    pptx_deck.save(deck, out_dir, "pick_fates", slide_w, slide_h)


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
