#!/usr/bin/env python3
"""Crop three raw particles from each extraction render into one carry stack.

The stacks ride the two arrows of the supplementary protocol figure that leave a final
classification: into the fine-tune block on the loop side, and into ab-initio
reconstruction on the full set. Both stages consume particles rather than class
averages, so the arrows carry raw particles. The tiles are the middle row of CryoSPARC's
extraction render for the block, so the stacks stay crops of figures CryoSPARC rendered.

Frame colour and the diagonal layout follow the other carry stacks of
`build_protocol_cycles.py`, with the tiles resized to the class-tile size so the borders
print at the same thickness.

NEEDS A LIVE CRYOSPARC INSTANCE, once, to fetch the two extraction renders. The job uids
below are the authors' instance; a fresh run produces different ones.

    python ../lib/cs_fetch_assets.py --project P1 --out /tmp/extract \\
        --spec 'J173=extracted_particles,J474=extracted_particles'
    # rename each to <block>_extract_particles.jpg in the assets directory

    python build_teacher_strip.py --assets /tmp/extract
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import class_sheets                                    # noqa: E402
import figure_paths                                    # noqa: E402

GRID = 3          # the extraction render is a 3x3 grid of particles
ROW = 1           # the middle row, whose particles are the most clearly centred
TRIM = 5          # shave the white separators off the cell edges
TILE = 128        # the class tiles of the other carries, so the stacks scale alike

# block -> the extract job whose render the tiles are cropped from.
BLOCKS = {"loop": "J173", "full": "J474"}


def strip(block, job, assets: Path, out_dir: Path):
    source = assets / f"{block}_extract_particles.jpg"
    if not source.is_file():
        raise SystemExit(f"missing {source}; fetch the extraction render first")
    img = Image.open(source).convert("RGB")
    cw, ch = img.width / GRID, img.height / GRID
    tiles = []
    for col in range(GRID):
        box = (round(col * cw) + TRIM, round(ROW * ch) + TRIM,
               round((col + 1) * cw) - TRIM, round((ROW + 1) * ch) - TRIM)
        tiles.append(img.crop(box).resize((TILE, TILE), Image.LANCZOS))
    dst = out_dir / f"{block}_carry_teacher.png"
    page = class_sheets.stack(
        [class_sheets.frame(t, class_sheets.COLORS["aside"]) for t in tiles], dst)
    print("  %-24s %s row %d  %dx%d  %.0f kB"
          % (dst.name, job, ROW, page.width, page.height, dst.stat().st_size / 1024))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, type=Path,
                    help="directory holding <block>_extract_particles.jpg")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the stacks go (default $RAPICK_FIGURES_OUT/protocol)")
    for block, job in BLOCKS.items():
        ap.add_argument(f"--job-{block}", default=job,
                        help=f"the extract job the {block} tiles come from "
                             f"(default {job}, the authors' instance)")
    args = ap.parse_args()

    out_dir = args.out_dir or figure_paths.figures_out("protocol")
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in BLOCKS:
        strip(block, getattr(args, f"job_{block}"), args.assets, out_dir)


if __name__ == "__main__":
    main()
