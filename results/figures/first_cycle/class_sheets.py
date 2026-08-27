"""The class-average tile machinery the three sheets in this directory share.

Every tile is an image CryoSPARC renders for one class inside the `select_2D` job that
reads a classification, fetched with `lib/cs_fetch_assets.py`; nothing here is redrawn.
A tile is framed in the colour of the fate that classification's selection gives its
class, and the tiles are grouped by fate rather than left in class order, so the size of
the discarded block is what the eye lands on.

The palette is shared so that the first-cycle figure and the protocol sheets can be read
against each other. The green is the one the pipeline figure of the main paper gives a
kept class.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

N_CLASSES = 50
COLS = 10
BORDER = 11
GAP = 8
STACK_STEP = 40
STACK_PAD = 13    # the white margin between the overlapped tiles of a carry stack

# A class is framed by what happens to it after the classification the panel shows: it
# leaves the loop and is kept (`aside`), it goes into the next classification (`pool`),
# or it is discarded (`drop`).
COLORS = {"aside": (44, 160, 44), "pool": (44, 116, 200), "drop": (206, 62, 40)}


def frame(tile, color):
    """The class image with a coloured border around it."""
    out = Image.new("RGB", (tile.width + 2 * BORDER, tile.height + 2 * BORDER), color)
    out.paste(tile, (BORDER, BORDER))
    return out


def load(job, index, assets: Path):
    """One class tile, as `lib/cs_fetch_assets.py` names it."""
    path = Path(assets) / f"{job}__class2D_{index}.png"
    if not path.is_file():
        raise SystemExit(f"missing {path}; fetch the class images first with "
                         f"results/figures/lib/cs_fetch_assets.py")
    return Image.open(path).convert("RGB")


def row(tiles, dst: Path):
    """Framed tiles side by side on white, the layout one sheet uses."""
    cell_w, cell_h = tiles[0].width + GAP, tiles[0].height + GAP
    rows = -(-len(tiles) // COLS)
    page = Image.new("RGB", (min(COLS, len(tiles)) * cell_w - GAP, rows * cell_h - GAP),
                     (255, 255, 255))
    for n, tile in enumerate(tiles):
        page.paste(tile, ((n % COLS) * cell_w, (n // COLS) * cell_h))
    dst.parent.mkdir(parents=True, exist_ok=True)
    page.save(dst, optimize=True)
    return page


def stack(tiles, dst: Path):
    """Framed tiles overlapped on a diagonal, the layout one carry group uses.

    The first tile sits in front at the lower left and every later tile steps by
    STACK_STEP up and to the right behind it, so the group spends the width of one tile
    rather than one per tile. The corners the tiles leave empty stay transparent, which
    lets the arrow the group rides show through.

    Each tile but the last carries a white margin of STACK_PAD above and to its right,
    so its border does not touch the border of the tile behind it. Only those two sides
    take the margin: the tiles behind step up and to the right.
    """
    last = len(tiles) - 1
    page = Image.new("RGBA", (tiles[0].width + last * STACK_STEP,
                              tiles[0].height + last * STACK_STEP), (0, 0, 0, 0))
    for n in range(last, -1, -1):
        x, y = n * STACK_STEP, (last - n) * STACK_STEP
        if n < last:
            pad = Image.new("RGBA", (tiles[n].width + STACK_PAD,
                                     tiles[n].height + STACK_PAD), (255, 255, 255, 255))
            page.paste(pad, (x, y - STACK_PAD))
        page.paste(tiles[n], (x, y))
    dst.parent.mkdir(parents=True, exist_ok=True)
    page.save(dst, optimize=True)
    return page


def grouped_order(fates):
    """Class indices grouped by fate: kept, then pooled, then discarded."""
    return ([i for i in range(N_CLASSES) if fates[i] == "aside"]
            + [i for i in range(N_CLASSES) if fates[i] == "pool"]
            + [i for i in range(N_CLASSES) if fates[i] == "drop"])
