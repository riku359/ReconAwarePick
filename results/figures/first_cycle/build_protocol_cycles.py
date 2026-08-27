#!/usr/bin/env python3
"""The class averages of every CryoSift cycle, as the sheets and carry strips the
supplementary protocol figure sets on its arrows.

That figure is drawn in TikZ in the manuscript; this builds the eight sheets and the ten
short strips it places. It is `build_first_cycle_fig.py` run over every cycle rather
than over the first one alone, and shares its palette, border and spacing, so the two
figures can be read against each other.

Each sheet is the 50 class averages of one classification, framed in the colour of the
fate that classification's selection gives the class and grouped by that fate. The tiles
are the images CryoSPARC renders for each class inside the `select_2D` job that reads
the classification, so nothing here is redrawn. Which classes each selection kept comes
from the iterative selector's own record,
`$RAPICK_WORK/select2d/<project>_<class2d>_iter/state.json`.

Two blocks: `loop`, one round of the feedback loop on the 300 annotated micrographs, and
`full`, the full-set arm. `attractor` and `round0` read the same first classification:
the first holds its classes out of the loop, the second sends them into it, and what
neither takes is discarded.

NEEDS A LIVE CRYOSPARC INSTANCE, once, to fetch the tiles: fifty per job, eight jobs.
The uids below are the authors' instance and a fresh run produces different ones, so
override them with `--job <block>.<step>=<uid>` and read the kept-class lists out of
your own `state.json`.

    for J in J182 J183 J186 J191 J198; do
      SPEC=$(python3 -c "print(','.join('$J=class2D_%d.png' % i for i in range(50)))")
      python ../lib/cs_fetch_assets.py --project P1 --spec "$SPEC" --out /tmp/tiles
    done   # and J481 J482 J484 J486 J496

    python build_protocol_cycles.py --assets /tmp/tiles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import class_sheets                                    # noqa: E402
import figure_paths                                    # noqa: E402

# Which select_2D job renders the tiles of each classification, and which classes it
# kept. The uids are documented defaults; the class lists come from state.json.
STEPS = {
    "loop": {
        "attractor": ("J182", [0, 10, 19, 26, 30]),
        "round0": ("J183", [1, 7, 11, 13, 14, 17, 20, 21, 24, 27, 28, 31, 35, 36, 40, 41,
                            44, 46, 49]),
        "round1": ("J186", [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20,
                            22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39,
                            41, 43, 44, 45, 46, 47, 48, 49]),
        "round2": ("J191", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                            18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
        "final": ("J198", [0, 2, 5, 7, 10, 12, 13, 14, 17, 18, 19, 20, 22, 24, 25, 26, 27,
                           29, 30, 31, 32, 34, 39, 40, 41, 43, 44, 48, 49]),
    },
    "full": {
        "attractor": ("J481", [0, 10, 13, 19, 24, 30, 41, 49]),
        "round0": ("J482", [4, 7, 14, 17, 20, 26, 28, 31, 46]),
        "round1": ("J484", [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19,
                            20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37,
                            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
        "round2": ("J486", [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                            19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
        "final": ("J496", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 18, 19, 20, 23, 24, 26,
                           27, 28, 30, 31, 32, 33, 34, 37, 39, 40, 41, 42, 43, 44, 45, 46,
                           48, 49]),
    },
}

# The four panels of one block, in the order the stage runs them. Each is the
# classification named on the left, coloured by the selection that reads it.
PANELS = (("init",   "attractor", "round0"),
          ("cycle1", None,        "round1"),
          ("cycle2", None,        "round2"),
          ("final",  "final",     None))

# A few class averages to set on each arrow of the figure, so the arrow shows what
# travels along it rather than only where it goes. Each entry is (name, the panel the
# classes come from, their fate there); the classes are the first of that fate in class
# order, which keeps the choice reproducible. `aside` leaves the first classification for
# the last one and `kept` leaves the last one for the rest of the pipeline, so both are
# green; the three `pool` strips are the blue classes each classification hands to the
# next.
CARRY_N = 3
CARRIES = (("aside", "init",   "aside"),
           ("pool0", "init",   "pool"),
           ("pool1", "cycle1", "pool"),
           ("pool2", "cycle2", "pool"),
           ("kept",  "final",  "aside"))


def parse_job_overrides(items):
    """`["loop.round1=J9"]` -> `{("loop", "round1"): "J9"}`, checking the names."""
    out = {}
    for item in items or ():
        target, _, uid = item.partition("=")
        block, _, step = target.partition(".")
        if not uid or block not in STEPS or step not in STEPS[block]:
            raise SystemExit(
                f"--job expects <block>.<step>=<uid> with block in {sorted(STEPS)} and "
                f"step in {sorted(STEPS['loop'])}, got: {item}")
        out[(block, step)] = uid
    return out


def carry(job, fates, fate, assets: Path, dst: Path):
    """The first few classes of one fate, framed in its colour, as one diagonal stack."""
    chosen = [i for i in range(class_sheets.N_CLASSES) if fates[i] == fate][:CARRY_N]
    tiles = [class_sheets.frame(class_sheets.load(job, i, assets),
                                class_sheets.COLORS[fate]) for i in chosen]
    page = class_sheets.stack(tiles, dst)
    print("  %-24s %s  %s %s  %dx%d  %.0f kB"
          % (dst.name, job, fate, chosen, page.width, page.height,
             dst.stat().st_size / 1024))


def sheet(job, fates, assets: Path, dst: Path):
    """One classification, its tiles grouped by fate and framed in the fate's colour."""
    tiles = [class_sheets.frame(class_sheets.load(job, i, assets),
                                class_sheets.COLORS[fates[i]])
             for i in class_sheets.grouped_order(fates)]
    page = class_sheets.row(tiles, dst)
    counts = {f: sum(1 for v in fates.values() if v == f) for f in class_sheets.COLORS}
    print("  %-24s %s  aside %2d  pool %2d  drop %2d  %dx%d  %.0f kB"
          % (dst.name, job, counts["aside"], counts["pool"], counts["drop"],
             page.width, page.height, dst.stat().st_size / 1024))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, type=Path,
                    help="directory holding <job>__class2D_<i>.png for all eight selections")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the sheets go (default $RAPICK_FIGURES_OUT/protocol)")
    ap.add_argument("--job", action="append", default=[], metavar="BLOCK.STEP=UID",
                    help="override one select_2D job uid, e.g. loop.round1=J9")
    args = ap.parse_args()

    overrides = parse_job_overrides(args.job)
    out_dir = args.out_dir or figure_paths.figures_out("protocol")
    out_dir.mkdir(parents=True, exist_ok=True)

    for block, steps in STEPS.items():
        print(block)
        panels = {}
        for name, aside_step, pool_step in PANELS:
            fates = {i: "drop" for i in range(class_sheets.N_CLASSES)}
            job = None
            for step, fate in ((pool_step, "pool"), (aside_step, "aside")):
                if step is None:
                    continue
                job, kept = steps[step]
                job = overrides.get((block, step), job)
                for i in kept:
                    fates[i] = fate
            panels[name] = (job, fates)
            sheet(job, fates, args.assets, out_dir / f"{block}_{name}.png")
        for name, panel, fate in CARRIES:
            job, fates = panels[panel]
            carry(job, fates, fate, args.assets, out_dir / f"{block}_carry_{name}.png")


if __name__ == "__main__":
    main()
