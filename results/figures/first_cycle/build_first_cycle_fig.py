#!/usr/bin/env python3
"""Fig. S5: the 50 class averages of one entry, laid out by what the first CryoSift
cycle does to them.

CryoSPARC renders every class of a classification as its own image inside the
`select_2D` job that reads it, so the tiles here are its renders and nothing is redrawn.
The three fates come from the iterative selector's own record,
`$RAPICK_WORK/select2d/<project>_<class2d>_iter/state.json`, whose `attractor` and
`round0` steps name the classes each one keeps. Everything neither keeps is discarded
permanently.

Both entries shown are the `both` condition of the ablation table (contamination mask
plus 2D class selection), on the full micrograph set.

NEEDS A LIVE CRYOSPARC INSTANCE, once, to fetch the tiles. The job uids below are the
authors' instance; a fresh run produces the same chain with different uids, so pass your
own with `--job-<entry>` and read the kept-class lists out of your own `state.json`.

    # fetch the 50 class images of one selection
    SPEC=$(python3 -c "print(','.join('J115=class2D_%d.png' % i for i in range(50)))")
    python ../lib/cs_fetch_assets.py --project P1 --spec "$SPEC" --out /tmp/cls81

    python build_first_cycle_fig.py --assets-10081 /tmp/cls81 --assets-10345 /tmp/cls45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import class_sheets                                    # noqa: E402
import figure_paths                                    # noqa: E402

# EMPIAR entry -> (the select_2D job that renders the tiles, the classes set aside, the
# classes kept for the next cycle). The job uids are documented defaults from the
# authors' instance; the class lists are copied from the selector's own state.json.
ENTRIES = {
    "10081": dict(
        job="J115",
        aside=[0, 7, 12, 13, 19, 24, 26, 27, 28, 30, 44, 46],
        pool=[1, 4, 5, 11, 14, 20, 23, 25, 31, 35, 37, 41, 43, 48, 49],
    ),
    "10345": dict(
        job="J225",
        aside=[44],
        pool=[16, 18, 19, 33, 41],
    ),
}


def build(entry, spec, assets: Path, out_dir: Path) -> Path:
    n = class_sheets.N_CLASSES
    fates = {i: "drop" for i in range(n)}
    for i in spec["aside"]:
        fates[i] = "aside"
    for i in spec["pool"]:
        fates[i] = "pool"

    tiles = [class_sheets.frame(class_sheets.load(spec["job"], i, assets),
                                class_sheets.COLORS[fates[i]])
             for i in class_sheets.grouped_order(fates)]

    dst = out_dir / f"first_cycle_{entry}.png"
    sheet = class_sheets.row(tiles, dst)
    kept = len(spec["aside"]) + len(spec["pool"])
    print("%s  %d set aside, %d kept, %d discarded  %dx%d  %.1f kB"
          % (entry, len(spec["aside"]), len(spec["pool"]), n - kept,
             sheet.width, sheet.height, dst.stat().st_size / 1024))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for entry, spec in ENTRIES.items():
        ap.add_argument(f"--assets-{entry}", required=True,
                        help=f"directory holding <job>__class2D_*.png for EMPIAR-{entry}")
        ap.add_argument(f"--job-{entry}", default=spec["job"],
                        help=f"the select_2D job that rendered them "
                             f"(default {spec['job']}, the authors' instance)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the sheets go (default $RAPICK_FIGURES_OUT/selection)")
    args = ap.parse_args()

    out_dir = args.out_dir or figure_paths.figures_out("selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry, spec in ENTRIES.items():
        spec = dict(spec, job=getattr(args, f"job_{entry}"))
        build(entry, spec, Path(getattr(args, f"assets_{entry}")), out_dir)


if __name__ == "__main__":
    main()
