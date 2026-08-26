#!/usr/bin/env python
"""Figure the keep/drop of a manual select_2D and of CryoSift, in a shared order.

Each figure lays out every class of the class_2D, framing the classes that selector
kept in green and the ones it dropped in red. The order is the same in both figures
(ascending CryoSift score), so a tile whose frame colour differs between the two is a
class the selectors split on.

To see only the classes that differ, at a larger size, use plot_selection_diff.py.

Usage:
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.plot_selection_panel \\
        --job <class_2D uid> --manual-select <select_2D uid> \\
        --out results/figures/select2d/<empiar id>
"""

import argparse
from pathlib import Path

import numpy as np

from .cryosift_env import (
    DEFAULT_ENV_FILE,
    connect,
    read_env,
    require,
    resolve_job_dir,
    resolve_work_dir,
)
from .plot_selection_diff import class_averages, manual_kept_classes, read_scores

N_COLS = 10          # 50 classes fit a 10x5 grid, which suits half a slide

KEEP_COLOR = "tab:green"
DROP_COLOR = "tab:red"
DIFF_COLOR = "tab:orange"


TILE_IN = 0.72       # side of one tile, in inches
TITLE_IN = 0.45      # the two-line heading band left free at the top


def draw(png_path, images, order, kept, differing, title, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Margins are settled here. tight_layout leaves close to an inch of whitespace
    # under the heading band when the axes it packs are aspect-locked imshows.
    n_rows = int(np.ceil(len(order) / N_COLS))
    fig_h = TILE_IN * n_rows + TITLE_IN
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(TILE_IN * N_COLS, fig_h))
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.006,
                        top=1 - TITLE_IN / fig_h, wspace=0.05, hspace=0.05)
    for ax, idx in zip(axes.ravel(), order):
        ax.imshow(images[idx], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        is_kept = idx in kept
        for spine in ax.spines.values():
            spine.set_edgecolor(KEEP_COLOR if is_kept else DROP_COLOR)
            spine.set_linewidth(3.0 if idx in differing else 1.6)
        if idx in differing:
            ax.text(0.06, 0.94, "*", transform=ax.transAxes, ha="left", va="top",
                    fontsize=13, fontweight="bold", color=DIFF_COLOR)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")

    # Text drawn into the figure stays ASCII: matplotlib's default font carries no CJK
    # glyphs and renders them as tofu boxes.
    fig.suptitle(f"{title}\n{subtitle}", fontsize=9, y=0.995, va="top")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"wrote {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="class_2D job uid, e.g. J59")
    parser.add_argument("--manual-select", required=True,
                        help="job uid of the hand-curated select_2D, e.g. J211")
    parser.add_argument("--project", help="CryoSPARC project uid (default: CRYOSPARC_PROJECT)")
    parser.add_argument("--cutoff", type=float, default=3.5)
    parser.add_argument("--out", required=True, help="directory to write the PNGs into")
    parser.add_argument("--out-root",
                        help="parent to look for scores.csv under (default $RAPICK_WORK/select2d)")
    parser.add_argument("--env", default=None,
                        help=f"CryoSPARC credentials .env (default {DEFAULT_ENV_FILE})")
    args = parser.parse_args()

    env = read_env(args.env)
    project = args.project or require(env, "CRYOSPARC_PROJECT", args.env)
    cs = connect(env, args.env)
    job_dir = resolve_job_dir(cs, project, args.job)

    scores = read_scores(resolve_work_dir(args.out_root)
                         / f"{project}_{args.job}" / "scores.csv")
    manual_keep = manual_kept_classes(resolve_job_dir(cs, project, args.manual_select))
    cryosift_keep = {i for i, (score, _) in scores.items() if score < args.cutoff}
    differing = manual_keep ^ cryosift_keep

    images = class_averages(job_dir, args.job)
    order = sorted(scores, key=lambda i: scores[i][0])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [("manual", manual_keep, f"manual selection ({args.manual_select})"),
              ("cryosift", cryosift_keep, f"CryoSift selection (cutoff {args.cutoff})")]
    for name, kept, label in panels:
        n_particles = sum(scores[i][1] for i in kept)
        draw(out_dir / f"selection_{name}_{args.job}.png", images, order, kept, differing,
             f"{label}: keep {len(kept)}/{len(scores)} classes, {n_particles:,} particles",
             "green = kept (good), red = dropped (bad), * = the two selectors disagree"
             "   |   sorted by CryoSift score")


if __name__ == "__main__":
    main()
