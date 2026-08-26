#!/usr/bin/env python
"""Figure only the classes where a hand-curated select_2D and CryoSift disagree.

Classes the two agree on (both keep, or both drop) are left out. The figure answers
which classes the two selectors split on, and what those classes' scores and particle
counts are.

Usage:
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.plot_selection_diff \\
        --job <class_2D uid> --manual-select <select_2D uid> \\
        --out results/figures/select2d/<empiar id>
"""

import argparse
import csv
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


def manual_kept_classes(select_dir):
    """The class indices appearing in a select_2D's particles_selected.cs.

    These are the classes a person kept.
    """
    particles = np.load(select_dir / "particles_selected.cs", mmap_mode="r")
    return {int(c) for c in np.unique(np.asarray(particles["alignments2D/class"]))}


def read_scores(csv_path):
    with open(csv_path, newline="") as f:
        return {int(r["class_idx"]): (float(r["cryosift_score"]), int(r["n_particles"]))
                for r in csv.DictReader(f)}


def class_averages(job_dir, job_uid):
    import mrcfile
    mrc_path = sorted(job_dir.glob(f"{job_uid}_*_class_averages.mrc"))[-1]
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        return np.array(mrc.data)


def draw(png_path, images, scores, groups, cutoff, title):
    """Draw one row per group, where groups = [(heading, [class_idx, ...]), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Floor the width so the heading is not cut off on an ID whose difference is a
    # single class.
    n_cols = max(len(idxs) for _, idxs in groups)
    fig, axes = plt.subplots(len(groups), n_cols,
                             figsize=(max(1.7 * n_cols + 1.4, 7.0), 2.1 * len(groups)),
                             squeeze=False)
    for row, (label, idxs) in enumerate(groups):
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(idxs):
                ax.axis("off")
                continue
            idx = idxs[col]
            score, n_particles = scores[idx]
            ax.imshow(images[idx], cmap="gray")
            ax.set_title(f"#{idx}  {score:.2f}\n{n_particles:,}p", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("tab:blue" if row == 0 else "tab:orange")
                spine.set_linewidth(2)
        axes[row][0].set_ylabel(label, fontsize=9)

    # Text drawn into the figure stays ASCII: matplotlib's default font carries no CJK
    # glyphs and renders them as tofu boxes.
    fig.suptitle(f"{title}  |  CryoSift cutoff {cutoff}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png_path, dpi=130)
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
    parser.add_argument("--out", required=True, help="directory to write the PNG into")
    parser.add_argument("--out-root",
                        help="parent to look for scores.csv under (default $RAPICK_WORK/select2d)")
    parser.add_argument("--env", default=None,
                        help=f"CryoSPARC credentials .env (default {DEFAULT_ENV_FILE})")
    args = parser.parse_args()

    env = read_env(args.env)
    project = args.project or require(env, "CRYOSPARC_PROJECT", args.env)
    cs = connect(env, args.env)
    job_dir = resolve_job_dir(cs, project, args.job)
    select_dir = resolve_job_dir(cs, project, args.manual_select)

    scores = read_scores(resolve_work_dir(args.out_root)
                         / f"{project}_{args.job}" / "scores.csv")
    manual_keep = manual_kept_classes(select_dir)
    cryosift_keep = {i for i, (s, _) in scores.items() if s < args.cutoff}

    manual_only = sorted(manual_keep - cryosift_keep)
    cryosift_only = sorted(cryosift_keep - manual_keep)
    if not manual_only and not cryosift_only:
        raise SystemExit(f"{args.job}: the manual ({args.manual_select}) and CryoSift "
                         f"class sets are identical, so there is no difference to draw")

    groups = [(f"manual keep\ncryosift drop\n({args.manual_select})", manual_only),
              ("cryosift keep\nmanual drop", cryosift_only)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(out_dir / f"cryosift_vs_manual_{args.job}.png",
         class_averages(job_dir, args.job), scores, groups, args.cutoff,
         f"{project} {args.job}: manual vs CryoSift disagreement")


if __name__ == "__main__":
    main()
