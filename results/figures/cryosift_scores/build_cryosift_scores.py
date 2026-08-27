#!/usr/bin/env python3
"""Fig. 5: CryoSift class-score distributions of the four entries, with the discard
threshold drawn on top.

The point of the figure is that 4.5 is a fixed number on a scale that shifts from entry
to entry, so the same threshold cuts the tail on three entries and the bulk of the
distribution on EMPIAR-10345.

Reads `results/tables/cryosift_class_scores.csv`, the `baseline` stack of each entry:
50 classes per entry, one CryoSift score each. That file used to be read out of a
sibling checkout of the research repository; it is committed here instead.

Runs standalone: needs matplotlib and that one CSV, nothing else.

    python build_cryosift_scores.py [--out cryosift_scores.pdf]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402

CSV_PATH = figure_paths.TABLES_ROOT / "cryosift_class_scores.csv"

DATASETS = ["10081", "10093", "10345", "10532"]
DISCARD_THRESHOLD = 4.5

# The stack the figure reads: the picker's own candidates, before the contamination
# mask. The CSV also carries the masked stack of each entry, which this figure does not
# draw.
STACK = "baseline"

# The bins have to reach past the largest score of any entry (5.692 on EMPIAR-10532), or
# matplotlib drops the classes above the last edge and the panels no longer hold 50.
BINS = [1.0 + 0.25 * i for i in range(20)]

# One colour per entry, the same key the loop-rounds figure uses, so a reader who has
# seen one reads the other without re-learning it.
COLORS = {
    "10081": "#4C72B0",
    "10093": "#DD8452",
    "10345": "#55A868",
    "10532": "#C44E52",
}


def read_scores(path: Path):
    """{empiar_id: [score, ...]} for the un-masked stack of each entry."""
    if not path.is_file():
        raise SystemExit(f"{path} not found; it carries the 50 class scores per entry")
    scores = {eid: [] for eid in DATASETS}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["stack"] == STACK and row["empiar_id"] in scores:
                scores[row["empiar_id"]].append(float(row["cryosift_score"]))
    empty = [eid for eid, values in scores.items() if not values]
    if empty:
        raise SystemExit(f"{path} has no `{STACK}` rows for {', '.join(empty)}")
    return scores


def draw(scores, out_path: Path):
    fig, axes = plt.subplots(4, 1, figsize=(3.3, 3.4), sharex=True, sharey=True)
    panel_counts = []
    for ax, eid in zip(axes, DATASETS):
        values = sorted(scores[eid])
        counts, _, _ = ax.hist(
            values,
            bins=BINS,
            color=COLORS[eid],
            edgecolor="white",
            linewidth=0.4,
        )
        panel_counts.append(counts)
        ax.axvline(DISCARD_THRESHOLD, color="black", linewidth=1.1, zorder=5)
        discarded = sum(1 for v in values if v >= DISCARD_THRESHOLD)
        ax.text(
            0.02,
            0.98,
            f"EMPIAR-{eid}   {discarded}/{len(values)} discarded",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    # One count scale for all four panels, so the 50 classes of each entry cover the
    # same area and a narrow distribution reads as narrow rather than as fewer classes.
    # The factor is headroom for the panel label above the tallest bar.
    axes[0].set_ylim(0, max(c.max() for c in panel_counts) * 1.15)

    axes[-1].set_xlabel("CryoSift class score", fontsize=8.5)
    axes[-1].tick_params(labelsize=8)
    # The scale runs particle to non-particle, so name its two ends under the ticks
    # instead of spelling the convention out in the axis label.
    for score, meaning in ((1, "particle"), (5, "non-particle")):
        axes[-1].annotate(
            meaning,
            xy=(score, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )
    # Above the top panel, so it never collides with a bar or a panel label.
    axes[0].annotate(
        f"discard threshold {DISCARD_THRESHOLD}",
        xy=(DISCARD_THRESHOLD, 1.04),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout(h_pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=CSV_PATH,
                        help="class-score table (default results/tables/"
                             "cryosift_class_scores.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PDF (default $RAPICK_FIGURES_OUT/cryosift_scores.pdf)")
    args = parser.parse_args()
    out = args.out or figure_paths.figures_out() / "cryosift_scores.pdf"
    draw(read_scores(args.csv), Path(out))


if __name__ == "__main__":
    main()
