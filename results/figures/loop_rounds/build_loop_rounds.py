#!/usr/bin/env python3
"""Per-round loop diagnostics: permanent-discard and final-survival fractions.

This is the plot that shows the loop is a one-round effect: round 1 moves the fractions
and rounds 2 and 3 stay there. EMPIAR-10093 and 10532 move the way the design predicts,
EMPIAR-10081 moves the other way, EMPIAR-10345 does not move.

The two series come from `results/tables/loop_rounds.json`, the file behind Table 6.
They sit in that file's `not_in_paper` block rather than in `values`, because the table
stopped printing these two rates when this plot was made:

    permanent_discard_pct_unrounded   (n_class2d - attractor_kept - round0_kept)
                                      / n_class2d: the share of the particles reaching
                                      2D classification that the first CryoSift
                                      iteration rejects permanently
    final_survival_pct_unrounded      kept_particles / n_class2d

Both are per entry, one value per round, in round order. Note the denominator: it is the
particles that reach 2D classification, not the round's raw picks, which is what
`after_purify_share` in the printed table divides by.

The values used to be transcribed into this file as literals so that the table and the
plot could not drift apart; reading the one file keeps that and drops the duplication.

Runs standalone: needs matplotlib and that one table file, nothing else. The manuscript
carries this plot in its candidate supplement rather than in the main paper; it is the
figure form of Table 6.

    python build_loop_rounds.py [--out loop_rounds.pdf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402
import tables                                          # noqa: E402

TABLE = "loop_rounds"
ENTRIES = ["10081", "10093", "10345", "10532"]

DISCARD_BLOCK = "permanent_discard_pct_unrounded"
SURVIVAL_BLOCK = "final_survival_pct_unrounded"

COLORS = {
    "10081": "#4C72B0",
    "10093": "#DD8452",
    "10345": "#55A868",
    "10532": "#C44E52",
}
MARKERS = {"10081": "o", "10093": "s", "10345": "^", "10532": "D"}


def series(block: str, entry: str):
    """One entry's per-round percentages, in round order."""
    node = tables.extra(TABLE, block, entry)
    if not isinstance(node, list):
        raise SystemExit(f"{tables.table_path(TABLE)}: not_in_paper[{block!r}][{entry!r}] "
                         f"is a {type(node).__name__}, expected one value per round")
    return [float(v) for v in node]


def read_rounds():
    """{entry: [(round, discard %, survive %), ...]}."""
    out = {}
    for entry in ENTRIES:
        discard = series(DISCARD_BLOCK, entry)
        survive = series(SURVIVAL_BLOCK, entry)
        if len(discard) != len(survive):
            raise SystemExit(
                f"{tables.table_path(TABLE)}: EMPIAR-{entry} has {len(discard)} discard "
                f"values against {len(survive)} survival values")
        out[entry] = [(n, discard[n], survive[n]) for n in range(len(discard))]
    return out


def draw_panel(ax, rounds_by_entry, column, ylabel, title):
    for entry in ENTRIES:
        rows = rounds_by_entry[entry]
        ax.plot(
            [r[0] for r in rows],
            [r[column] for r in rows],
            color=COLORS[entry],
            marker=MARKERS[entry],
            markersize=3.5,
            linewidth=1.2,
            label=f"EMPIAR-{entry}",
        )
    ax.set_xticks(sorted({r[0] for rows in rounds_by_entry.values() for r in rows}))
    ax.set_xlabel("feedback round", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def draw(rounds_by_entry, out_path: Path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.3, 1.9))
    # Round 0 picks with the base checkpoint, so a design that works would pull the
    # discard fraction down and push the survival fraction up as rounds accumulate.
    draw_panel(left, rounds_by_entry, 1, "permanently discarded (%)", "(a) discarded")
    draw_panel(right, rounds_by_entry, 2, "surviving to selection (%)", "(b) survived")

    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.tight_layout(w_pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None,
                        help="output PDF (default $RAPICK_FIGURES_OUT/loop_rounds.pdf)")
    args = parser.parse_args()
    out = args.out or figure_paths.figures_out() / "loop_rounds.pdf"
    draw(read_rounds(), Path(out))


if __name__ == "__main__":
    main()
