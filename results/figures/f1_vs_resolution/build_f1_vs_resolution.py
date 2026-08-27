#!/usr/bin/env python3
"""Fig. 4: 2D macro F1 against GSFSC 0.143 resolution, one point per (picker, entry).

Both axes are the measured values the paper prints, read out of `results/tables/` so
that the figure and the tables cannot drift apart:

    F1          results/tables/detection_2d.json (Table S2) -- macro F1 on the 300
                CryoPPP-annotated micrographs, at the operating point that produced the
                STAR fed to reconstruction
    resolution  results/tables/main_results.json (Table 2) -- best of three seeds on
                the full micrograph set

The values used to be transcribed into this file as literals, for exactly that
anti-drift reason. `results/tables/` now carries the same numbers together with their
provenance, so the literals are gone and the property is kept.

Runs standalone: needs matplotlib and the two table files, nothing else.

    python build_f1_vs_resolution.py [--out f1_vs_resolution.pdf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402
import tables                                          # noqa: E402

# colour = entry
DATASETS = {
    "10081": "#1f77b4",
    "10093": "#e4572e",
    "10345": "#17a398",
    "10532": "#6a4c93",
}

IDEAL_COLOR = "#d62728"

# marker = picker. The keys are the release condition names of docs/PAPER_TO_CODE.md:
# the CryoTransformer row of the tables is the `baseline` condition.
PICKERS = {
    "cryolo": ("crYOLO", "o"),
    "topaz": ("Topaz", "s"),
    "cryosegnet": ("CryoSegNet", "^"),
    "baseline": ("CryoTransformer", "D"),
}

# The tables state the same quantity under slightly different field names depending on
# whether a cell carries its provenance alongside the value.
F1_FIELDS = ("macro_F1", "macro_f1", "f1", "F1")
RES_FIELDS = ("published", "raw")


def read_points():
    """{(entry, picker): (macro F1, best-of-three resolution)} from the two tables."""
    points = {}
    for entry in DATASETS:
        for picker in PICKERS:
            f1 = tables.number("detection_2d", picker, entry, field=F1_FIELDS)
            res = tables.number("main_results", picker, entry, field=RES_FIELDS)
            points[(entry, picker)] = (f1, res)
    return points


def draw(points, out_path: Path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#4d4d4d",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.set_axisbelow(True)

    # Ideal trend: the diagonal of the data range, from the worst resolution at the
    # lowest F1 to the best resolution at the highest F1. A guide to the eye, not a fit:
    # if the 2D score ordered the reconstructions, the points would follow it.
    f1s = [p[0] for p in points.values()]
    res = [p[1] for p in points.values()]
    ax.plot([min(f1s), max(f1s)], [max(res), min(res)], color=IDEAL_COLOR,
            linestyle=(0, (5, 3)), linewidth=1.2, zorder=1)

    for (dataset, picker), (f1, resolution) in points.items():
        ax.plot(f1, resolution, marker=PICKERS[picker][1], color=DATASETS[dataset],
                markersize=6.0, markeredgecolor="white", markeredgewidth=0.6,
                linestyle="none", zorder=3)

    ax.set_xlabel("2D macro F1")
    ax.set_ylabel("GSFSC 0.143 resolution (Å)")
    ax.set_xlim(0.22, 0.86)
    ax.set_ylim(3.2, 7.6)

    # Both legends sit inside the axes, in the empty upper-right region: no point has
    # F1 >= 0.55 together with a resolution worse than 5.5 A, and the ideal trend line
    # stays below it there. A white patch masks the grid under the text.
    dataset_handles = [Line2D([], [], marker="o", color=color, linestyle="none",
                              markersize=6.0, label=name)
                       for name, color in DATASETS.items()]
    dataset_handles.append(Line2D([], [], color=IDEAL_COLOR, linestyle=(0, (5, 3)),
                                  linewidth=1.2, label="ideal trend"))
    picker_handles = [Line2D([], [], marker=marker, color="#6b6b6b",
                             linestyle="none", markersize=6.0, label=name)
                      for name, marker in PICKERS.values()]
    legend_kw = dict(fontsize=8, handletextpad=0.4, borderpad=0.3,
                     borderaxespad=0.3, labelspacing=0.25, frameon=True,
                     framealpha=0.9, edgecolor="none")
    dataset_legend = ax.legend(handles=dataset_handles, ncol=1, loc="upper right",
                               handlelength=1.6, **legend_kw)
    ax.add_artist(dataset_legend)

    fig.tight_layout(pad=0.4)

    # The picker legend draws after the dataset legend, so its white patch would cover
    # the dataset markers if the boxes overlapped, and side by side the pair reaches the
    # 10093 points. Stack it under the dataset legend instead, anchored to that legend's
    # measured bottom-right corner.
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted().transform
    right_frac = to_axes(dataset_legend.get_window_extent().max)[0]
    bottom_frac = to_axes(dataset_legend.get_window_extent().min)[1]
    ax.legend(handles=picker_handles, ncol=1, loc="upper right",
              bbox_to_anchor=(right_frac, bottom_frac - 0.01), handlelength=1.0,
              **{**legend_kw, "borderaxespad": 0.0})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None,
                        help="output PDF (default $RAPICK_FIGURES_OUT/f1_vs_resolution.pdf)")
    args = parser.parse_args()
    out = args.out or figure_paths.figures_out() / "f1_vs_resolution.pdf"
    draw(read_points(), Path(out))


if __name__ == "__main__":
    main()
