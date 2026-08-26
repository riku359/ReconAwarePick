"""matplotlib figure builders for the reconstruction benchmark.

Pure functions over the small `derived/*` CSVs and `metrics.json` -- no CryoSPARC
access, so the figures redraw fast and reproducibly. Style constants (a single muted
hue for data-level distributions, the median accent) live in one block so the look is
consistent and project-specific choices are separated from the generic plotting.

Only matplotlib + numpy are used; CSVs are read with the stdlib `csv` module because
pandas is deliberately not a dependency of this package.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path

# --- style (single source of truth for the figure look) -----------------
HIST_FILL = "#4C78A8"     # one muted blue: V7 is a single, data-level distribution
HIST_EDGE = "#2A4E6C"     # darker rim so bars read individually
MEDIAN_INK = "#C44E52"    # warm accent reserved for the median marker/label
GRID_INK = "#B0B0B0"      # recessive gridlines
TEXT_INK = "#222222"
GT_INK = "#1B5E20"        # GT = oracle upper bound: green, anchored first
LEAK_INK = "#B71C1C"      # in-distribution/leak source: red, kept off the cross-picker ranking

# One panel per covariate: (csv column, human label, unit). Defocus / astigmatism
# / CTF-fit are the CTF covariates Patch CTF fits; relative ice thickness is the
# extra micrograph-quality covariate Patch-CTF reports (kept when present).
CTF_PANELS = (
    ("defocus_um",        "Defocus",                "µm"),
    ("astig_A",           "Astigmatism",            "Å"),
    ("ctf_fit_A",         "CTF fit resolution",     "Å"),
    ("ice_thickness_rel", "Relative ice thickness", ""),
)

# --- per-source scatter style (single source of truth for S1/S2/V1) -----
# source -> colour + marker in ONE place so the source-to-style mapping stays
# consistent across every scatter. GT is the oracle anchor (a reference line, no
# picker F1); crYOLO is drawn in the leak colour and, on ids where it is
# in-distribution, hatched. The three
# leak-free picker hues are the validated categorical slots blue / yellow / violet
# (dataviz palette) and the marker *shapes* differ too, so identity survives
# colour-blindness and greyscale. (display name, hex, matplotlib marker).
SOURCE_STYLE = {
    "gt":              {"display": "GT",              "color": GT_INK,     "marker": "*"},
    "topaz":           {"display": "Topaz",           "color": "#2A78D6",  "marker": "o"},
    "cryotransformer": {"display": "CryoTransformer", "color": "#C98500",  "marker": "s"},
    "cryosegnet":      {"display": "CryoSegNet",      "color": "#4A3AA7",  "marker": "^"},
    "cryolo":          {"display": "crYOLO",          "color": LEAK_INK,   "marker": "D"},
    # baseline-ablation's 4 arms (all CryoTransformer, differing only in
    # cleaner/CryoSift filtering) -- same base hue family as cryotransformer above,
    # distinguished by marker.
    "cryotransformer_clean_tri":               {"display": "+ cleaner",
                                                 "color": GT_INK, "marker": "^"},
    "cryotransformer_cryosift_iter":           {"display": "+ CryoSift",
                                                 "color": "#4A3AA7", "marker": "D"},
    "cryotransformer_clean_tri_cryosift_iter": {"display": "+ cleaner + CryoSift",
                                                 "color": "#2A78D6", "marker": "o"},
}

# Training-set leak per picker: the ids each pretrained model was trained on, so
# those points are flagged (hatched marker + "leak" note) and read OFF the
# cross-picker ranking.
LEAK_IDS = {
    "cryolo": {"10017", "10028", "10081"},
    "topaz":  {"10028"},
}


def is_leak(source: str, empiar_id: str) -> bool:
    return empiar_id in LEAK_IDS.get(source, set())


def read_csv_columns(csv_path: str | Path) -> dict:
    """Read a derived CSV into {column: list[value]} without pandas. Numeric
    strings become floats; anything unparseable (e.g. the micrograph name) stays
    a string."""
    with Path(csv_path).open(newline="") as fh:
        reader = _csv.DictReader(fh)
        columns: dict = {name: [] for name in (reader.fieldnames or [])}
        for row in reader:
            for name, value in row.items():
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    columns[name].append(value)
    return columns


def _percentile(ordered: list, q: float) -> float:
    """Linear-interpolated q-th percentile (q in [0,100]) of a pre-sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(ordered):
        return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])
    return ordered[lo]


def _median(values: list) -> float:
    return _percentile(sorted(values), 50)


def _robust_range(values: list) -> tuple:
    """Display range that hides only FAR outliers (failed-CTF astigmatism/CTF-fit
    blowups that stretch the axis and crush the real distribution into one bin),
    never a clean distribution's mild tails. Start from the Tukey far-out fence
    [Q1-3*IQR, Q3+3*IQR]; on each side snap back to the data extreme UNLESS that
    extreme sits more than one fence-width beyond it (= a true far outlier). Clean
    data therefore keeps its full range (nothing hidden)."""
    ordered = sorted(values)
    lo_d, hi_d = ordered[0], ordered[-1]
    q1, q3 = _percentile(ordered, 25), _percentile(ordered, 75)
    iqr = q3 - q1
    lo_f, hi_f = q1 - 3 * iqr, q3 + 3 * iqr
    span = hi_f - lo_f
    if span <= 0:
        return lo_d, hi_d
    hi = hi_f if hi_d > hi_f + span else hi_d
    lo = lo_f if lo_d < lo_f - span else lo_d
    return (lo_d, hi_d) if hi <= lo else (lo, hi)


def _draw_hist(ax, values: list, label: str, unit: str, bins: int) -> None:
    """One covariate panel: histogram (robust x-range) + median line labelled with
    its value, plus a terse count of any off-scale outliers so nothing is hidden
    silently."""
    lo, hi = _robust_range(values)
    n_off = sum(1 for v in values if v < lo or v > hi)
    ax.hist(values, bins=bins, range=(lo, hi),
            color=HIST_FILL, edgecolor=HIST_EDGE, linewidth=0.6)
    ax.set_xlim(lo, hi)

    med = _median(values)
    ax.axvline(med, color=MEDIAN_INK, linewidth=2.0, zorder=3)
    unit_suffix = f" {unit}" if unit else ""
    note = f"median {med:.4g}{unit_suffix}"
    if n_off:
        note += f"\n{n_off} off-scale"
    ax.annotate(note, xy=(0.96, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, color=MEDIAN_INK)

    ax.set_xlabel(f"{label} ({unit})" if unit else label, fontsize=10)
    ax.set_ylabel("micrographs", fontsize=10)
    ax.grid(axis="y", color=GRID_INK, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_ctf_covariates(csv_path, out_paths, title=None, bins=24) -> list:
    """Histogram the shared micrographs' CTF / quality covariates.

    The micrographs are shared across every condition, so this figure maps *data
    difficulty*, not any picker's behaviour (the surrounding docs note this; the
    figure itself is kept minimal -- a short `title` and a per-panel median only).
    One panel per covariate in `CTF_PANELS` that is present and numeric in the CSV.
    Writes every path in `out_paths` (e.g. a .png for inline use and a .pdf vector
    for the paper) and returns the list of paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = read_csv_columns(csv_path)
    panels = [(col, label, unit) for col, label, unit in CTF_PANELS
              if col in columns and columns[col]
              and all(isinstance(v, float) for v in columns[col])]
    if not panels:
        raise ValueError(f"no plottable CTF covariates in {csv_path}")

    ncols = 2 if len(panels) > 1 else 1
    nrows = -(-len(panels) // ncols)          # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (col, label, unit) in zip(axes, panels):
        _draw_hist(ax, columns[col], label, unit, bins)
    for ax in axes[len(panels):]:             # hide any unused grid cell
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1.0))

    written = []
    for path in out_paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def montage_plots(panels, out_paths, suptitle=None, stacked=False,
                  panel_size=(3.4, 2.7)) -> list:
    """Tile already-rendered plot images (one per panel) into a single sheet.

    This does NOT recompute anything: it lays CryoSPARC's own per-job renders (A4
    FSC / A5 viewing / A7 class-average PNGs) side by side so the sources read
    together. Each panel keeps its source image's own autoscale, so the sheet is a
    contact sheet, never a common-scale overlay. Kept deliberately
    sparse -- one short suptitle, one short per-panel title -- so the figure stays
    readable; the autoscale caveat lives in the docs, not on every image. Generic on
    purpose: the source order, per-panel titles and colours are the caller's choices.

    panels: list of {"png": path, "title": str, "color": hex}. panel_size is the
    per-panel (width, height) in inches. stacked=True stacks panels in one column
    (for wide montages); else one row. Writes every out_paths entry (png inline /
    pdf vector) and returns the paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        raise ValueError("montage_plots: no panels")
    pw, ph = panel_size   # per-panel (width, height) in inches
    if stacked:
        fig, axes = plt.subplots(n, 1, figsize=(pw, ph * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(pw * n, ph))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, panel in zip(axes, panels):
        ax.imshow(mpimg.imread(panel["png"]))
        ax.set_axis_off()
        ax.set_title(panel["title"], fontsize=11, color=panel.get("color", TEXT_INK))

    # A tall stacked figure needs an inch-based top reservation, else the suptitle
    # collides with the first panel's title (a percentage margin is too small when
    # the figure is ~30 in tall). A single row separates fine with plain tight_layout.
    if suptitle and stacked:
        fig_h = ph * n
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig_h))
        fig.suptitle(suptitle, fontsize=13, y=1 - 0.32 / fig_h)
    elif suptitle:
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout()
    else:
        fig.tight_layout()

    written = []
    for path in out_paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def montage_grid(rows, col_headers, out_paths, suptitle=None, cell_size=(2.4, 2.4)) -> list:
    """Tile a rows x columns grid of already-rendered PNGs into one sheet.

    Like `montage_plots` but 2D: column headers label the top, a row label sits at
    the left of each row (e.g. datasets down, sources across). Purely lays images
    out -- each keeps its own render's scale (a contact grid, not a common-scale
    overlay). Generic: the caller owns the labels, colours and order.

    rows: list of {"label": str, "color": hex, "cells": [png_or_None, ...]} with
    each `cells` aligned to `col_headers`. A None cell renders blank (e.g. a source
    with no map). cell_size is the per-cell (width, height) in inches. Writes every
    out_paths entry and returns the paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    nrows, ncols = len(rows), len(col_headers)
    if nrows == 0 or ncols == 0:
        raise ValueError("montage_grid: empty grid")
    cw, ch = cell_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(cw * ncols, ch * nrows), squeeze=False)

    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            png = row["cells"][c] if c < len(row["cells"]) else None
            if png:
                ax.imshow(mpimg.imread(str(png)))
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(col_headers[c], fontsize=12, color=TEXT_INK)
            if c == 0:
                ax.set_ylabel(row["label"], fontsize=12, rotation=90,
                              color=row.get("color", TEXT_INK), labelpad=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if suptitle else 1.0))
    written = _save_all(fig, out_paths, dpi=150)
    plt.close(fig)
    return written


def _save_all(fig, out_paths, dpi=200) -> list:
    written = []
    for path in out_paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path))
    return written


def plot_f1_vs_resolution(points, anchor, out_paths, title=None, caption=None) -> list:
    """S1 -- 2D F1 (x) vs 3D resolution (y) scatter, one marker per picker.

    The claim: *a picker's 2D F1 does
    not predict its downstream 3D resolution*. So the figure is deliberately a bare
    scatter with NO trend line -- the eye should see the non-monotonicity, not a
    fitted slope over 4 points.

    Encoding (all from the §0 read-conventions, wired through `SOURCE_STYLE`):
      * y-axis is INVERTED so better resolution (smaller Å) is UP -- then "no upward
        trend with F1" is literally "the points don't climb left-to-right".
      * `anchor` (GT) is a horizontal dashed line, the oracle upper bound, drawn
        first; GT has no picker F1 so it is never a scatter point.
      * a leak picker (`point["leak"]`) gets a hatched marker + "(leak)" so it is
        read off the cross-picker ranking.
      * a thin vertical bar spans each picker's seed 0/1/2 resolution range; the
        solid marker is the best seed (best-of-3), so seed sensitivity is visible.

    points: list of {source, f1, res_best, res_min, res_max, leak}. `source` keys
        into SOURCE_STYLE. anchor: {source, res, display} for the GT line. Marker
        identity is backed by direct text labels (names in ink, not the series
        colour) so it never rests on colour alone. Writes every out_paths entry
        (png inline / pdf vector) and returns the paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    # GT oracle line first: the fixed upper-bound datum every point is read against.
    # Label at the RIGHT end so it clears the "better ↑" note in the top-left corner.
    ax.axhline(anchor["res"], color=GT_INK, linestyle="--", linewidth=1.6, zorder=1)
    ax.annotate(f'{anchor["display"]} (oracle) {anchor["res"]:.2f} Å',
                xy=(0.985, anchor["res"]), xycoords=("axes fraction", "data"),
                va="bottom", ha="right", fontsize=9, color=GT_INK, fontweight="bold")

    for pt in points:
        style = SOURCE_STYLE[pt["source"]]
        color = style["color"]
        leak = pt.get("leak", False)

        # seed 0/1/2 spread: thin bar behind the best-seed marker.
        ax.plot([pt["f1"], pt["f1"]], [pt["res_min"], pt["res_max"]],
                color=color, linewidth=1.4, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.scatter([pt["f1"]], [pt["res_best"]], s=170, marker=style["marker"],
                   facecolor="none" if leak else color, edgecolor=color,
                   linewidths=2.2 if leak else 1.4,
                   hatch="////" if leak else None, zorder=4)

        name = style["display"] + (" (leak)" if leak else "")
        ax.annotate(f"{name}\nF1 {pt['f1']:.2f} · {pt['res_best']:.2f} Å",
                    xy=(pt["f1"], pt["res_best"]), xytext=(9, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9, color=TEXT_INK)

    xs = [pt["f1"] for pt in points]
    ax.set_xlim(min(xs) - 0.09, max(xs) + 0.12)   # headroom for the right-hand labels
    ax.invert_yaxis()                              # better resolution (smaller Å) UP
    ax.annotate("better ↑", xy=(0.015, 0.975), xycoords="axes fraction",
                va="top", ha="left", fontsize=9, color=GRID_INK, fontstyle="italic")

    ax.set_xlabel("2D particle-wise macro F1  (unified greedy matcher, recon operating point)",
                  fontsize=10)
    ax.set_ylabel("3D resolution — GSFSC 0.143 (Å)", fontsize=10)
    ax.grid(color=GRID_INK, linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend = the encoding key (the non-obvious marks), NOT a second copy of the
    # names already sitting on the points.
    handles = [
        mlines.Line2D([], [], color=GT_INK, linestyle="--", linewidth=1.6,
                      label="GT — oracle upper bound (no picker F1)"),
        mlines.Line2D([], [], color=TEXT_INK, marker="D", markerfacecolor="none",
                      markeredgewidth=2.0, linestyle="none", markersize=10,
                      label="hatched = in-distribution (training leak)"),
        mlines.Line2D([], [], color=TEXT_INK, linewidth=1.4, alpha=0.55,
                      label="bar = seed 0/1/2 spread (marker = best-of-3)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True,
              framealpha=0.9, edgecolor=GRID_INK)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5,
                 color=GRID_INK, wrap=True)

    fig.tight_layout()
    written = _save_all(fig, out_paths, dpi=200)
    plt.close(fig)
    return written
