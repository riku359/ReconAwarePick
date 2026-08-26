#!/usr/bin/env python
"""plot_mask_postproc_figures.py -- draw the figures that explain the two mask
post-processings (Fig. S2 and Fig. S3 of the paper).

Figure C  what `fixJumpInBorders` does. It applies **the upstream implementation as it is**
      to a synthetic mask and shows before/after, then shows the same breakdown on real
      data with a pair of real masks. Passing a real function instead of hand-drawing a
      schematic settles the question "does that behaviour actually happen" on the spot.
Figure D  how triangular-window blending removes the seam. Where neighbouring windows
      disagree, a uniform average folds at the window border while a triangular window
      hands over smoothly.

Neither needs a GPU or the model weights. The real masks are read at model scale from the
arrays that compare_official_vs_triangular.py writes.

`fixJumpInBorders` is imported from the upstream checkout at
$RAPICK_THIRD_PARTY/micrograph_cleaner_em, which the script puts on sys.path itself.

Usage (a venv that has matplotlib; the micrograph_cleaner venv does not):
    python plot_mask_postproc_figures.py --out "$RAPICK_WORK/figures/postproc"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env  # noqa: E402

# The upstream implementation, used as it is, from $RAPICK_THIRD_PARTY/micrograph_cleaner_em.
sys.path.insert(0, str(env.upstream_dir()))
from micrograph_cleaner_em.predictMask import fixJumpInBorders  # noqa: E402

# A real example where the released mask flooded a band (the same micrograph as the
# failure gallery).
REAL_EXAMPLE = "10532__FoilHole_24136295_Data_24136382_24136384_20200224_020538_Fractions_patch_aligned.npz"

L = dict(p1="(1) uniform average of overlapping windows", p2="(2) after fixJumpInBorders",
         p3="(3) real data: official mask", p4="(4) same micrograph: triangular mask",
         border="window border",
         a2="the row before the border is copied to the image edge,\nturning the clean lower half into contamination",
         a3="rotational streaks and blocks from the TTA", a4="the same field, without seams",
         cap_c="(1)(2) apply the upstream fixJumpInBorders to a synthetic mask. "
               "(3)(4) are one EMPIAR-10532 micrograph at model scale.",
         win="window {i}", zero="weight 0 at its own edge", x="pixel position", y="weight",
         cap_d="Overlapping windows are summed with triangular weights. Each falls to zero "
               "at its edge, so contributions hand over smoothly.")

MODEL_IMG_SIZE = 256
STRIDE = MODEL_IMG_SIZE // 2          # strideFactor=2

TEXT  = "#0b0b0b"
MUTED = "#52514e"
GRID  = "#d8d7d2"
ACCENT = "#e34948"                    # used only for the annotations that point at a breakdown
BLUE   = "#2a78d6"
AQUA   = "#1baf7a"


def synthetic_seam(size: int = 376) -> np.ndarray:
    """A synthetic mask with a step at a window border.

    The upper part has a high-probability band standing in for a carbon edge, which falls
    to the clean area below at the border (row=STRIDE). `fixJumpInBorders` fires when the
    fraction of columns that drop by at least 0.4 across the border is at least
    (stride-1)/W, so the band spans two thirds of the image width to trip it for certain.
    """
    mask = np.full((size, size), 0.05, dtype=np.float64)
    mask[STRIDE - 60:STRIDE, :int(size * 0.67)] = 0.92        # the carbon edge just above the border
    mask[40:90, int(size * 0.72):] = 0.85                     # unrelated real contamination (should survive)
    return mask


def show_mask(ax, mask: np.ndarray, title: str, mark_boundary: bool = False) -> None:
    ax.imshow(mask, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=TEXT, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if mark_boundary:
        ax.axhline(STRIDE, color="#7ee0ff", linewidth=1.2, linestyle=(0, (5, 3)))
        ax.text(mask.shape[1] - 6, STRIDE + 8, L["border"], color="#7ee0ff", fontsize=8,
                va="top", ha="right")


def figure_c(out: Path, arrays: Path, example: str) -> Path:
    before = synthetic_seam()
    after, jump_found = fixJumpInBorders(before.copy(), axis=0, stride=STRIDE)
    if not jump_found:
        raise SystemExit("the synthetic mask did not trip fixJumpInBorders "
                         "(the premise of the figure no longer holds)")

    npz = np.load(arrays / example, allow_pickle=True)
    official, triangular = npz["off"].astype(np.float32), npz["tri"].astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.15))
    show_mask(axes[0], before, L["p1"], mark_boundary=True)
    show_mask(axes[1], after, L["p2"])
    show_mask(axes[2], official, L["p3"])
    show_mask(axes[3], triangular, L["p4"])

    axes[1].annotate(L["a2"],
                        xy=(0.30, 0.55), xycoords="axes fraction",
                        xytext=(0.30, 0.95), textcoords="axes fraction",
                        color=ACCENT, fontsize=7.5, ha="center", va="top",
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=0.9))
    axes[2].annotate(L["a3"],
                        xy=(0.5, 0.04), xycoords="axes fraction",
                        color=ACCENT, fontsize=7.5, ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT, lw=0.9))
    axes[3].annotate(L["a4"],
                        xy=(0.5, 0.04), xycoords="axes fraction",
                        color=MUTED, fontsize=7.5, ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.9))

    fig.text(0.5, 0.02, L["cap_c"], ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    png = out / "C_fixjump_mechanism.png"
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return png


def figure_d(out: Path) -> Path:
    """A schematic of triangular-window blending, drawing only the per-window weights.

    The numerical comparison against the released version is carried by the qualitative
    comparison, so all this has to convey is the one point that the weight falls to zero
    at the window edge.
    """
    patch, stride = 256, 128
    starts = [0, stride, 2 * stride, 3 * stride]
    tri = 1.0 - np.abs((np.arange(patch) - (patch - 1) / 2) / ((patch - 1) / 2))

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for i, s0 in enumerate(starts):
        x = np.arange(s0, s0 + patch)
        ax.fill_between(x, tri, color=AQUA, alpha=0.16, linewidth=0)
        ax.plot(x, tri, color=AQUA, linewidth=2.0)
        ax.text(s0 + patch / 2, 1.06, L["win"].format(i=i + 1), ha="center",
                color=MUTED, fontsize=9)

    ax.annotate(L["zero"], xy=(starts[2], 0.0), xytext=(starts[2] + 26, 0.30),
                color=ACCENT, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))

    ax.set_xlim(-20, starts[-1] + patch + 20)
    ax.set_ylim(-0.05, 1.22)
    ax.set_xlabel(L["x"], color=MUTED, fontsize=10)
    ax.set_ylabel(L["y"], color=MUTED, fontsize=10)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    fig.text(0.5, -0.02, L["cap_d"], ha="center", color=MUTED, fontsize=9)
    fig.tight_layout()
    png = out / "D_triangular_window.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output directory (default: $RAPICK_WORK/figures/postproc)")
    ap.add_argument("--arrays", default=None,
                    help="compare_official_vs_triangular.py's arrays "
                         "(default: $RAPICK_WORK/mask_compare/arrays)")
    ap.add_argument("--example", default=REAL_EXAMPLE,
                    help="the npz in --arrays whose real masks panels (3) and (4) draw")
    args = ap.parse_args(argv)

    out = Path(args.out or os.path.join(env.work_root(), "figures", "postproc"))
    arrays = Path(args.arrays or os.path.join(env.work_root(), "mask_compare", "arrays"))
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False

    print(figure_c(out, arrays, args.example))
    print(figure_d(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
