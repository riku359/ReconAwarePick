#!/usr/bin/env python3
"""Fig. S2 and Fig. S3: the two contamination-mask post-processings.

Fig. S2 (mask_postproc_real.pdf)
    The released post-processing against this repository's triangular blending, on a
    micrograph carrying a stripe artifact. The micrograph is EMPIAR-10532
    FoilHole_24136458_..._021514, identified by correlation (0.87) against the saved
    default overlays of the contamination stage. Two panels, a red-only overlay
    (alpha = 0.65 x probability) on the denoised micrograph, no contour and no pick
    markers. Three committed-elsewhere assets, named after the micrograph's stem:

        mic_<stem>.jpg            the denoised micrograph, 1600 px
        mask_off_<stem>_gray.png  the released post-processing's mask, from the stored
                                  full-resolution npz
        mask_tri_<stem>_gray.png  the triangular-blend mask, computed at the 376 px
                                  model scale

    They are image assets rather than code, so they are not committed here: point
    `--assets` at the directory holding them. `src/rapick/cleaner/` regenerates both
    masks from the micrograph.

Fig. S3 (mask_blend_weights.pdf)
    Why uniform averaging steps at window borders and triangular blending does not.
    Entirely synthetic: four windows at 50% overlap, each predicting the same smooth
    field plus a window-specific offset; the assembled value is the weighted average of
    the covering windows. This panel needs no assets and always runs.

An alternative version of Fig. S2 that is not built, kept because the code for it is
here: the saved filter comparison on one micrograph, re-titled for the paper. Each panel
is the denoised micrograph, the mask in red, its 0.5 contour in yellow, the discarded
picks as red circles and the kept picks as green circles; the left panel is the released
`predictMask` post-processing and the right panel the triangular blending. The baked
header strip with the removed/kept counts is cropped off, because those counts are not
backed by a table in the paper. `split_filtercmp` below does that split.

    python build_mask_postproc_figs.py --assets <dir>     # both figures
    python build_mask_postproc_figs.py                    # Fig. S3 only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402

STEM = "24136458"    # the micrograph Fig. S2 shows (the stripe example)
DISP = 1600          # display resolution of the composite panels

TEXT = "#0b0b0b"
MUTED = "#52514e"
ACCENT = "#e34948"
RED = np.array([0.82, 0.10, 0.10], dtype=np.float32)
CONTOUR = "#ffd400"


def split_filtercmp(path):
    """Split a saved filter-comparison jpg into its two panels.

    Drops the baked header strip (a black band with the removed/kept counts) and the
    separator column between the panels.
    """
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"))
    rowmean = img.mean(axis=(1, 2))
    header_end = int(np.argmax(rowmean > 110))          # first non-header row
    body = img[header_end:]
    colmean = body.mean(axis=(0, 2))
    mid = len(colmean) // 2
    band = np.arange(mid - 40, mid + 40)
    sep = band[colmean[band] < 60]
    left = body[:, :sep.min()] if len(sep) else body[:, :mid]
    right = body[:, sep.max() + 1:] if len(sep) else body[:, mid:]
    return left, right


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": "#4d4d4d",
        "pdf.fonttype": 42,
    })


def note(ax, text, xy, xytext, color=ACCENT, fontsize=8.5):
    ax.annotate(text, xy=xy, xycoords="axes fraction",
                xytext=xytext, textcoords="axes fraction",
                color=color, fontsize=fontsize, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                          lw=0.8, alpha=0.95))


def load_assets(assets: Path, stem: str):
    from PIL import Image

    missing = [name for name in (f"mic_{stem}.jpg", f"mask_off_{stem}_gray.png",
                                 f"mask_tri_{stem}_gray.png")
               if not (assets / name).is_file()]
    if missing:
        raise SystemExit(f"missing under {assets}: {', '.join(missing)}")

    # The stored masks are in the raw mask frame, which is upside-down relative to the
    # denoised jpg. The same flipud is applied by the overlay renderer in
    # src/rapick/cleaner/overlay_panel.py.
    mic = np.asarray(Image.open(assets / f"mic_{stem}.jpg").convert("L"),
                     dtype=np.float32) / 255.0
    masks = {}
    for key in ("off", "tri"):
        m = Image.open(assets / f"mask_{key}_{stem}_gray.png").convert("L")
        m = m.resize((DISP, DISP), Image.BILINEAR)
        masks[key] = np.flipud(np.asarray(m, dtype=np.float32) / 255.0)
    return mic, masks


def overlay(mic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (0.65 * mask)[..., None]
    base = np.repeat(mic[..., None], 3, axis=2)
    return base * (1.0 - alpha) + RED[None, None, :] * alpha


def figure_real(assets: Path, stem: str, out_path: Path) -> None:
    mic, masks = load_assets(assets, stem)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.95),
                             gridspec_kw=dict(wspace=0.06))
    for ax, mask, title in ((axes[0], masks["off"], "released post-processing"),
                            (axes[1], masks["tri"], "triangular blending (ours)")):
        ax.imshow(overlay(mic, mask))
        ax.set_title(title, fontsize=9, color=TEXT, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#b9b8b4")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_weights(out_path: Path) -> None:
    patch, starts = 2.0, [0.0, 1.0, 2.0, 3.0]
    x = np.linspace(0, 5, 2001)
    field = 0.45 + 0.14 * np.sin(2.2 * x)
    # Same-parity windows disagree, so the uniform assembly jumps visibly at the borders
    # x=2 ((b2-b0)/2) and x=3 ((b3-b1)/2).
    biases = [0.22, 0.0, -0.18, 0.16]
    colors = ["#1baf7a", "#2a78d6"]

    def weights(kind):
        w = []
        for s in starts:
            inside = (x >= s) & (x <= s + patch)
            if kind == "uniform":
                wi = inside.astype(float)
            else:
                wi = np.where(inside,
                              1.0 - np.abs(x - (s + patch / 2)) / (patch / 2),
                              0.0)
            w.append(wi)
        return w

    def assembled(w):
        num = np.zeros_like(x)
        den = np.zeros_like(x)
        for wi, b in zip(w, biases):
            num += wi * np.clip(field + b, 0, 1)
            den += wi
        return num / np.maximum(den, 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.8), sharex=True,
                             gridspec_kw=dict(hspace=0.14, wspace=0.12))
    titles = ["uniform averaging\n(released)", "triangular blending\n(ours)"]
    for col, kind in enumerate(("uniform", "triangular")):
        w = weights(kind)
        ax_w, ax_m = axes[0, col], axes[1, col]
        for i, wi in enumerate(w):
            # Alternate slight height offsets so the overlapping uniform boxcars stay
            # distinguishable.
            scale = 1.0 if i % 2 == 0 else 0.94
            ax_w.fill_between(x, wi * scale, color=colors[i % 2],
                              alpha=0.14, linewidth=0)
            ax_w.plot(x, wi * scale, color=colors[i % 2], lw=1.4)
        ax_m.plot(x, assembled(w), color=TEXT, lw=1.6)
        ax_w.set_title(titles[col], fontsize=9, color=TEXT, pad=3)
        for ax in (ax_w, ax_m):
            for s in starts[1:] + [starts[-1] + patch]:
                if 1.1 < s < 3.9:
                    ax.axvline(s, color="#b9b8b4", lw=0.8,
                               linestyle=(0, (4, 3)), zorder=0)
            # Keep the image edges out of view so only the interior borders x=2 and x=3
            # are compared.
            ax.set_xlim(1.1, 3.9)
            ax.set_xticks([])
            ax.tick_params(labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        ax_w.set_ylim(-0.06, 1.15)
        ax_m.set_ylim(0.05, 0.95)
        ax_w.set_yticks([0, 1])
        ax_m.set_yticks([])
        if col == 1:
            ax_w.set_yticklabels([])

    axes[0, 0].set_ylabel("window\nweight", fontsize=8.5, color=TEXT)
    axes[1, 0].set_ylabel("assembled\nmask", fontsize=8.5, color=TEXT)
    axes[1, 0].annotate("step at the\nwindow border",
                        xy=(2.02, 0.33), xycoords="data",
                        xytext=(2.72, 0.78), textcoords="data",
                        color=ACCENT, fontsize=8, ha="center",
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
    axes[1, 1].annotate("no step", xy=(2.02, 0.34), xycoords="data",
                        xytext=(2.62, 0.78), textcoords="data",
                        color=MUTED, fontsize=8, ha="center",
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    fig.supxlabel("position along one image axis", fontsize=8.5, color=MUTED, y=0.02)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, default=None,
                        help="directory holding mic_<stem>.jpg and the two mask PNGs; "
                             "without it only Fig. S3 is built")
    parser.add_argument("--stem", default=STEM,
                        help="micrograph stem the assets are named after")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the PDFs go (default $RAPICK_FIGURES_OUT)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else figure_paths.figures_out()
    style()
    if args.assets:
        figure_real(args.assets.expanduser(), args.stem,
                    out_dir / "mask_postproc_real.pdf")
    else:
        print("no --assets: skipping Fig. S2, which needs the micrograph and its masks")
    figure_weights(out_dir / "mask_blend_weights.pdf")


if __name__ == "__main__":
    main()
