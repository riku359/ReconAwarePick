#!/usr/bin/env python3
"""Tile already-rendered local-resolution panels into one figure and its per-entry cuts.

render_locres_3d.py renders and tiles in one pass, which is right while every panel comes
out of the same run. The four-entry figure no longer does. EMPIAR-10081 and 10532 are the
panels the paper carries, drawn from placements that were frozen by hand and whose JSON is
gone, so they can only be reused; 10093 and 10345 are rendered fresh against the current
spec. This script takes both kinds as PNGs and lays them out.

Rows are scaled to a common height, where render_locres_3d.py pads them to a common pixel
box. Panel size in pixels only means something inside one run: the camera fits the
reference map of its row, so a row drawn at another time or another --panel-px lands on
its own scale and padding would put one row at half the size of the next. The scale is
chosen per row, so all five panels of a row keep the single scale their comparison needs,
and the same box is used for the combined figure and for the per-entry ones, which makes
each per-entry file one row of the combined figure rather than a separate layout.

  python tile_locres_panels.py --manifest panels.json \\
      --out locres_maps.pdf --per-entry locres_maps_{entry}.pdf

The manifest names each row's panels, in column order, and the palette stops the row was
coloured with:

  {"rows": [{"entry": "10081", "stops": [4.174, 9.526, 15.32],
             "panels": [{"label": "crYOLO", "png": "..."}, ...]}]}
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from PIL import Image


def crop_to_content(image):
    """Drop the transparent margin ChimeraX leaves around the surface."""
    opaque = image[..., 3] > 0.01
    rows_used, cols_used = np.where(opaque.any(axis=1))[0], np.where(opaque.any(axis=0))[0]
    if not len(rows_used) or not len(cols_used):
        return image
    return image[rows_used[0] : rows_used[-1] + 1, cols_used[0] : cols_used[-1] + 1]


def on_white(image):
    """Composite onto white before resampling.

    The figure background is white, so this changes nothing that is drawn, and it keeps
    the resampler from pulling whatever sits in the fully transparent pixels into the
    edge of the surface.
    """
    rgb, alpha = image[..., :3], image[..., 3:4]
    return rgb * alpha + (1.0 - alpha)


def scaled(image, factor):
    if abs(factor - 1.0) < 1e-3:
        return image
    height, width = image.shape[:2]
    target = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    resampled = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).resize(
        target, Image.LANCZOS)
    return np.asarray(resampled).astype(np.float32) / 255.0


def pad_to(image, height, width):
    canvas = np.ones((height, width, image.shape[2]), dtype=image.dtype)
    top, left = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
    canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
    return canvas


def load_rows(manifest_path):
    """Every panel cropped to its content, with each row scaled to the common height."""
    rows = json.loads(Path(manifest_path).read_text())["rows"]
    for row in rows:
        for panel in row["panels"]:
            panel["image"] = on_white(crop_to_content(plt.imread(panel["png"])))
        row["height"] = max(p["image"].shape[0] for p in row["panels"])
        row["width"] = max(p["image"].shape[1] for p in row["panels"])

    common_height = max(row["height"] for row in rows)
    for row in rows:
        factor = common_height / row["height"]
        for panel in row["panels"]:
            panel["image"] = scaled(panel["image"], factor)
        row["width"] = int(round(row["width"] * factor))
    return rows, common_height, max(row["width"] for row in rows)


def tile(rows, out_path, box_h, box_w):
    """Grid of the panels, each row with its own colour bar on the right.

    Rows carry different molecules on different resolution scales, so one range over all
    of them saturates the rows whose values are small. Per row is the comparison the
    figure exists for; what is dropped, comparing entries against each other, is not
    meaningful anyway.
    """
    n_rows, n_cols = len(rows), max(len(row["panels"]) for row in rows)

    panel_h_in = 1.55 * box_h / box_w
    fig = plt.figure(figsize=(1.55 * n_cols + 0.62, panel_h_in * n_rows + 0.35))
    grid = fig.add_gridspec(n_rows, n_cols + 1,
                            width_ratios=[1] * n_cols + [0.10],
                            wspace=0.02, hspace=0.04,
                            left=0.07, right=0.90, top=0.93, bottom=0.02)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_axis_off()
            if c >= len(row["panels"]):
                continue
            panel = row["panels"][c]
            ax.imshow(pad_to(panel["image"], box_h, box_w))
            if r == 0:
                ax.set_title(panel["label"], fontsize=8)
            if c == 0:
                ax.text(-0.04, 0.5, f"EMPIAR-{row['entry']}", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)

        bar_axis = fig.add_subplot(grid[r, n_cols])
        low, mid, high = row["stops"]
        # Matches ChimeraX's three-stop palette: blue at the best resolution, white at
        # the midpoint, red at the worst, with the midpoint placed where it really is.
        row_map = colors.LinearSegmentedColormap.from_list(
            f"locres{r}", [(0.0, "blue"), ((mid - low) / (high - low), "white"),
                           (1.0, "red")])
        mappable = cm.ScalarMappable(norm=colors.Normalize(low, high), cmap=row_map)
        bar = fig.colorbar(mappable, cax=bar_axis)
        bar.ax.tick_params(labelsize=6.5)
        if r == 0:
            bar.set_label("local resolution (Å)", fontsize=7)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path,
                        help="the combined figure, one row per entry")
    parser.add_argument("--per-entry", type=str, metavar="PATTERN",
                        help="one file per entry, e.g. locres_maps_{entry}.pdf")
    args = parser.parse_args()
    if not args.out and not args.per_entry:
        parser.error("nothing to write: pass --out, --per-entry, or both")

    rows, box_h, box_w = load_rows(args.manifest)
    for row in rows:
        low, mid, high = row["stops"]
        print(f"  EMPIAR-{row['entry']}  palette {low:.4g},blue:{mid:.4g},white:{high:.4g},red")
    if args.out:
        tile(rows, args.out, box_h, box_w)
    if args.per_entry:
        for row in rows:
            tile([row], Path(args.per_entry.format(entry=row["entry"])), box_h, box_w)


if __name__ == "__main__":
    main()
