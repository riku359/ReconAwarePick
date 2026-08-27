#!/usr/bin/env python3
"""Fig. 3: density maps coloured by local resolution, on one scale per row.

This is the figure CryoTransformer's Supplementary S10 shows, with two changes.

S10 gives every panel its own colour bar, so red on a 0-13.4 A panel and red on a
0-34.9 A panel mean different things and its columns cannot be compared by eye. Here the
range is computed once per row, over every map in that row, and one bar is drawn for it.

S10 also leaves every map in whatever orientation its refinement ended in. Here each row
is brought into one frame: the reference map is oriented on the principal axes of its own
density, and the other maps of the row are fitted onto it, so a difference between two
panels is a difference between the maps and not between two viewpoints. A row whose
placement was decided by hand is replayed from a frozen pose file instead of refitted
(`--poses`), because the fit uses random restarts and would not repeat.

How the volumes are prepared, and why, is in `lib/locres_render_lib.py`. In short: the
sharpened map is masked by the refinement mask before anything else, and the contour
level is chosen by enclosed volume rather than by a percentile of the whole box.

NEEDS CHIMERAX. `--chimerax` names the executable, otherwise `$RAPICK_CHIMERAX`,
otherwise the macOS bundle. On macOS ChimeraX must run windowed: the bundle has no
OSMesa, so `--offscreen` and `--nogui` both fail with "OpenGL rendering is not
available", and windows flash up and close by themselves for every panel. On Linux point
`--chimerax` at `chimerax_headless.sh` next to this file, which supplies OSMesa.

    python render_locres_3d.py --spec spec.json --out locres_maps.pdf \\
        --reference crYOLO --poses poses/poses_10093.json

Read this directory's README before rendering: it carries the reference-map trap, the
silhouette-width trap and the palette stops of the two rows that can no longer be
re-rendered.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import mrcfile                                         # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib import cm, colors                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402
import locres_render_lib as prep                       # noqa: E402

# Local resolution is noisy at the mask edge, so the row's range is taken between these
# percentiles of the pooled values rather than at the extremes.
RANGE_PERCENTILES = (2.0, 98.0)


def prepare(panel, work_dir):
    """Write the masked volume for one panel and record what rendering it needs."""
    # Drawn: the sharpened map at whatever level ChimeraX picks on open, which is the
    # GUI procedure this figure follows. Masking and an explicit level are used only to
    # work out the orientation, where noise outside the molecule would dominate the
    # principal axes; they never change what is rendered.
    density, mask, voxel, origin = prep.masked_density(panel["map"], panel["mask"])
    level = prep.contour_level(density, mask)
    rotation, centroid = prep.principal_frame(density, level, voxel, origin)

    # Absolute: ChimeraX resolves open/save paths against its own working directory.
    masked_path = Path(work_dir).resolve() / f"{Path(panel['map']).stem}__masked.mrc"
    with mrcfile.new(str(masked_path), overwrite=True) as handle:
        handle.set_data(density.astype(np.float32))
        handle.voxel_size = voxel
        handle.header.origin.x, handle.header.origin.y, handle.header.origin.z = origin

    panel["masked"] = str(masked_path)
    panel["level"] = level
    panel["frame"] = (rotation, centroid)
    return panel


def locres_values(panel):
    """Local-resolution values inside the refinement mask, where they mean something."""
    locres, _, _ = prep.read(panel["locres"])
    mask, _, _ = prep.read(panel["mask"])
    inside = mask > 0.5
    if inside.shape != locres.shape:
        raise ValueError(f"shape mismatch for {panel['label']}")
    return locres[inside & (locres > 0)]


def palette_stops(panels, percentiles=RANGE_PERCENTILES):
    """(best, mid, worst) local resolution in Angstrom for one row's colour scale.

    Taken from the values inside the refinement masks, so background zeros and the long
    tail above the mask edge do not set the ends. The midpoint is the median, which puts
    white where most of the surface sits and leaves both colours visible.
    """
    pooled = np.concatenate([locres_values(p) for p in panels])
    low, high = np.percentile(pooled, percentiles)
    mid = float(np.median(pooled))
    return float(low), mid, float(high)


def palette_argument(stops):
    """ChimeraX `palette` spec: blue at the best resolution, red at the worst."""
    low, mid, high = stops
    return f"{low:.4g},blue:{mid:.4g},white:{high:.4g},red"


def pose_matrix_command(matrix, target):
    """ChimeraX `view matrix` string for a 3x4 placement frozen in the GUI.

    The two forms are punctuated differently: the camera takes the numbers after a
    space, a model after its spec and a comma.
    """
    numbers = ",".join("%.8g" % value for row in matrix for value in row)
    if target == "camera":
        return "view matrix camera %s" % numbers
    return "view matrix models %s,%s" % (target, numbers)


def run_chimerax(commands, chimerax):
    with tempfile.NamedTemporaryFile("w", suffix=".cxc", delete=False) as handle:
        handle.write("\n".join(commands) + "\n")
        script_path = handle.name
    result = subprocess.run(
        [chimerax, "--exit", "--silent", "--cmd", f"open {script_path}"],
        capture_output=True, text=True, timeout=1800,
    )
    try:
        Path(script_path).unlink()          # `missing_ok` needs a newer interpreter
    except FileNotFoundError:
        pass
    return result


def render_row(row, reference, value_range, render_dir, chimerax, panel_px,
               silhouette_width):
    """Every panel of one row, in the reference's frame and under one camera.

    The reference is placed on its own principal axes and each other map is fitted onto
    it. Colouring happens before the fit, because `color sample` reads the
    local-resolution volume at the surface's current position; once the vertices carry
    their colours the surface can be moved freely.
    """
    turn = row.get("turn", "")
    panel_turns = row.get("panel_turns", {})
    frozen = row.get("frozen")
    palette = value_range

    def frozen_panel(label):
        """What the session drew this panel from, and where it put it. A mirrored panel
        is a volume ChimeraX derived and wrote out, not the file in the spec."""
        return frozen["panels"][label]

    if frozen:
        # Placement decided by hand in the GUI and frozen with save_poses.py. Nothing is
        # refitted here: the row is drawn exactly as it was approved, which is the only
        # way the figure and the session can be held to the same arrangement.
        reference_frame = pose_matrix_command(
            frozen_panel(reference["label"])["matrix"], "#1")
    else:
        rotation, centroid = reference["frame"]
        reference_frame = prep.view_matrix_command(rotation, centroid, model="#1")

    for panel in row["panels"]:
        png = Path(render_dir) / f"{row['entry']}_{panel['label'].replace(' ', '_')}.png"
        is_reference = panel is reference

        if frozen:
            reference_map = frozen_panel(reference["label"])["map"]
            panel_map = frozen_panel(panel["label"])["map"]
            panel_locres = frozen_panel(panel["label"])["locres"]
        else:
            reference_map = reference["masked"]
            panel_map, panel_locres = panel["masked"], panel["locres"]

        commands = ["set bgColor white", f"open {reference_map}"]
        if is_reference:
            target, locres_model = "#1", "#2"
            commands += [f"open {panel_locres}"]
        else:
            target, locres_model = "#2", "#3"
            commands += [f"open {panel_map}", f"open {panel_locres}"]

        commands += [
            "volume #1 style surface",
            f"volume {target} style surface",
            f"volume {locres_model} style image hide",
            f"color sample {target} map {locres_model} palette {palette}",
            "view initial",
            reference_frame,
        ]
        if not is_reference and frozen:
            commands.append(pose_matrix_command(frozen_panel(panel["label"])["matrix"],
                                                "#2"))
        elif not is_reference:
            # Start the fit from this map's own principal frame, which is already close,
            # then let fitmap resolve the residual flip. Principal axes are ambiguous
            # whenever two extents are similar, which is exactly where the unfitted
            # frames disagreed.
            commands += [
                prep.view_matrix_command(*panel["frame"], model="#2"),
                "fitmap #2 inMap #1",
            ]
        # Applied after the fit, so it moves every model of the row together and the
        # panels keep sharing a viewpoint. Without it a squat particle is seen straight
        # down its own short axis, which hides the shape.
        # Row tweak first, then anything specific to this panel: a map that came out of
        # refinement the other way up needs a further half turn to match the rest of its
        # row. This is a rotation of the model, not a mirror of the image, so the
        # structure stays the hand it was reconstructed in.
        # A frozen row carries its turns already, baked into the poses it was saved at.
        extra = [] if frozen else [
            t for t in (turn, panel_turns.get(panel["label"], "")) if t]
        commands += extra + [
            "lighting soft",
            # The width is in screen pixels, so it does not mean the same thing on every
            # machine: the macOS build draws on a Retina framebuffer and the default 1
            # comes out two to three times heavier than a Linux server's, which reads as
            # a black outline rather than the thin one the figure was designed with.
            f"graphics silhouettes true width {silhouette_width}",
        ]
        if frozen:
            # Orientation from the session, framing still from the reference: the GUI
            # window is a different shape from a square panel, so its zoom would crop.
            commands.append(pose_matrix_command(frozen["camera"], "camera"))
        commands.append("view #1")           # one camera per row, set by the reference
        if not is_reference:
            commands.append("hide #1 models")
        commands += [
            f"save {png.resolve()} width {panel_px} height {panel_px} "
            f"supersample 3 transparentBackground true",
            "exit",
        ]

        # Drop any earlier render first. ChimeraX exits 0 on some failures (a wrong
        # glibc, no OSMesa), and with the previous run's file still in place the
        # existence check below passes and a stale panel is silently reused.
        if png.exists():
            png.unlink()
        result = run_chimerax(commands, chimerax)
        if not png.exists():
            raise RuntimeError(f"no image for {row['entry']} {panel['label']}\n"
                               f"{(result.stderr or result.stdout)[-1200:]}")
        panel["png"] = str(png)
        print(f"  rendered {png.name}", flush=True)


def crop_to_content(image):
    opaque = image[..., 3] > 0.01
    rows_used, cols_used = np.where(opaque.any(axis=1))[0], np.where(opaque.any(axis=0))[0]
    if not len(rows_used) or not len(cols_used):
        return image
    return image[rows_used[0]: rows_used[-1] + 1, cols_used[0]: cols_used[-1] + 1]


def pad_to(image, height, width):
    canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top, left = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
    canvas[top: top + image.shape[0], left: left + image.shape[1]] = image
    return canvas


def tile(rows, out_path):
    """Grid of the rendered panels, each row with its own colour bar on the right.

    Rows carry different molecules on different resolution scales, so one range over all
    of them saturates the rows whose values are small. Per row is the comparison the
    figure exists for; what is dropped, comparing entries against each other, is not
    meaningful anyway. EMPIAR-10345 is the reason this matters: its declared pixel size
    is half the physical one, so its row sits on a different Angstrom scale from the
    other three and on a global range it would saturate red and show nothing.
    """
    n_rows, n_cols = len(rows), max(len(row["panels"]) for row in rows)

    cropped = {}
    for row in rows:
        for panel in row["panels"]:
            cropped[panel["png"]] = crop_to_content(plt.imread(panel["png"]))
    box_h = max(image.shape[0] for image in cropped.values())
    box_w = max(image.shape[1] for image in cropped.values())

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
            ax.imshow(pad_to(cropped[panel["png"]], box_h, box_w))
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    for row in rows:
        print(f"  EMPIAR-{row['entry']}  palette " + palette_argument(row["stops"]))
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chimerax", default=None,
                        help="ChimeraX executable; else $RAPICK_CHIMERAX, else the "
                             "macOS bundle. Use chimerax_headless.sh on Linux.")
    parser.add_argument("--panel-px", type=int, default=520,
                        help="rendered panel size; the committed figure used 1040, "
                             "which halves the silhouette relative to the structure")
    parser.add_argument("--silhouette-width", type=float, default=1.0,
                        help="ChimeraX silhouette width in screen pixels. 1 reproduces "
                             "a Linux render; the committed figure used 0.5, which is "
                             "as thin as the setting goes")
    parser.add_argument("--reference", default="CryoTransformer",
                        help="panel label each row is oriented and fitted onto. Align "
                             "to the best map of the row, not to a fixed method: on "
                             "EMPIAR-10093 and 10345 the CryoTransformer map is the "
                             "worst of its row, and the committed figure used crYOLO")
    parser.add_argument("--stops", action="append", default=[], metavar="ENTRY=B,W,R",
                        help="override a row's palette stops, e.g. 10081=8,10,14")
    parser.add_argument("--turn", action="append", default=[], metavar="ENTRY=COMMAND",
                        help='extra ChimeraX rotation for one row, e.g. 10081="turn x 65"')
    parser.add_argument("--panel-turn", action="append", default=[],
                        metavar="ENTRY:LABEL=COMMAND",
                        help='extra rotation for one panel, e.g. 10081:Ours="turn x 180"')
    parser.add_argument("--poses", action="append", default=[], metavar="PATH",
                        help="a poses JSON written by locres_gui/save_poses.py in a "
                             "ChimeraX session; that row is drawn at the frozen "
                             "placement and camera instead of being refitted. "
                             "Repeatable, one per row.")
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--entries", nargs="+",
                        help="render only these entries, in the spec's order")
    args = parser.parse_args()

    chimerax = figure_paths.chimerax_command(args.chimerax)

    rows = json.loads(args.spec.read_text())["rows"]
    if args.entries:
        rows = [row for row in rows if row["entry"] in args.entries]
        if not rows:
            sys.exit(f"no rows left after filtering to {args.entries}")
    panels = [panel for row in rows for panel in row["panels"]]
    missing = [p["label"] for p in panels
               if not all(Path(p[k]).exists() for k in ("map", "locres", "mask"))]
    if missing:
        sys.exit(f"volumes not found for: {', '.join(missing)}")

    tweaks = dict(item.split("=", 1) for item in args.turn)
    panel_tweaks = {}
    for item in args.panel_turn:
        target, command = item.split("=", 1)
        entry, label = target.split(":", 1)
        panel_tweaks.setdefault(entry, {})[label] = command
    frozen_rows = {}
    for item in args.poses:
        state = json.loads(Path(item).read_text())
        frozen_rows[state["entry"]] = state

    stop_overrides = {}
    for item in args.stops:
        entry, values = item.split("=", 1)
        stop_overrides[entry] = tuple(float(v) for v in values.split(","))

    render_dir = args.render_dir or args.out.parent / "locres_panels"
    render_dir.mkdir(parents=True, exist_ok=True)
    work_dir = render_dir / "masked"
    work_dir.mkdir(exist_ok=True)

    for row in rows:
        for panel in row["panels"]:
            prepare(panel, work_dir)
        row["stops"] = stop_overrides.get(row["entry"]) or palette_stops(row["panels"])
        row["range"] = (row["stops"][0], row["stops"][-1])
        row["turn"] = tweaks.get(row["entry"], "")
        row["panel_turns"] = panel_tweaks.get(row["entry"], {})
        row["frozen"] = frozen_rows.get(row["entry"])
        if row["frozen"]:
            absent = [p["label"] for p in row["panels"]
                      if p["label"] not in row["frozen"]["panels"]]
            if absent:
                sys.exit("poses for EMPIAR-%s have no placement for: %s"
                         % (row["entry"], ", ".join(absent)))
        reference = next((p for p in row["panels"] if p["label"] == args.reference),
                         row["panels"][0])
        print(f"EMPIAR-{row['entry']}: reference {reference['label']}, palette "
              + palette_argument(row["stops"])
              + (", frozen poses" if row["frozen"] else "")
              + (f", {row['turn']}" if row["turn"] and not row["frozen"] else "")
              + (f", panel turns {sorted(row['panel_turns'])}"
                 if row["panel_turns"] and not row["frozen"] else ""),
              flush=True)
        render_row(row, reference, palette_argument(row["stops"]), render_dir,
                   chimerax, args.panel_px, args.silhouette_width)

    tile(rows, args.out)


if __name__ == "__main__":
    main()
