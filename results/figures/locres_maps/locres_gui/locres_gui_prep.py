#!/usr/bin/env python3
"""Prepare one entry's panels for an interactive ChimeraX session.

Does exactly what `render_locres_3d.prepare()` does (mask by the refinement mask, pick
the contour level by enclosed volume, derive the principal frame) and writes the result
as JSON for `locres_gui_session.py` to open in the GUI. Nothing here differs from the
figure's own pipeline, so what the GUI shows is what the figure shows.

`--work` is where the masked volumes are written. The committed pose files name their
volumes under that directory, so re-running this with the same `--work` refills what
they point at.

Add `"fit": "envelope"` to the params it writes for the smoothed, both-hands fit; the
raw-density fit inside the renderer cannot cross a mirror.

    python locres_gui_prep.py --spec spec.json --entry 10093 --reference crYOLO \\
        --work /tmp/locres/gui/masked_cryolo --out params_10093.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import locres_render_lib as prep                                    # noqa: E402

RANGE_PERCENTILES = (2.0, 98.0)


def locres_values(panel):
    locres, _, _ = prep.read(panel["locres"])
    mask, _, _ = prep.read(panel["mask"])
    inside = mask > 0.5
    return locres[inside & (locres > 0)]


def palette_stops(panels):
    pooled = np.concatenate([locres_values(p) for p in panels])
    low, high = np.percentile(pooled, RANGE_PERCENTILES)
    return float(low), float(np.median(pooled)), float(high)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("/tmp/locres/masked"),
                    help="where the masked volumes are written")
    ap.add_argument("--reference", default="CryoTransformer",
                    help="the panel every other map is fitted onto. Align to the best "
                         "map of the row: the committed poses used crYOLO")
    ap.add_argument("--turn", default="")
    ap.add_argument("--panel-turn", action="append", default=[], metavar="LABEL=COMMAND")
    args = ap.parse_args()

    import mrcfile

    rows = json.loads(args.spec.read_text())["rows"]
    row = next((r for r in rows if r["entry"] == args.entry), None)
    if row is None:
        sys.exit("no row for " + args.entry)
    args.work.mkdir(parents=True, exist_ok=True)

    panels = []
    for panel in row["panels"]:
        density, mask, voxel, origin = prep.masked_density(panel["map"], panel["mask"])
        level = prep.contour_level(density, mask)
        rotation, centroid = prep.principal_frame(density, level, voxel, origin)
        masked_path = args.work.resolve() / f"{args.entry}_{panel['label']}__masked.mrc"
        with mrcfile.new(str(masked_path), overwrite=True) as handle:
            handle.set_data(density.astype(np.float32))
            handle.voxel_size = voxel
            handle.header.origin.x, handle.header.origin.y, handle.header.origin.z = origin
        # rotation rows are screen x/y/z = mid, long, short principal axis
        panels.append({
            "label": panel["label"],
            "masked": str(masked_path),
            "locres": panel["locres"],
            "level": level,
            "rotation": [list(map(float, r)) for r in rotation],
            "centroid": list(map(float, centroid)),
            "axis_long": list(map(float, rotation[1])),
            "axis_mid": list(map(float, rotation[0])),
            "axis_short": list(map(float, rotation[2])),
        })
        print(f"  {panel['label']:16s} level={level:.5g}  masked -> {masked_path.name}",
              flush=True)

    low, mid, high = palette_stops(row["panels"])
    params = {
        "entry": args.entry,
        "reference": args.reference,
        "palette": f"{low:.4g},blue:{mid:.4g},white:{high:.4g},red",
        "turn": args.turn,
        "panel_turns": dict(item.split("=", 1) for item in args.panel_turn),
        "panels": panels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(params, indent=1))
    print(f"EMPIAR-{args.entry}: palette {params['palette']}"
          + (f", {args.turn}" if args.turn else "")
          + (f", panel turns {sorted(params['panel_turns'])}"
             if params["panel_turns"] else ""))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
