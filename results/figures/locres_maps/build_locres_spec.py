#!/usr/bin/env python3
"""Write the render spec for Fig. 3 from the reconstruction pipeline's own manifests.

Reads each condition's `metrics.json` rather than scanning the CryoSPARC workspace, so
the volumes rendered are the recorded best-of-three winner and its local-resolution job:
the same pair the tables in the paper are computed from. `metrics.json` carries the job
directory alongside the uid, so no project path has to be supplied and no CryoSPARC
connection is needed. The volumes themselves do have to be on a filesystem this machine
can read.

    python build_locres_spec.py --out spec.json [--setting full]

The panels are the five columns of the paper's Table 2, in that order, and the labels
are the names the figure prints. `Ours` is the `fb` condition: the round-1 checkpoint
through the contamination mask and the 2D class selection.

If a manifest lacks the keys this expects it says so and skips that panel, rather than
guessing at a volume path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402

# Column order of the main results table, with Ours last. The second element of each
# pair is the release condition name of docs/PAPER_TO_CODE.md, which is the directory
# the reconstruction pipeline writes each arm's manifest under.
ARMS = [
    ("crYOLO", "cryolo"),
    ("Topaz", "topaz"),
    ("CryoSegNet", "cryosegnet"),
    ("CryoTransformer", "baseline"),
    ("Ours", "fb"),
]
ENTRIES = ["10081", "10093", "10345", "10532"]


def manifest_roots(explicit):
    """Where to look for `empiar_<id>/<setting>/<condition>/metrics.json`.

    `$RAPICK_WORK` unless `--experiments` names other roots; the first root holding an
    arm wins, so a condition kept in a second tree can be pulled into one figure.
    """
    if explicit:
        return [Path(root).expanduser() for root in explicit]
    return [figure_paths.work_root()]


def find_metrics(roots, entry, condition, setting):
    for root in roots:
        path = root / f"empiar_{entry}" / setting / condition / "metrics.json"
        if path.exists():
            return path
    return None


def job_dir(metrics, key):
    entry = metrics.get("maps", {}).get(key)
    if not isinstance(entry, dict) or not entry.get("dir"):
        raise KeyError(f"maps.{key}.dir missing; maps has {sorted(metrics.get('maps', {}))}")
    return Path(entry["dir"])


def latest(directory, pattern):
    """Refinement volumes carry an iteration number, so take the highest one."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} in {directory}")
    return str(matches[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="manifest roots holding empiar_<id>/<setting>/<condition>/ "
                             "(default: $RAPICK_WORK)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--setting", default="full",
                        help="`full` (the whole deposition) or `annot` (the 300 "
                             "annotated micrographs)")
    parser.add_argument("--entries", nargs="+", default=ENTRIES)
    args = parser.parse_args()

    roots = manifest_roots(args.experiments)
    rows, skipped = [], []
    for entry in args.entries:
        panels = []
        for label, condition in ARMS:
            metrics_path = find_metrics(roots, entry, condition, args.setting)
            if metrics_path is None:
                skipped.append(f"{entry}/{label}: no metrics.json in any root")
                continue
            try:
                metrics = json.loads(metrics_path.read_text())
                refine, locres = job_dir(metrics, "refine"), job_dir(metrics, "local_res")
                panels.append({
                    "label": label,
                    "map": latest(refine, f"{refine.name}_*_volume_map_sharp.mrc"),
                    "locres": latest(locres, f"{locres.name}_map_locres.mrc"),
                    # Local resolution is only meaningful inside the refinement mask; the
                    # renderer uses this to bound the row's colour range.
                    "mask": latest(locres, f"{locres.name}_mask_refine.mrc"),
                })
            except (KeyError, FileNotFoundError, ValueError) as error:
                skipped.append(f"{entry}/{label}: {error}")
        if panels:
            rows.append({"entry": entry, "panels": panels})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {args.out}: {sum(len(r['panels']) for r in rows)} panels "
          f"over {len(rows)} entries")
    for note in skipped:
        print(f"  skipped {note}")


if __name__ == "__main__":
    main()
