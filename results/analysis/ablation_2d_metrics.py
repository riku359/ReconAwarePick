#!/usr/bin/env python3
"""2D detection scores of the five ablation conditions.

For each condition of the ablation table this rebuilds the particle stack that condition
delivers to reconstruction, converts it to a top-left-origin STAR, and scores it against
the CryoPPP annotations with `src/rapick/eval/calc_common_2d_metrics.py`: the same
matcher the 2D detection table uses (distance-ascending greedy, one to one,
radius = diameter / 2, macro over the annotated micrographs). The scorer only evaluates
micrographs the annotations cover, so feeding it the full-set stack restricts it to the
annotated subset by itself.

Where each condition's stack comes from:

    baseline, mask      `normalized.star` of that condition's manifest directory. It is
                        RELION bottom-origin, so y -> H - y with H taken from the import
                        job's `micrograph_shape[0]`. Flipping is not optional: without
                        it the scorer's orientation check warns and the macro F1 on
                        EMPIAR-10081 falls from 0.610 to 0.206.
    select, both, fb    `J<n>_passthrough_particles_selected.cs` of that condition's
                        final 2D-selection job, with x = center_x_frac * shape[1] and
                        y = shape[0] * (1 - center_y_frac). That axis order was checked
                        against `normalized.star` on one micrograph: it matches exactly,
                        and the other order is off by about 100 px.

Backs: the 2D precision, recall and F1 of the five ablation conditions, and the
observation that the ablation's resolution ordering is not the 2D ordering.

Reads the CryoSPARC project directory, `$RAPICK_WORK/empiar_<id>/full/<condition>/`, and
the annotations under `$RAPICK_DATA`.

    python ablation_2d_metrics.py --project-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

from rapick.eval import calc_common_2d_metrics as ccm  # noqa: E402

CONDITIONS = ("baseline", "mask", "select", "both", "fb")

# The raw-picks import job of each entry, which carries the micrograph shape.
IMPORT_JOBS = {"10081": "J11", "10093": "J56", "10345": "J48", "10532": "J15"}

# The final 2D-selection job of the three selected conditions. These are the authors'
# instance; a fresh run produces different uids, so pass --job <id>.<condition>=<uid>.
SELECT_JOBS = {
    "select": {"10081": "J94", "10093": "J307", "10345": "J214", "10532": "J156"},
    "both": {"10081": "J131", "10093": "J345", "10345": "J271", "10532": "J148"},
    "fb": {"10081": "J496", "10093": "J569", "10345": "J501", "10532": "J554"},
}

# The stack size each condition should deliver, from the particle table. A stack that
# comes out a different size is a different run, and scoring it would silently report
# numbers for something else.
EXPECT = {
    "baseline": {"10081": 259335, "10093": 754434, "10345": 494061, "10532": 604430},
    "mask": {"10081": 244924, "10093": 753440, "10345": 484649, "10532": 576473},
    "select": {"10081": 131926, "10093": 271886, "10345": 26005, "10532": 184103},
    "both": {"10081": 126181, "10093": 301589, "10345": 29214, "10532": 178722},
    "fb": {"10081": 127859, "10093": 295644, "10345": 27838, "10532": 208931},
}

STAR_HEADER = ("data_particles\n\nloop_\n_rlnMicrographName #1\n"
               "_rlnCoordinateX #2\n_rlnCoordinateY #3\n")


def as_str(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def star_from_cs(project: Path, job: str, out: Path) -> int:
    """A select_2D passthrough .cs -> a top-left-origin GT-aligned STAR."""
    arr = np.load(project / job / f"{job}_passthrough_particles_selected.cs",
                  allow_pickle=True)
    paths = arr["location/micrograph_path"]
    xf = arr["location/center_x_frac"]
    yf = arr["location/center_y_frac"]
    shape = arr["location/micrograph_shape"]
    with out.open("w") as handle:
        handle.write(STAR_HEADER)
        for i in range(len(arr)):
            height, width = shape[i][0], shape[i][1]
            handle.write("%s %.3f %.3f\n" % (os.path.basename(as_str(paths[i])),
                                             xf[i] * width, height - yf[i] * height))
    return len(arr)


def star_flipped(src: Path, height: int, out: Path) -> int:
    """A bottom-origin STAR -> the same picks with y measured from the top."""
    n = 0
    with out.open("w") as handle:
        handle.write(STAR_HEADER)
        for line in src.open():
            tokens = line.split()
            if len(tokens) >= 3 and tokens[0].endswith(".mrc"):
                handle.write("%s %s %.3f\n"
                             % (tokens[0], tokens[1], height - float(tokens[2])))
                n += 1
    return n


def parse_job_overrides(items):
    out = {}
    for item in items or ():
        target, _, uid = item.partition("=")
        entry, _, condition = target.partition(".")
        if not uid or condition not in SELECT_JOBS or entry not in SELECT_JOBS[condition]:
            raise SystemExit(f"--job expects <entry>.<condition>=<uid> with condition in "
                             f"{sorted(SELECT_JOBS)}, got: {item}")
        out[(entry, condition)] = uid
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--ids", nargs="+", default=list(analysis_env.CORE_IDS))
    ap.add_argument("--setting", default="full",
                    help="which manifest setting holds the baseline and mask stacks")
    ap.add_argument("--job", action="append", default=[], metavar="ENTRY.CONDITION=UID",
                    help="override one 2D-selection job uid, e.g. 10081.fb=J9")
    ap.add_argument("--import-job", action="append", default=[], metavar="ID=UID",
                    help="override the import job the micrograph height is read from")
    ap.add_argument("--no-check-counts", action="store_true",
                    help="score even when a stack is not the size the paper reports")
    ap.add_argument("--out", default=None,
                    help="default $RAPICK_WORK/analysis/ablation_2d_metrics.json")
    args = ap.parse_args()

    project = analysis_env.project_dir(args.project_dir)
    overrides = parse_job_overrides(args.job)
    import_jobs = dict(IMPORT_JOBS, **dict(i.split("=", 1) for i in args.import_job))

    out_file = analysis_env.out_path("ablation_2d_metrics.json", args.out)
    work = out_file.parent / "ablation_2d_stars"
    work.mkdir(parents=True, exist_ok=True)

    result = {"entries": {}}
    for eid in args.ids:
        imported = np.load(project / import_jobs[eid] / "imported_particles.cs",
                           allow_pickle=True)
        height = int(imported["location/micrograph_shape"][0][0])

        counts = {}
        for condition in ("baseline", "mask"):
            src = analysis_env.manifest_dir(eid, args.setting, condition) / "normalized.star"
            if not src.is_file():
                raise SystemExit(f"missing {src}")
            counts[condition] = star_flipped(src, height, work / f"{eid}_{condition}.star")
        for condition in ("select", "both", "fb"):
            job = overrides.get((eid, condition), SELECT_JOBS[condition][eid])
            counts[condition] = star_from_cs(project, job, work / f"{eid}_{condition}.star")

        for condition, n in counts.items():
            expected = EXPECT[condition].get(eid)
            if expected is not None and n != expected and not args.no_check_counts:
                raise SystemExit("%s %s: %d particles, expected %d. A different stack "
                                 "scores something the paper does not report; pass "
                                 "--no-check-counts to score it anyway."
                                 % (eid, condition, n, expected))

        result["entries"][eid] = {}
        for condition in CONDITIONS:
            scored = ccm.run_single(int(eid), str(work / f"{eid}_{condition}.star"),
                                    None, None, 0.5, False)
            result["entries"][eid][condition] = scored
            print(eid, condition, round(scored["macro_P"], 4), round(scored["macro_R"], 4),
                  round(scored["macro_F1"], 4), scored["n_pred_eval"])

    out_file.write_text(json.dumps(result, indent=1))
    print("saved", out_file)


if __name__ == "__main__":
    main()
