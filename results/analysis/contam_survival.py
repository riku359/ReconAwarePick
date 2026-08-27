#!/usr/bin/env python3
"""What fraction of the particles that survive 2D class selection land on contamination.

The `select` condition applies the 2D class selection without the contamination mask.
This takes the particles left in its final `select_2D` and asks how many of them fall
inside the stored contamination mask, which is the set the `mask` condition would have
removed.

The claim in the related work is that contamination survives classification as a
coherent false-positive class and can enter the pseudo-labels as a positive. What is
counted here is that leakage itself.

This measures whether the 2D class selection failed to remove what the mask would have
removed. It is not a measurement of the mask's accuracy: the mask is used as the
definition of the set the `mask` condition removes, not as ground truth for
contamination.

The coordinate convention has to match the production filter. That filter resizes the
mask back to micrograph resolution and tests `flipud(mask)[round(y), round(x)] >= 0.5`.
Whether CryoSPARC's `center_y_frac` is measured from the top or the bottom depends on
the import, so this first runs both conventions over the raw picks and adopts the one
that reproduces the mask-stage removal rates the particle table reports.

Backs: the contamination-leakage percentage quoted for the `select` condition.

Reads the CryoSPARC project directory, `$RAPICK_WORK/masks/<id>/`, and the output of
`mask_coverage.py` for the inverted-micrograph split.

    python contam_survival.py --project-dir <dir> --coverage mask_coverage.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

# The raw-picks import job of each entry, and the final select_2D of the `select`
# condition. These are the authors' instance; a fresh run produces different uids.
RAW_JOBS = {"10081": "J11", "10093": "J56", "10345": "J48", "10532": "J15"}
SELECT_JOBS = {"10081": "J94", "10093": "J307", "10345": "J214", "10532": "J156"}

# The mask-stage removal rate the particle table reports, as a percentage of raw picks.
# The coordinate convention is chosen as the one that reproduces these.
KNOWN_MASK_PCT = {"10081": 5.56, "10093": 0.13, "10345": 1.91, "10532": 4.63}

# A micrograph is called inverted when the mask covers more than this share of the frame,
# the same threshold mask_coverage.py's `gt_50pct` band uses.
INVERTED_COVERAGE = 0.50

THRESHOLD = 0.5

_cache = {}


def load_mask(mask_dir: Path, stem: str):
    key = (str(mask_dir), stem)
    if key in _cache:
        return _cache[key]
    path = mask_dir / f"{stem}_tri.npz"
    if not path.exists():
        _cache[key] = None
        return None
    with np.load(path, allow_pickle=False) as z:
        mask = (z["tri"].astype(np.float32) > THRESHOLD)
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = mask
    return mask


def stem_of(path: str) -> str:
    """A CryoSPARC micrograph path -> the npz stem.

    `J3/imported/000013423610471554956_HCN1apo_0375_2xaligned.mrc`
        -> `HCN1apo_0375_2xaligned`
    """
    base = Path(path).name
    if base.endswith(".mrc"):
        base = base[:-4]
    head, sep, tail = base.partition("_")
    if sep and head.isdigit() and len(head) > 12:
        base = tail
    return base


def read_particles(project: Path, job: str, kind: str):
    if kind == "import":
        return np.load(project / job / "imported_particles.cs", allow_pickle=True)
    return np.load(project / job / f"{job}_passthrough_particles_selected.cs",
                   allow_pickle=True)


def as_str(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def hits(arr, mask_dir: Path, flip: bool):
    """Per particle: inside the mask, and whether its micrograph has no stored mask."""
    paths = arr["location/micrograph_path"]
    xf = np.asarray(arr["location/center_x_frac"], dtype=np.float64)
    yf = np.asarray(arr["location/center_y_frac"], dtype=np.float64)
    inside = np.zeros(len(xf), dtype=bool)
    missing = np.zeros(len(xf), dtype=bool)
    order = np.argsort(paths, kind="stable")
    i = 0
    n_mic, n_missing_mic = 0, 0
    while i < len(order):
        j = i
        current = paths[order[i]]
        while j < len(order) and paths[order[j]] == current:
            j += 1
        idx = order[i:j]
        mask = load_mask(mask_dir, stem_of(as_str(current)))
        n_mic += 1
        if mask is None:
            missing[idx] = True
            n_missing_mic += 1
        else:
            ny, nx = mask.shape
            # The fractions are 0 to 1. Take the equivalent of round(y) on the mask grid.
            r = np.clip(np.rint((1.0 - yf[idx] if flip else yf[idx]) * ny - 0.5),
                        0, ny - 1).astype(np.int64)
            c = np.clip(np.rint(xf[idx] * nx - 0.5), 0, nx - 1).astype(np.int64)
            inside[idx] = mask[r, c]
        i = j
    return inside, missing, n_mic, n_missing_mic


def calibrate(project, ids, raw_jobs, out):
    """Pick the y convention that reproduces the known mask-stage removal rates."""
    calibration = {}
    for empiar in sorted(ids):
        arr = read_particles(project, raw_jobs[empiar], "import")
        mask_dir = analysis_env.mask_dir(empiar)
        row = {"n_raw": int(len(arr)), "known_mask_pct": KNOWN_MASK_PCT[empiar]}
        for flip in (False, True):
            inside, missing, n_mic, n_missing = hits(arr, mask_dir, flip)
            row["flip" if flip else "noflip"] = round(100.0 * inside.sum() / len(arr), 3)
            row["n_micrographs"] = n_mic
            row["n_micrographs_without_mask"] = n_missing
            row["n_particles_without_mask"] = int(missing.sum())
        calibration[empiar] = row
        print("%s raw=%d  noflip=%.2f%%  flip=%.2f%%  known=%.2f%%  (mic %d, no-mask %d)"
              % (empiar, row["n_raw"], row["noflip"], row["flip"], row["known_mask_pct"],
                 row["n_micrographs"], row["n_micrographs_without_mask"]), flush=True)
    error = {key: sum(abs(calibration[e][key] - calibration[e]["known_mask_pct"])
                      for e in calibration)
             for key in ("noflip", "flip")}
    convention = min(error, key=error.get)
    out["calibration"] = calibration
    out["convention"] = convention
    out["convention_abs_error_sum"] = error
    print("convention -> %s (abs err sum noflip=%.2f flip=%.2f)"
          % (convention, error["noflip"], error["flip"]), flush=True)
    return convention


def inverted_stems(coverage_file: Path, empiar: str):
    """The micrographs whose mask covers more than half the frame, from mask_coverage.py."""
    coverage = json.loads(Path(coverage_file).read_text())
    entry = coverage.get(empiar)
    if not entry:
        return set()
    return {row["micrograph"].removesuffix("_tri.npz")
            for row in entry["worst_25"] if row["coverage"] > INVERTED_COVERAGE}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--ids", nargs="+", default=list(analysis_env.CORE_IDS))
    ap.add_argument("--coverage", default=None,
                    help="mask_coverage.py's JSON, for the inverted-micrograph split")
    ap.add_argument("--raw-job", action="append", default=[], metavar="ID=UID",
                    help="override the raw-picks import job of one entry")
    ap.add_argument("--select-job", action="append", default=[], metavar="ID=UID",
                    help="override the final select_2D of one entry")
    ap.add_argument("--out", default=None,
                    help="default $RAPICK_WORK/analysis/contam_survival.json")
    args = ap.parse_args()

    project = analysis_env.project_dir(args.project_dir)
    raw_jobs = dict(RAW_JOBS, **dict(item.split("=", 1) for item in args.raw_job))
    select_jobs = dict(SELECT_JOBS, **dict(item.split("=", 1) for item in args.select_job))
    out_file = analysis_env.out_path("contam_survival.json", args.out)

    result = {"threshold": THRESHOLD, "convention": None, "calibration": {}, "entries": {}}
    convention = calibrate(project, args.ids, raw_jobs, result)
    out_file.write_text(json.dumps(result, indent=1))

    flip = (convention == "flip")
    for empiar in sorted(args.ids):
        arr = read_particles(project, select_jobs[empiar], "select")
        mask_dir = analysis_env.mask_dir(empiar)
        inside, missing, n_mic, n_missing = hits(arr, mask_dir, flip)
        n = len(arr)
        stems = inverted_stems(args.coverage, empiar) if args.coverage else set()
        paths = arr["location/micrograph_path"]
        is_inverted = np.array([stem_of(as_str(p)) in stems for p in paths], dtype=bool)
        keep = ~is_inverted
        record = {
            "select_job": select_jobs[empiar],
            "n_survivors": int(n),
            "n_in_mask": int(inside.sum()),
            "pct_in_mask": round(100.0 * inside.sum() / n, 3),
            "n_micrographs": n_mic,
            "n_micrographs_without_mask": n_missing,
            "n_particles_without_mask": int(missing.sum()),
            "n_inverted_micrographs_counted": len(stems),
            "n_particles_on_inverted": int(is_inverted.sum()),
            "excl_inverted": {
                "n_survivors": int(keep.sum()),
                "n_in_mask": int((inside & keep).sum()),
                "pct_in_mask": round(100.0 * (inside & keep).sum() / max(keep.sum(), 1), 3),
            },
            "on_inverted_only": {
                "n_survivors": int(is_inverted.sum()),
                "n_in_mask": int((inside & is_inverted).sum()),
                "pct_in_mask": (round(100.0 * (inside & is_inverted).sum()
                                      / is_inverted.sum(), 3)
                                if is_inverted.sum() else None),
            },
        }
        per_mic = {}
        for path in paths[inside]:
            stem = stem_of(as_str(path))
            per_mic[stem] = per_mic.get(stem, 0) + 1
        record["top_micrographs"] = sorted(per_mic.items(), key=lambda kv: -kv[1])[:15]
        record["n_micrographs_with_contamination"] = len(per_mic)
        result["entries"][empiar] = record
        print("%s survivors=%d in-mask=%d (%.2f%%)  excl-inverted %.2f%%  "
              "on-inverted %s%% (%d particles on %d inverted mic)"
              % (empiar, n, record["n_in_mask"], record["pct_in_mask"],
                 record["excl_inverted"]["pct_in_mask"],
                 record["on_inverted_only"]["pct_in_mask"],
                 record["n_particles_on_inverted"], len(stems)), flush=True)
        out_file.write_text(json.dumps(result, indent=1))

        # The coordinates of the leaked particles, so a micrograph can be looked at.
        csv = out_file.parent / f"contam_survivors_{empiar}.csv"
        xf = np.asarray(arr["location/center_x_frac"])
        yf = np.asarray(arr["location/center_y_frac"])
        with csv.open("w") as handle:
            handle.write("micrograph,x_frac,y_frac,on_inverted_micrograph\n")
            for k in np.nonzero(inside)[0]:
                handle.write("%s,%.6f,%.6f,%d\n"
                             % (stem_of(as_str(paths[k])), xf[k], yf[k], int(is_inverted[k])))
        print("   coords ->", csv, flush=True)

    out_file.write_text(json.dumps(result, indent=1))
    print("done ->", out_file, flush=True)


if __name__ == "__main__":
    main()
