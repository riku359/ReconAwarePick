#!/usr/bin/env python3
"""Local resolution taken again inside a mask common to every condition of an entry.

The per-condition local-resolution quartiles are measured inside each map's own
refinement mask. Those masks are generated per refinement, so they differ in size
between conditions, and a looser mask picks up more of the unresolved periphery and
worsens the median. That makes a comparison across conditions partly a comparison of
mask sizes.

This intersects the five conditions' refinement masks per entry and takes the quartiles
again inside that common mask. Nothing new is computed in CryoSPARC: it reads the
existing `*_map_locres.mrc` and `*_mask_refine.mrc` of the same jobs.

Backs: the local-resolution quartiles quoted for the map-quality comparison, and the
statement that the ranking survives being measured inside one mask.

Reads a per-condition index of refine and local-resolution job uids (the shape of
`results/tables/revision/locres.json`) plus the CryoSPARC project directory. `--conditions`
has to name the `source` values that index actually uses, which is not necessarily the
release condition vocabulary.

    python locres_common_mask.py --project-dir <dir> --index locres.json
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

# The five conditions the map-quality comparison carries, in table order.
CONDITIONS = ("cryolo", "topaz", "cryosegnet", "baseline", "fb")

MRC_DTYPES = {2: np.float32, 1: np.int16, 0: np.int8, 6: np.uint16, 12: np.float16}


def read_mrc(path: Path):
    """One MRC volume, with the grid and pixel size read off the header."""
    with open(path, "rb") as handle:
        head = handle.read(1024)
        nx, ny, nz, mode = struct.unpack("<4i", head[:16])
        mx, my, mz = struct.unpack("<3i", head[28:40])
        xlen, ylen, zlen = struct.unpack("<3f", head[40:52])
        nsymbt = struct.unpack("<i", head[92:96])[0]
        handle.seek(1024 + nsymbt)
        dtype = MRC_DTYPES[mode]
        data = np.frombuffer(handle.read(nx * ny * nz * np.dtype(dtype).itemsize),
                             dtype=dtype)
    volume = data.reshape(nz, ny, nx).astype(np.float32)
    apix = xlen / mx if mx else 0.0
    return volume, {"shape": [nz, ny, nx], "apix": round(float(apix), 4)}


def load_job(project: Path, job_uid: str):
    """The local-resolution volume and refinement mask of one local-resolution job."""
    directory = project / job_uid
    locres = sorted(directory.glob("*_map_locres.mrc"))
    mask = sorted(directory.glob("*_mask_refine.mrc"))
    if not locres or not mask:
        return None
    volume, meta = read_mrc(locres[0])
    mask_volume, mask_meta = read_mrc(mask[0])
    return volume, mask_volume, meta, mask_meta


def quartiles(values):
    q = np.percentile(values, [25, 50, 75])
    return {"n_vox": int(values.size), "p25": float(q[0]),
            "median": float(q[1]), "p75": float(q[2])}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--index", required=True, type=Path,
                    help="JSON keyed '<dataset>|<condition>' with `refine` and "
                         "`local_res` job uids, as results/tables/revision/locres.json is")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--out", default=None,
                    help="default $RAPICK_WORK/analysis/locres_common.json")
    args = ap.parse_args()

    project = analysis_env.project_dir(args.project_dir)
    source = json.loads(args.index.read_text())
    by_dataset = {}
    for record in source.values():
        if "median" not in record:
            continue
        by_dataset.setdefault(record["dataset"], {})[record["source"]] = record

    result = {}
    for dataset in sorted(by_dataset):
        records = by_dataset[dataset]
        missing = [c for c in args.conditions if c not in records]
        if missing:
            print("%s: missing %s -- skip" % (dataset, missing), flush=True)
            continue

        loaded, grids, ok = {}, {}, True
        for condition in args.conditions:
            got = load_job(project, records[condition]["local_res"])
            if got is None:
                print("%s/%s: no locres or mask mrc" % (dataset, condition), flush=True)
                ok = False
                break
            volume, mask, meta, _ = got
            loaded[condition] = (volume, mask)
            grids[condition] = meta
        if not ok:
            continue

        shapes = {tuple(g["shape"]) for g in grids.values()}
        apixes = {g["apix"] for g in grids.values()}
        if len(shapes) != 1:
            print("%s: grids differ %s -- skip" % (dataset, grids), flush=True)
            result[dataset] = {"error": "grid mismatch", "grids": grids}
            continue

        common = np.ones(next(iter(loaded.values()))[0].shape, dtype=bool)
        per_condition = {}
        for condition in args.conditions:
            volume, mask = loaded[condition]
            own = mask > 0.5
            common &= own
            values = volume[own]
            values = values[np.isfinite(values) & (values > 0)]
            per_condition[condition] = {"own_mask": quartiles(values),
                                        "own_mask_vox": int(own.sum())}
        n_common = int(common.sum())
        for condition in args.conditions:
            volume, _ = loaded[condition]
            values = volume[common]
            values = values[np.isfinite(values) & (values > 0)]
            per_condition[condition]["common_mask"] = (quartiles(values) if values.size
                                                       else {"error": "empty"})
            per_condition[condition]["refine"] = records[condition]["refine"]
            per_condition[condition]["local_res"] = records[condition]["local_res"]

        result[dataset] = {
            "grid": sorted(shapes)[0],
            "apix": sorted(apixes),
            "n_common_vox": n_common,
            "own_mask_vox": {c: per_condition[c]["own_mask_vox"] for c in args.conditions},
            "sources": per_condition,
        }
        print("%s: common %d vox (own %s)"
              % (dataset, n_common,
                 {c: per_condition[c]["own_mask_vox"] for c in args.conditions}), flush=True)
        for condition in args.conditions:
            a, b = per_condition[condition]["own_mask"], per_condition[condition]["common_mask"]
            print("   %-16s own %.2f/%.2f/%.2f -> common %.2f/%.2f/%.2f"
                  % (condition, a["p25"], a["median"], a["p75"],
                     b.get("p25", float("nan")), b.get("median", float("nan")),
                     b.get("p75", float("nan"))), flush=True)

    out = analysis_env.out_path("locres_common.json", args.out)
    out.write_text(json.dumps(result, indent=1))
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
