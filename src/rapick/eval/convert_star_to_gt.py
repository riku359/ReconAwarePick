#!/usr/bin/env python3
"""
convert_star_to_gt.py -- convert an already-written picker STAR into CryoPPP GT format.

Any of the four pickers (crYOLO / Topaz / CryoTransformer / CryoSegNet) will do.
Columns are matched by their `_rln...` name, so the block name and any extra columns
(_rlnClassNumber / _rlnAnglePsi / _rlnDiameter, ...) are simply ignored.

The output matches the coordinate system and notation of the CryoPPP ground truth
($RAPICK_DATA/cryoppp/<ID>/ground_truth/empiar-<ID>_particles_selected.star):
  * block data_particles
  * columns = _rlnMicrographName, _rlnCoordinateX, _rlnCoordinateY
    [, _rlnAutopickFigureOfMerit] (kept when the source STAR has it; 3 columns if not)
  * coordinates = integers (px, mrc scale); Y flipped to a top-left origin as
    round(H - y) (all four pickers write a bottom origin natively; --no-flip-y disables)
  * _rlnMicrographName = "<micrograph>.mrc"
    (a per-micrograph STAR with no _rlnMicrographName takes its file stem)
  * granularity follows the input: a per-entry STAR stays one file, per-micrograph
    STAR files stay per-micrograph

H comes from the CryoPPP mrc header (ny); jpg and mrc share a scale, and H is constant
within an entry. The EMPIAR ID is taken from --empiar, or inferred from the path.

Usage:
  # one per-entry STAR (CryoTransformer / CryoSegNet)
  python convert_star_to_gt.py IN.star --out-dir OUT/ --empiar 10081
  # a directory of per-micrograph STAR files (crYOLO / Topaz native)
  python convert_star_to_gt.py picks/10081/*.star --out-dir OUT/10081 --empiar 10081

Environment: RAPICK_DATA must point at the downloaded input data
(docs/CONFIGURATION.md). There is no default; a missing variable is an error naming it.
"""
import argparse
import glob
import os
import re
import struct


def cryoppp_root():
    """$RAPICK_DATA/cryoppp -- the CryoPPP entries (micrographs + annotations)."""
    value = os.environ.get("RAPICK_DATA")
    if not value:
        raise SystemExit(
            "RAPICK_DATA is not set; it must point at the downloaded input data. "
            "See docs/CONFIGURATION.md.")
    return os.path.join(os.path.expanduser(value), "cryoppp")


def mrc_height(eid):
    """ny (= H) from the first CryoPPP .mrc header (constant within an entry), else None."""
    files = sorted(glob.glob(os.path.join(cryoppp_root(), str(eid), "micrographs", "*.mrc")))
    if not files:
        return None
    with open(files[0], "rb") as fh:
        _nx, ny = struct.unpack("<2i", fh.read(8))
    return ny


def infer_empiar(path):
    """Guess the EMPIAR ID (4-5 digits) from the path or file name; None if not found."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return stem
    m = re.search(r"EMPIAR[_-]?(\d{4,5})", path)
    if m:
        return m.group(1)
    m = re.search(r"empiar-(\d{4,5})", path)
    if m:
        return m.group(1)
    for part in os.path.normpath(path).split(os.sep):
        if part.isdigit() and 4 <= len(part) <= 5:
            return part
    return None


def read_star(path):
    """Read the STAR loop_ and return (dict name->column, list of data-row tokens).
    With several data_ blocks, the last loop containing _rlnCoordinateX wins."""
    cols, rows = {}, []
    cur_cols, cur_rows, in_loop = {}, [], False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s == "loop_":
                cur_cols, cur_rows, in_loop = {}, [], True
                continue
            if s.startswith("_rln"):
                m = re.search(r"#(\d+)", s)
                idx = int(m.group(1)) - 1 if m else len(cur_cols)
                cur_cols[s.split()[0]] = idx
                continue
            if s.startswith("data_") or s.startswith("#"):
                in_loop = False
                continue
            if in_loop:
                cur_rows.append(s.split())
                # keep a snapshot once this loop is known to carry coordinates
                if "_rlnCoordinateX" in cur_cols:
                    cols, rows = cur_cols, cur_rows
    return cols, rows


def write_gt_star(path, records, with_fom):
    """Write records=[(mic, xint, yint, fom_str_or_None)] in GT format."""
    with open(path, "w") as fh:
        fh.write("\ndata_particles\n\nloop_\n")
        fh.write("_rlnMicrographName #1\n")
        fh.write("_rlnCoordinateX #2\n")
        fh.write("_rlnCoordinateY #3\n")
        if with_fom:
            fh.write("_rlnAutopickFigureOfMerit #4\n")
        for mic, x, y, fom in records:
            if with_fom:
                fh.write(f"{mic} {x} {y} {fom}\n")
            else:
                fh.write(f"{mic} {x} {y}\n")


def convert_file(path, out_path, empiar, flip_y, keep_fom):
    eid = empiar or infer_empiar(path)
    if eid is None:
        raise SystemExit(f"cannot infer the EMPIAR ID from: {path} (pass --empiar)")
    height = mrc_height(eid)
    if height is None:
        raise SystemExit(
            f"{eid}: no mrc found, so H is unavailable ({cryoppp_root()}/{eid}/micrographs)")

    cols, rows = read_star(path)
    if "_rlnCoordinateX" not in cols or "_rlnCoordinateY" not in cols:
        raise SystemExit(f"no coordinate columns found in: {path}")
    ix, iy = cols["_rlnCoordinateX"], cols["_rlnCoordinateY"]
    imic = cols.get("_rlnMicrographName")
    ifom = cols.get("_rlnAutopickFigureOfMerit")
    with_fom = keep_fom and ifom is not None
    default_mic = os.path.splitext(os.path.basename(path))[0] + ".mrc"

    records = []
    for t in rows:
        try:
            x = int(round(float(t[ix])))
            y = float(t[iy])
        except (ValueError, IndexError):
            continue
        y = int(round(height - y)) if flip_y else int(round(y))
        mic = os.path.basename(t[imic]) if imic is not None and len(t) > imic else default_mic
        if not mic.endswith(".mrc"):
            mic += ".mrc"
        fom = t[ifom] if with_fom and len(t) > ifom else None
        records.append((mic, x, y, fom))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_gt_star(out_path, records, with_fom)
    return eid, len(records), with_fom


def main():
    ap = argparse.ArgumentParser(description="convert a picker STAR into CryoPPP GT format")
    ap.add_argument("inputs", nargs="+", help="input STAR (file/glob)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory (written under the same basename)")
    ap.add_argument("--empiar", default=None,
                    help="fix the EMPIAR ID (default: infer it from the path)")
    ap.add_argument("--no-flip-y", action="store_true",
                    help="disable the y -> H-y flip (input is already top-left)")
    ap.add_argument("--drop-fom", action="store_true",
                    help="drop _rlnAutopickFigureOfMerit and write only Mic/X/Y")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])

    grand = 0
    for p in paths:
        out_path = os.path.join(args.out_dir, os.path.basename(p))
        eid, n, with_fom = convert_file(
            p, out_path, args.empiar, not args.no_flip_y, not args.drop_fom)
        print(f"{os.path.basename(p)} [id={eid}] -> {out_path} : {n} particles"
              f" ({'Mic/X/Y+FoM' if with_fom else 'Mic/X/Y'})")
        grand += n
    print(f"TOTAL: {len(paths)} files, {grand} particles")


if __name__ == "__main__":
    main()
