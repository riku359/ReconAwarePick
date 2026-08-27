"""Reading and writing the GT-aligned STAR the loop passes between its stages.

The format is the repository's cross-picker convention: block `data_particles`, columns
`_rlnMicrographName, _rlnCoordinateX, _rlnCoordinateY` (plus an optional score column),
integer coordinates at micrograph scale, Y measured from the top-left. Every stage of a
round -- the picks, the contamination filter's survivors, the teacher labels, the
per-stage exports -- speaks it, which is what lets the five populations of a round be
compared as plain set operations on (micrograph, x, y).

These are the same rules as the cross-picker scorer's reader, restated here so that the
loop can parse and write its own inputs without importing a package it only shells out
to. Scoring itself is not duplicated: `run_loop.py` calls the scorer.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

HEADER = """data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
"""


def normalize_mic_name(raw: str) -> str:
    """`_rlnMicrographName` -> a comparison key: basename, leading `<digits>_` and
    `.mrc` removed.

    An annotation imported through CryoSPARC carries an import prefix of random digits
    (`>J1/imported/000...371_stack_..._DW.mrc`) that a picker's own STAR does not
    (`stack_..._DW.mrc`). Applying the same normalisation to both makes them one key.
    """
    mic = os.path.basename(raw)
    mic = re.sub(r"^\d+_", "", mic)
    if mic.endswith(".mrc"):
        mic = mic[:-4]
    return mic


def read_star_rows(path):
    """Read a STAR's `loop_` and return (column name -> index, list of row tokens).

    A block without coordinates (`data_optics`, say) is skipped and the loop carrying
    `_rlnCoordinateX` is the one adopted. The columns are bound the moment that header
    is seen rather than when the first data row arrives, so a STAR with zero rows -- a
    micrograph where the picker proposed nothing -- still reads as "coordinates
    present, rows empty" instead of losing its columns.
    """
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
            if s.startswith("_"):
                m = re.search(r"#(\d+)", s)
                idx = int(m.group(1)) - 1 if m else len(cur_cols)
                name = s.split()[0]
                cur_cols[name] = idx
                if name == "_rlnCoordinateX":
                    cols, rows = cur_cols, cur_rows   # adopt this loop; rows fills below
                continue
            if s.startswith("data_") or s.startswith("#"):
                in_loop = False
                continue
            if in_loop:
                cur_rows.append(s.split())
    return cols, rows


def load_star_points(path) -> dict:
    """One STAR as {micrograph key: [(x, y), ...]}. The score column is dropped."""
    cols, rows = read_star_rows(path)
    if "_rlnCoordinateX" not in cols or "_rlnCoordinateY" not in cols:
        raise ValueError(f"no coordinate columns in {path}")
    ix, iy = cols["_rlnCoordinateX"], cols["_rlnCoordinateY"]
    imic = cols.get("_rlnMicrographName")
    # A per-micrograph STAR may omit _rlnMicrographName; then the file name is the key.
    default_mic = normalize_mic_name(os.path.basename(str(path)))
    points: dict = {}
    for t in rows:
        if len(t) <= max(ix, iy):
            continue
        try:
            x, y = float(t[ix]), float(t[iy])
        except ValueError:
            continue
        mic = normalize_mic_name(t[imic]) if imic is not None and len(t) > imic else default_mic
        points.setdefault(mic, []).append((x, y))
    return points


def star_keys(path) -> set:
    """{(micrograph key, x_int, y_int)} for a GT-aligned STAR already on disk.

    Duplicate integer coordinates are fatal rather than deduplicated: the callers use
    these as sets, and a silently collapsed pair would understate one stage's count.
    """
    points = load_star_points(path)
    keys = {(mic, int(round(x)), int(round(y))) for mic, pts in points.items() for x, y in pts}
    n_rows = sum(len(pts) for pts in points.values())
    if len(keys) != n_rows:
        raise ValueError(f"{path} has {n_rows - len(keys)} duplicate integer coordinates")
    return keys


def write_star(path: Path, keys) -> None:
    """Write (micrograph, x, y) triples as a GT-aligned STAR, sorted."""
    with Path(path).open("w") as fh:
        fh.write(HEADER)
        for mic, x, y in sorted(keys):
            fh.write(f"{mic}.mrc {x} {y}\n")


def count_star_particles(star) -> int:
    """Particle rows in a GT-aligned STAR: the lines whose first field is a micrograph.

    The block header, `loop_` and the `_rln*` column declarations all sort out by that
    test, and no picker writes a comment line that would pass it.
    """
    n = 0
    with Path(star).open() as fh:
        for line in fh:
            first = line.split(None, 1)[0] if line.strip() else ""
            n += first.endswith(".mrc")
    return n


def micrograph_names(star) -> set:
    """The `.mrc` names a GT-aligned STAR mentions, exactly as written."""
    return {line.split()[0] for line in Path(star).read_text().splitlines()
            if line.split() and line.split()[0].endswith(".mrc")}


def build_grid(points, radius: float) -> dict:
    """Bucket points into `radius`-sized cells: {(cx, cy): [index, ...]}."""
    grid: dict = {}
    for i, (px, py) in enumerate(points):
        grid.setdefault((int(px // radius), int(py // radius)), []).append(i)
    return grid


def within(px: float, py: float, grid: dict, points, radius: float) -> bool:
    """Is (px, py) within `radius` of any point in the grid?"""
    cx, cy = int(px // radius), int(py // radius)
    r2 = radius * radius
    for gx in (cx - 1, cx, cx + 1):
        for gy in (cy - 1, cy, cy + 1):
            for gi in grid.get((gx, gy), ()):
                dx, dy = px - points[gi][0], py - points[gi][1]
                if dx * dx + dy * dy <= r2:
                    return True
    return False
