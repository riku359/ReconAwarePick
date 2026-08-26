"""Coordinate normalization: make a source star CryoSPARC-ready.

Built-in Import Particles uses center_y_frac = Y / ny with NO flip, but the
CryoPPP GT-aligned stars use a top-left (image) Y origin. So Y must be flipped to
`ny - Y` before import (verified: Extract centres particles only after this flip;
leaving it alone, and swapping the axes as well as flipping, both miss). We assume
a dataset's micrographs share one height (true for CryoPPP, per EMPIAR id).

This package owns the normalization rather than requiring pre-flipped stars
upstream, because the flip needs the micrograph height.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import os
import struct
from pathlib import Path

# MRC `mode` -> bytes per pixel, for the density modes CryoPPP micrographs use. An
# unlisted mode is treated as an implausible header (see micrograph_defect).
_MRC_BYTES_PER_MODE = {0: 1, 1: 2, 2: 4, 6: 2}


def read_mrc_shape(path: str) -> tuple[int, int]:
    """(nx, ny) = (width, height) from an MRC header's first two int32."""
    with open(path, "rb") as fh:
        nx, ny = struct.unpack("<ii", fh.read(8))
    return nx, ny


def dataset_micrograph_height(glob_pattern: str) -> int:
    """ny (height) of the first micrograph matching the glob."""
    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"no micrographs match {glob_pattern}")
    return read_mrc_shape(files[0])[1]


def micrograph_defect(path: str) -> str | None:
    """Return a reason string if the mrc is truncated/corrupt, else None.

    Catches the header-only stubs (e.g. 4096-byte files) that a naive `*.mrc` glob
    happily matches but that carry no image data: their 1024-byte header still reads,
    so import accepts them and only CTF later drops them — silently."""
    try:
        with open(path, "rb") as fh:
            nx, ny, nz, mode = struct.unpack("<iiii", fh.read(16))
    except Exception as exc:
        return f"unreadable header ({exc})"
    bytes_per_px = _MRC_BYTES_PER_MODE.get(mode)
    if not (bytes_per_px and nx > 0 and ny > 0 and nz > 0):
        return f"implausible header nx={nx} ny={ny} nz={nz} mode={mode}"
    min_size = 1024 + nx * ny * nz * bytes_per_px
    actual = os.path.getsize(path)
    if actual < min_size:
        return f"truncated {actual}B < {min_size}B for {nx}x{ny}x{nz}"
    return None


def micrograph_set_fingerprint(glob_pattern: str) -> str:
    """Identity of the imported micrograph set: sha256 over each matched file's
    (basename, size). Changes if a micrograph is added, removed, or replaced (a
    truncated stub swapped for the full file changes its size) — so a shared Import
    Micrographs job is not reused across a changed input set."""
    h = hashlib.sha256()
    for path in sorted(_glob.glob(glob_pattern)):
        h.update(os.path.basename(path).encode())
        h.update(b"\0")
        h.update(str(os.path.getsize(path)).encode())
        h.update(b"\n")
    return h.hexdigest()


def _find_coord_loop(lines: list) -> tuple[int, dict]:
    """Return (first_data_line_idx, {label: column_index}) for the loop_ block
    that contains _rlnCoordinateX."""
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols: dict = {}
            while j < len(lines) and lines[j].strip().startswith("_"):
                cols[lines[j].split()[0].lstrip("_")] = len(cols)
                j += 1
            if any(c.startswith("rlnCoordinateX") for c in cols):
                return j, cols
            i = j
        else:
            i += 1
    raise ValueError("no loop_ containing _rlnCoordinateX found in star")


def _is_data_row(s: str) -> bool:
    return bool(s) and not s.startswith(("_", "#", "data_")) and s != "loop_"


def normalize_star(src_star: str, out_star: str | Path, ny: int, y_flip: bool) -> str:
    """Write a CryoSPARC-ready copy of `src_star`. If y_flip, replace each
    _rlnCoordinateY with (ny - Y). Micrograph names are left untouched. Returns
    the path actually used for import (src_star itself when no transform)."""
    if not y_flip:
        return src_star

    lines = Path(src_star).read_text().splitlines()
    data_start, cols = _find_coord_loop(lines)
    yi = next(cols[c] for c in cols if c.startswith("rlnCoordinateY"))

    out_lines = lines[:data_start]
    for ln in lines[data_start:]:
        s = ln.strip()
        if not _is_data_row(s):
            out_lines.append(ln)
            continue
        parts = ln.split()
        if len(parts) > yi:
            parts[yi] = f"{ny - float(parts[yi]):.6f}"
        out_lines.append("  ".join(parts))

    out_star = Path(out_star)
    out_star.parent.mkdir(parents=True, exist_ok=True)
    out_star.write_text("\n".join(out_lines) + "\n")
    return str(out_star)
