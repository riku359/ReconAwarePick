#!/usr/bin/env python3
"""Keep only the particle rows whose micrograph basename is in a given list.

Each round leaves the 50 micrographs it trained on in `train_mics.txt`, one name per
line, so this is what restricts a STAR to that list -- or, with the complement, to the
250 the round did not train on. It is also how a picker STAR is cut down to a sampled
subset of micrographs generally.

Row selection is by the BASENAME of `_rlnMicrographName` (a STAR may or may not carry a
path or an import prefix; the trailing basename is what matches), so it works both for a
picker's STAR of plain basenames and for a prefixed annotation. Header, columns and order
are preserved verbatim -- only data rows outside the list are dropped -- so the pipeline's
normalise/import see the same format as the full STAR.

  python -m rapick.loop.filter_star_by_micrograph_list \\
      --star picks.star --list train_mics.txt --out picks_train.star
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _mic_col_and_data_start(lines: list) -> tuple:
    """(micrograph column index, first data line index) for the `loop_` block that
    declares `_rlnMicrographName`."""
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols: dict = {}
            while j < len(lines) and lines[j].strip().startswith("_"):
                cols[lines[j].split()[0].lstrip("_")] = len(cols)
                j += 1
            mic = next((cols[c] for c in cols if c.startswith("rlnMicrographName")), None)
            if mic is not None:
                return mic, j
            i = j
        else:
            i += 1
    raise ValueError("no loop_ containing _rlnMicrographName found in star")


def _is_data_row(s: str) -> bool:
    return bool(s) and not s.startswith(("_", "#", "data_")) and s != "loop_"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star", required=True)
    ap.add_argument("--list", required=True, help="one micrograph basename per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--invert", action="store_true",
                    help="keep the rows NOT in the list (the held-out micrographs)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    listed = {ln.strip() for ln in Path(args.list).read_text().splitlines() if ln.strip()}
    lines = Path(args.star).read_text().splitlines()
    mic_i, data_start = _mic_col_and_data_start(lines)

    out = lines[:data_start]
    kept = 0
    seen_mics: set = set()
    for ln in lines[data_start:]:
        s = ln.strip()
        if not _is_data_row(s):
            out.append(ln)
            continue
        parts = s.split()
        if len(parts) <= mic_i:
            continue
        base = os.path.basename(parts[mic_i])
        # A list entry may be written with or without the .mrc extension; both mean the
        # same micrograph, and mixing the two is the easy way to silently keep nothing.
        in_list = base in listed or os.path.splitext(base)[0] in listed
        if in_list != args.invert:
            out.append(ln)
            kept += 1
            seen_mics.add(base)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"{args.out}: kept {kept} particles over {len(seen_mics)} micrographs "
          f"({'excluding' if args.invert else 'from'} a list of {len(listed)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
