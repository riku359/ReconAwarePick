#!/usr/bin/env python3
"""Build a round's teacher labels from the CryoPPP annotation instead of the picks.

The lower row of Table 7. `export_teacher_star.py` writes the particles that survived
contamination removal and 2D class selection; this writes, for exactly the same
micrographs, the annotated particles. Everything else about the round is unchanged --
same sample size, same fine-tuning, same operating point -- so the pair isolates the
quality of the teacher.

That makes it an upper bound rather than a feedback loop: nothing here depends on what
the picker proposed.

  python -m rapick.loop.make_gt_teacher --empiar 10081 \\
      --mics-from $RAPICK_WORK/loop/10081_fb_gt/round0/train_mics.txt \\
      --out-dir   $RAPICK_WORK/loop/10081_fb_gt/round0

Writes into --out-dir:
  teacher_gt.star          the annotation restricted to the listed micrographs
  gt_teacher_summary.json  counts, and the micrographs the STAR could not carry

The restriction reuses `filter_star_by_micrograph_list`, which matches on the basename of
`_rlnMicrographName` and preserves the header and column order verbatim -- the CryoPPP
annotation carries a CryoSPARC import prefix on those names, so a plain string compare
against the micrograph list would keep nothing.

A micrograph with no annotated particle cannot appear in a STAR at all, so the teacher
can cover fewer micrographs than the list names. That is not an error and it is not
hypothetical -- on EMPIAR-10093 one of the 50 has no annotated particle -- so the count
is recorded rather than assumed, under `dropped_empty_gt_mics`.

REIMPLEMENTED FROM A WRITTEN PROCEDURE. The script that produced the published Table 7
numbers was never committed; this rebuilds what it is documented to have done and has not
been run end to end in this form.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import entries, paths
from .filter_star_by_micrograph_list import _is_data_row, _mic_col_and_data_start
from .filter_star_by_micrograph_list import main as filter_star


def read_mic_list(path: str | Path) -> list:
    """The micrograph names of a round's `train_mics.txt`, one per line."""
    names = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not names:
        raise SystemExit(f"{path} lists no micrographs")
    return names


def star_micrographs(path: str | Path) -> dict:
    """{micrograph basename -> particle count} for a GT-aligned or CryoPPP STAR."""
    lines = Path(path).read_text().splitlines()
    mic_i, data_start = _mic_col_and_data_start(lines)
    counts: dict = {}
    for line in lines[data_start:]:
        row = line.strip()
        if not _is_data_row(row):
            continue
        fields = row.split()
        if len(fields) <= mic_i:
            continue
        base = os.path.basename(fields[mic_i])
        counts[base] = counts.get(base, 0) + 1
    return counts


def build_teacher(empiar: str, mics_from: str | Path, out_dir: str | Path,
                  gt_star: str | Path = None) -> dict:
    """Restrict the entry's annotation to the listed micrographs. Returns the summary."""
    gt_star = Path(gt_star) if gt_star else paths.gt_star(empiar)
    if not gt_star.is_file():
        raise SystemExit(f"no CryoPPP annotation for {empiar} at {gt_star}")

    out_dir = Path(out_dir)
    out_star = out_dir / "teacher_gt.star"
    filter_star(["--star", str(gt_star), "--list", str(mics_from), "--out", str(out_star)])

    listed = read_mic_list(mics_from)
    per_mic = star_micrographs(out_star)
    # A listed micrograph is covered when the STAR carries a row for it. Compare on the
    # stem so a list written with or without `.mrc` matches either way, exactly as the
    # filter itself does.
    covered = {os.path.splitext(base)[0] for base in per_mic}
    dropped = sorted(name for name in listed
                     if os.path.splitext(os.path.basename(name))[0] not in covered)

    n_particles = sum(per_mic.values())
    if not n_particles:
        raise SystemExit(
            f"the annotation carries no particle on any of the {len(listed)} micrographs "
            f"in {mics_from}; check that the list and {gt_star} name the same micrographs")

    summary = {
        "empiar": empiar,
        "teacher": entries.TEACHER_GT,
        "gt_star": str(gt_star),
        "mics_from": str(mics_from),
        "teacher_star": str(out_star),
        "n_micrographs_listed": len(listed),
        "n_micrographs_with_gt": len(per_mic),
        "n_teacher_particles": n_particles,
        "particles_per_mic": round(n_particles / len(per_mic), 3),
        # Listed micrographs the STAR cannot carry because they hold no annotated
        # particle. Recorded, not silently absorbed into the count above.
        "dropped_empty_gt_mics": dropped,
    }
    (out_dir / "gt_teacher_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--empiar", required=True, choices=sorted(entries.ENTRIES),
                    help="the entry whose CryoPPP annotation becomes the teacher")
    ap.add_argument("--mics-from", required=True, dest="mics_from",
                    help="the round's train_mics.txt: the micrographs the pseudo-label "
                         "teacher was drawn from, one name per line")
    ap.add_argument("--out-dir", required=True, dest="out_dir",
                    help="the round directory to write teacher_gt.star into")
    ap.add_argument("--gt-star", default=None, dest="gt_star",
                    help="override the annotation (default: the entry's CryoPPP STAR "
                         "under $RAPICK_DATA)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_teacher(args.empiar, args.mics_from, args.out_dir, args.gt_star)
    print(f"[gt-teacher] {args.empiar}: {summary['n_teacher_particles']:,} annotated "
          f"particles over {summary['n_micrographs_with_gt']} of "
          f"{summary['n_micrographs_listed']} micrographs "
          f"-> {summary['teacher_star']}")
    if summary["dropped_empty_gt_mics"]:
        print(f"[gt-teacher] {len(summary['dropped_empty_gt_mics'])} listed micrograph(s) "
              f"carry no annotated particle and are absent from the teacher: "
              f"{', '.join(summary['dropped_empty_gt_mics'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
