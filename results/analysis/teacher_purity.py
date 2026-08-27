#!/usr/bin/env python3
"""Purity of a feedback-loop teacher label set against the CryoPPP annotations.

Purity is precision: of the particles the round-0 teacher carries, the fraction that
match an annotation. Matching is one to one, nearest first, within radius = diameter / 2,
restricted to the micrographs the teacher covers. The annotation STAR carries
CryoSPARC-style micrograph names, so names are normalised before they are compared.

Both loop arms are scored, which is what the comparison is for: `fb` runs the
contamination mask inside the loop and `fb_nomask` does not, so the difference in teacher
purity is what the mask contributes to the labels.

Backs: the teacher-purity figures quoted for the contamination mask inside the loop.

Reads `$RAPICK_WORK/loop/<id>[_<arm>]/round<n>/teacher.star` and the annotations under
`$RAPICK_DATA`. No CryoSPARC connection.

    python teacher_purity.py --ids 10081 10532
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

from rapick.loop import entries                        # noqa: E402
from rapick.loop.star import load_star_points          # noqa: E402

TEACHER_STAR = "teacher.star"


def purity(empiar: str, star: Path):
    """(micrographs scored, teacher labels, matches, purity)."""
    radius = entries.ENTRIES[empiar].diameter_px / 2.0
    teacher = load_star_points(star)
    annotations = load_star_points(analysis_env.gt_star(empiar))

    matches = total = scored_mics = 0
    for mic, points in teacher.items():
        predicted = np.asarray(points, float)
        total += len(predicted)
        annotated = np.asarray(annotations.get(mic, []), float)
        if not len(annotated):
            continue
        scored_mics += 1
        distances = np.linalg.norm(predicted[:, None, :] - annotated[None, :, :], axis=2)
        pairs = sorted((distances[i, j], i, j)
                       for i in range(len(predicted)) for j in range(len(annotated))
                       if distances[i, j] <= radius)
        used_p, used_a = set(), set()
        for _distance, i, j in pairs:
            if i not in used_p and j not in used_a:
                used_p.add(i)
                used_a.add(j)
                matches += 1
    return scored_mics, total, matches, (matches / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=["10081", "10532"])
    ap.add_argument("--arms", nargs="+", default=sorted(entries.ARMS),
                    help="loop arms to score; `fb` masks inside the loop, "
                         "`fb_nomask` does not")
    ap.add_argument("--round", type=int, default=0,
                    help="which round's teacher labels (default 0)")
    args = ap.parse_args()

    for empiar in args.ids:
        for arm in args.arms:
            star = entries.round_dir(empiar, args.round, arm) / TEACHER_STAR
            if not star.exists():
                print(f"{empiar}  {arm:10s}  {TEACHER_STAR} not found ({star})")
                continue
            mics, total, matched, value = purity(empiar, star)
            print(f"{empiar}  {arm:10s}  mics={mics}  teacher={total:5d}  "
                  f"TP={matched:5d}  purity={value:.4f}")


if __name__ == "__main__":
    main()
