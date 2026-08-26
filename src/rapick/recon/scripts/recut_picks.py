#!/usr/bin/env python3
"""Re-cut an existing picks.star, so an arm can change its operating point without re-picking.

The loop's own operating point (`predict.py --selection legacy_idxfix`) keeps the top 75%
of 600 queries per micrograph: a RELATIVE quota, so the pick count is set by the quota and
by NMS, never by how confident the model actually is. Two cuts are offered here, and the
pair is what makes a result readable:

  --score-min t    keep picks with _rlnAutopickFigureOfMerit >= t. An ABSOLUTE cut on the
                   sigmoid particle-probability, so the count is whatever the model's own
                   score distribution puts above t. This is the treatment: fewer picks AND
                   a higher precision.
  --subsample-n N  keep N picks drawn uniformly at random from the whole star. The count
                   matches a threshold arm while the junk fraction stays at the original
                   value. This is the control: without it, a threshold arm that fails to
                   improve cannot be told apart from "it just had fewer particles".

Uniform over the whole star, not per micrograph, on purpose: the threshold cut is global
too, and it removes more from the micrographs that are full of low-score picks. Matching
that with a per-micrograph quota would put the control on a different axis.

The header is copied verbatim and only data rows are dropped, so the output is the same
dialect the cleaner filter and CryoSPARC's import_particles already read.

  python recut_picks.py --star picks.star --out picks_t070.star --score-min 0.70
  python recut_picks.py --star picks.star --out picks_rand.star --subsample-n 167821 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCORE_COL = "_rlnAutopickFigureOfMerit"


def split_star(path: Path):
    """(header lines, data rows, 0-based index of the score column).

    A data row is one whose first field is a micrograph file, which is how every
    particle counter in this repository identifies one, so the counts agree.
    """
    header, rows, score_idx, col_n = [], [], None, 0
    for line in path.read_text().splitlines(keepends=True):
        first = line.split(None, 1)[0] if line.strip() else ""
        if first.endswith(".mrc"):
            rows.append(line)
            continue
        if first.startswith("_rln"):
            if first == SCORE_COL:
                score_idx = col_n
            col_n += 1
        header.append(line)
    return header, rows, score_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star", required=True, type=Path, help="picks.star to re-cut")
    ap.add_argument("--out", required=True, type=Path, help="where to write the re-cut star")
    ap.add_argument("--score-min", type=float, help=f"keep rows with {SCORE_COL} >= this")
    ap.add_argument("--subsample-n", type=int, help="keep this many rows, uniformly at random")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --subsample-n")
    args = ap.parse_args()

    if (args.score_min is None) == (args.subsample_n is None):
        ap.error("give exactly one of --score-min / --subsample-n")

    header, rows, score_idx = split_star(args.star)
    if not rows:
        ap.error(f"{args.star} has no particle rows")

    if args.score_min is not None:
        if score_idx is None:
            ap.error(f"{args.star} has no {SCORE_COL} column; an absolute cut is meaningless")
        kept = [r for r in rows if float(r.split()[score_idx]) >= args.score_min]
        cut = {"mode": "score_min", "score_min": args.score_min}
    else:
        if args.subsample_n > len(rows):
            ap.error(f"--subsample-n {args.subsample_n} exceeds the {len(rows)} rows present")
        # sorted() so the output keeps the input's micrograph grouping and score order;
        # a shuffled star would still import, but every diff against it would be noise.
        kept = [rows[i] for i in
                sorted(random.Random(args.seed).sample(range(len(rows)), args.subsample_n))]
        cut = {"mode": "subsample", "subsample_n": args.subsample_n, "seed": args.seed}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(header) + "".join(kept))

    mics_in = {r.split()[0] for r in rows}
    mics_out = {r.split()[0] for r in kept}
    summary = {**cut, "star_in": str(args.star), "star_out": str(args.out),
               "picks_in": len(rows), "picks_out": len(kept),
               "kept_fraction": round(len(kept) / len(rows), 5),
               "n_micrographs_in": len(mics_in), "n_micrographs_out": len(mics_out)}
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{len(kept):,} / {len(rows):,} picks kept ({len(kept) / len(rows):.1%}) "
          f"over {len(mics_out)}/{len(mics_in)} micrographs -> {args.out}")
    # A micrograph that loses every pick is not an error, but it silently shrinks the set
    # the downstream stages see, so it has to be visible in the log and not only the json.
    if len(mics_out) != len(mics_in):
        print(f"warn: {len(mics_in) - len(mics_out)} micrograph(s) kept no picks at all")


if __name__ == "__main__":
    raise SystemExit(main())
