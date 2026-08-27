#!/usr/bin/env python3
"""Turn a 2D selection into the next round's training labels (paper Sec. 3.5).

Run it with the interpreter that has cryosparc-tools (the `recon` environment):

  python -m rapick.loop.export_teacher_star \\
      --project P1 --select2d J244 --extract J100 --empiar 10081 \\
      --input-star $RAPICK_WORK/loop/10081/round0/cryotransformer_clean_tri.star \\
      --seed 1 --num-mics 50 --out-dir $RAPICK_WORK/loop/10081/round0

Writes into --out-dir:
  teacher.star     GT-aligned STAR of the surviving particles on the sampled micrographs
  train_mics.txt   the sampled micrograph names, one per line
  summary.json     counts + the coordinate-convention check

The teacher set is 50 micrographs drawn with a fixed per-round seed from the micrographs
that carry surviving particles -- a micrograph whose every pick was rejected is not a
zero-particle training example -- and fine-tuning splits them 40 train / 10 validation.

Why the sampling lives here and not in the fine-tuner: its own micrograph subsampling
happens *after* the train/validation split, so 50 sampled there would not become 40 + 10.
Sampling up front also leaves the chosen 50 on disk, per round, as a reproduction input.

Coordinates. The picks were imported with a Y flip (`ny - Y`), and Import Particles then
stores center_y_frac = Y_normalized / ny with no further flip. So the GT-aligned
(top-left origin) coordinates come back as

    X_gt = center_x_frac * nx
    Y_gt = ny * (1 - center_y_frac)

That inverse is derived, not guessed -- but a silent convention change upstream would
corrupt every label without failing, so --extract makes the script prove it: every
extracted particle must land on a pick of the input STAR. The annotation is deliberately
not used for this check; it exists for these entries but would not generalise to one
without it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from . import paths, star
from .common import connect_cryosparc


def load_particle_coords(job, output_name: str):
    """(mic_key, X_gt, Y_gt) per particle, in the GT-aligned top-left convention."""
    parts = job.load_output(output_name, slots=["location"])
    mic_keys = [star.normalize_mic_name(str(p)) for p in parts["location/micrograph_path"]]
    shapes = np.array(parts["location/micrograph_shape"])          # (H, W) per particle
    ny = shapes[:, 0].astype(float)
    nx = shapes[:, 1].astype(float)
    x_gt = np.array(parts["location/center_x_frac"], dtype=float) * nx
    y_gt = ny * (1.0 - np.array(parts["location/center_y_frac"], dtype=float))
    return mic_keys, x_gt, y_gt


def check_against_input_star(mic_keys, x_gt, y_gt, star_path: Path, tol: float):
    """Fraction of particles that land on a pick of the input STAR.

    Extraction legitimately drops particles whose box would cross the micrograph edge,
    so this is a subset test, not an equality test: every *extracted* particle must be a
    pick, but not every pick survives extraction.
    """
    picks = star.load_star_points(str(star_path))          # {mic_key: [(x, y), ...]}
    grids = {mic: (star.build_grid(pts, tol), pts) for mic, pts in picks.items()}

    hits = 0
    for key, px, py in zip(mic_keys, x_gt, y_gt):
        found = grids.get(key)
        if found and star.within(px, py, found[0], found[1], tol):
            hits += 1
    return hits / max(len(mic_keys), 1), len(picks)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--select2d", required=True,
                    help="completed select_2D job whose kept particles become the labels")
    ap.add_argument("--extract", default=None,
                    help="the extract job under the same stack; used only to prove the "
                         "coordinate convention against --input-star (skipped if omitted)")
    ap.add_argument("--input-star", default=None,
                    help="the STAR that was imported for this stack (the convention "
                         "check's reference); required together with --extract")
    ap.add_argument("--empiar", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="micrograph-sampling seed; the loop passes the round number + 1, "
                         "so each round draws a different 50 and every rerun draws the "
                         "same 50 as the run before it")
    ap.add_argument("--num-mics", type=int, default=50)
    ap.add_argument("--mics-from", default=None,
                    help="use exactly the micrographs named in this file, one per line, "
                         "instead of sampling. For an arm that must train on the same "
                         "micrographs as another arm, where the sampling pool differs")
    ap.add_argument("--all-mics", action="store_true",
                    help="use every micrograph with surviving particles instead of "
                         "sampling --num-mics of them")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tolerance-px", type=float, default=2.0,
                    help="how far an extracted particle may sit from its originating pick "
                         "and still count as matched (see check_against_input_star)")
    ap.add_argument("--min-match-frac", type=float, default=0.99,
                    help="abort if fewer than this fraction of extracted particles land "
                         "on an input-STAR pick")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cs = connect_cryosparc(paths.load_env())
    project = cs.find_project(args.project)

    # Prove the coordinate convention on the full extracted set before trusting the
    # subset. A wrong inverse would still produce a plausible-looking STAR.
    check = None
    if args.extract:
        if not args.input_star:
            sys.exit("--extract needs --input-star")
        # 2 px, not the scorer's particle radius: the round trip is exact. Measured over
        # 71,814 particles, the residual to the originating pick is 0.0002 px at worst --
        # float32 in the .cs fractions and nothing else. Matching at a particle radius
        # would still pass while quietly off by tens of pixels.
        tol = args.tolerance_px
        ex_keys, ex_x, ex_y = load_particle_coords(project.find_job(args.extract),
                                                   "particles")
        frac, n_mic_star = check_against_input_star(
            ex_keys, ex_x, ex_y, Path(args.input_star).expanduser(), tol)
        check = {"extract_job": args.extract, "n_extract": len(ex_keys),
                 "input_star": str(args.input_star), "n_micrographs_in_star": n_mic_star,
                 "match_frac": round(frac, 5), "tolerance_px": tol}
        print(f"[check] {frac:.4%} of {len(ex_keys)} extracted particles land on an "
              f"input-STAR pick (tol {tol:.0f} px)")
        if frac < args.min_match_frac:
            sys.exit(f"error: coordinate convention check failed ({frac:.4%} < "
                     f"{args.min_match_frac:.2%}). Labels would be wrong -- stopping.")

    keys, xs, ys = load_particle_coords(project.find_job(args.select2d),
                                        "particles_selected")
    by_mic: dict = {}
    for key, px, py in zip(keys, xs, ys):
        by_mic.setdefault(key, []).append((px, py))
    print(f"[select2d] {args.select2d}: {len(keys)} surviving particles on "
          f"{len(by_mic)} micrographs")

    # The retraining micrographs are drawn from those holding surviving particles, so a
    # micrograph whose every pick was rejected is not a 0-particle training example.
    pool = sorted(by_mic)
    if args.mics_from:
        want = [ln.strip() for ln in open(args.mics_from) if ln.strip()]
        chosen = sorted(m for m in want if m in by_mic)
        missing = [m for m in want if m not in by_mic]
        if missing:
            print(f"[mics-from] {len(missing)} of {len(want)} named micrographs carry no "
                  f"surviving particles here and are dropped: {missing[:3]}")
        if not chosen:
            sys.exit(f"error: none of the micrographs in {args.mics_from} carry survivors")
        print(f"[mics-from] using {len(chosen)} micrographs from {args.mics_from}")
    elif args.all_mics:
        chosen = pool
    else:
        if len(pool) < args.num_mics:
            sys.exit(f"error: only {len(pool)} micrographs carry surviving particles, "
                     f"need {args.num_mics}")
        chosen = sorted(random.Random(args.seed).sample(pool, args.num_mics))

    rows = []
    for mic in chosen:
        for px, py in by_mic[mic]:
            rows.append((mic, int(round(px)), int(round(py))))

    star_path = out_dir / "teacher.star"
    with star_path.open("w") as fh:
        fh.write(star.HEADER)
        for mic, px, py in rows:
            fh.write(f"{mic}.mrc {px} {py}\n")

    (out_dir / "train_mics.txt").write_text("\n".join(chosen) + "\n")

    summary = {
        "project": args.project, "select2d": args.select2d, "empiar": args.empiar,
        "seed": args.seed, "num_mics": "all" if args.all_mics else args.num_mics,
        "n_surviving_particles": len(keys),
        "n_micrographs_with_survivors": len(by_mic),
        "n_teacher_particles": len(rows),
        "particles_per_mic": round(len(rows) / len(chosen), 1),
        "convention_check": check,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[write] {star_path}: {len(rows)} particles on {len(chosen)} micrographs "
          f"({summary['particles_per_mic']}/mic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
