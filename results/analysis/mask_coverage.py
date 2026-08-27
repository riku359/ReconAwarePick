#!/usr/bin/env python3
"""How often the contamination mask inverts, over every micrograph of the full sets.

The paper shows the mask inverting on EMPIAR-10532 with three examples and says it does
not measure the frequency. The masks are cached as npz for every micrograph, so the
frequency can be counted with no further inference.

Coverage is the fraction of pixels where the mask (float16, 0 to 1) exceeds 0.5. That is
the same threshold `src/rapick/cleaner/filter_star_from_masks.py` uses to decide whether
a pick is dropped, so a coverage of 0.9 means the mask would drop nine tenths of the
frame.

The thresholds for calling a mask inverted are printed as a ladder rather than fixed
here, because the distribution has to be looked at before one of them is chosen. The
paper's inversion count uses coverage above 50%.

Backs: the mask-inversion frequency quoted in the discussion of where the contamination
mask fails (Fig. 6).

Reads `$RAPICK_WORK/masks/<id>/*_tri.npz`. No CryoSPARC connection.

    python mask_coverage.py [--ids 10532] [--out mask_coverage.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

THRESHOLD = 0.5
HIST_EDGES = [0, .01, .02, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.0]
LADDER = (5, 10, 20, 30, 40, 50, 60, 70, 80)


def coverage_of(path: Path):
    with np.load(path, allow_pickle=False) as z:
        tri = z["tri"].astype(np.float32)
    return float((tri > THRESHOLD).mean()), tri.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=list(analysis_env.CORE_IDS))
    ap.add_argument("--out", default=None, help="default $RAPICK_WORK/analysis/mask_coverage.json")
    args = ap.parse_args()

    result = {}
    for empiar in args.ids:
        directory = analysis_env.mask_dir(empiar)
        files = sorted(directory.glob("*.npz"))
        if not files:
            print(f"no masks for {empiar} under {directory}", file=sys.stderr)
            continue
        coverage, shape = [], None
        for path in files:
            value, shape = coverage_of(path)
            coverage.append((path.name, value))
        values = np.array([c for _, c in coverage])
        q = np.percentile(values, [5, 25, 50, 75, 95])
        bands = {f"gt_{t:02d}pct": int((values > t / 100).sum()) for t in LADDER}
        worst = sorted(coverage, key=lambda kv: -kv[1])[:25]
        result[empiar] = {
            "n_micrographs": len(files),
            "mask_shape": list(shape),
            "mean": float(values.mean()),
            "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]),
            "max": float(values.max()), "min": float(values.min()),
            "counts_above": bands,
            "hist_edges": HIST_EDGES,
            "hist": np.histogram(values, bins=HIST_EDGES)[0].tolist(),
            "worst_25": [{"micrograph": k, "coverage": v} for k, v in worst],
        }
        print(f"{empiar}: n={len(files)} median={q[2]:.4f} p95={q[4]:.4f} "
              f"max={values.max():.4f} >50%={bands['gt_50pct']} "
              f">20%={bands['gt_20pct']} >10%={bands['gt_10pct']}", flush=True)

    out = analysis_env.out_path("mask_coverage.json", args.out)
    out.write_text(json.dumps(result, indent=1))
    print("done ->", out)


if __name__ == "__main__":
    main()
