#!/usr/bin/env python3
"""Inspect a finished Patch CTF job before anything downstream consumes it.

`completed` does not mean correct. A Patch CTF that ran over a partial download, or over
a micrograph carrying an S3 error page in place of image data, still reports success —
the damage shows up as micrographs missing from the output, as fits pinned at the search
bound, or as a defocus distribution that has collapsed. This reads the job's exposure
dataset and reports the four things that separate those cases from a healthy run:

  count      output exposures vs the expected micrograph count
  fit        micrographs whose CTF fit resolution is missing, non-finite, or beyond a cutoff
  defocus    the df1/df2 distribution, plus fits sitting at a suspicious extreme
  astig      |df1 - df2|, which blows up when a fit has not converged

The verdict is advisory for `fit`/`astig` (some datasets genuinely fit poorly) and hard
for `count` (a missing micrograph is always wrong). Read the numbers, do not just take
the exit code.

A pixel size understated by a factor s makes every fitted defocus land s^2 low, so
EMPIAR-10345 — which this repo deliberately runs at CryoPPP's 0.673 A instead of the
physical 1.345 A — reports a defocus about 4x smaller than its neighbours. That is
expected here and is not a failure. See configs/datasets/empiar_10345.yaml.

  python verify_patch_ctf.py --job J2 --expected 300
  python verify_patch_ctf.py --manifest \\
      "$RAPICK_WORK/empiar_10081/annot/_shared/manifest.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rapick.recon import config                      # noqa: E402
from rapick.recon.api import CryoSPARCApi            # noqa: E402

# Beyond this the fit carries no usable high-frequency information. Advisory only:
# a thick-ice dataset can legitimately put a tail of micrographs here.
DEFAULT_FIT_CUTOFF_A = 10.0

# Patch CTF's own search bounds (v4.7 defaults). A fit resting exactly on a bound did
# not converge inside the search range.
SEARCH_MIN_DF_A, SEARCH_MAX_DF_A = 1000.0, 50000.0

# How far outside the p5..p95 defocus band counts as "this fit is not describing the
# micrograph". Wide enough that a genuinely spread defocus series stays inside it.
OUTLIER_FACTOR = 3.0


def _percentiles(values, points=(0, 5, 25, 50, 75, 95, 100)):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return {}
    return {p: ordered[min(n - 1, int(round(p / 100.0 * (n - 1))))] for p in points}


def _fmt(d, decimals=0):
    """Defocus and astigmatism run to five digits, fit resolution to one — so the
    caller picks the precision rather than every column rounding to whole angstroms."""
    return "  ".join(f"p{p}={v:,.{decimals}f}" for p, v in d.items())


def report(dset, expected: int | None, fit_cutoff_A: float) -> list[str]:
    """Return the list of hard failures; print everything either way."""
    import numpy as np

    failures = []
    n = len(dset)
    print(f"  exposures out : {n:,}" + (f"  (expected {expected:,})" if expected else ""))
    if expected is not None and n != expected:
        failures.append(f"exposure count {n} != expected {expected}")

    fields = set(dset.fields())

    def col(name):
        return np.asarray(dset[name], dtype=float) if name in fields else None

    df1, df2 = col("ctf/df1_A"), col("ctf/df2_A")
    fit = col("ctf/ctf_fit_to_A")

    if df1 is None or df2 is None:
        failures.append("no ctf/df1_A or ctf/df2_A column — did Patch CTF actually run?")
        return failures

    finite = np.isfinite(df1) & np.isfinite(df2) & (df1 > 0) & (df2 > 0)
    n_bad = int((~finite).sum())
    print(f"  defocus valid : {int(finite.sum()):,}   invalid/non-finite: {n_bad}")
    if n_bad:
        failures.append(f"{n_bad} micrographs have no usable defocus")

    if finite.any():
        mean_df = (df1[finite] + df2[finite]) / 2.0
        print(f"  defocus (A)   : {_fmt(_percentiles(mean_df))}")
        print(f"  defocus (um)  : median {np.median(mean_df) / 1e4:.3f}")

        at_bound = int(((mean_df <= SEARCH_MIN_DF_A * 1.001) |
                        (mean_df >= SEARCH_MAX_DF_A * 0.999)).sum())
        print(f"  at search bound: {at_bound}"
              + ("   <-- these did not converge" if at_bound else ""))

        # A fit can fail without producing a negative or non-finite defocus: it just lands
        # somewhere absurd. Patch CTF is not bit-deterministic, so the same unfittable
        # micrograph shows up as a negative df2 in one run and as an order-of-magnitude
        # outlier in the next. Judge against the population rather than against zero.
        pct = _percentiles(mean_df, (5, 95))
        lo, hi = pct[5] / OUTLIER_FACTOR, pct[95] * OUTLIER_FACTOR
        outliers = int(((mean_df < lo) | (mean_df > hi)).sum())
        print(f"  defocus outliers: {outliers} outside [{lo:,.0f}, {hi:,.0f}] A "
              f"(p5/p95 /x{OUTLIER_FACTOR:g})   [advisory]")

        astig = np.abs(df1[finite] - df2[finite])
        print(f"  astigmatism(A): {_fmt(_percentiles(astig))}")

        # A collapsed distribution (every micrograph fitted to the same value) means the
        # fit was not driven by the images.
        if np.ptp(mean_df) < 1.0:
            failures.append("defocus is constant across all micrographs — fit is not "
                            "responding to the data")

    if fit is None:
        print("  fit resolution: column absent")
    else:
        good = np.isfinite(fit) & (fit > 0)
        n_nofit = int((~good).sum())
        print(f"  fit res (A)   : {_fmt(_percentiles(fit[good]), 2)}   no-fit: {n_nofit}")
        if n_nofit:
            failures.append(f"{n_nofit} micrographs produced no CTF fit")
        n_poor = int((fit[good] > fit_cutoff_A).sum())
        print(f"  fit worse than {fit_cutoff_A:g} A: {n_poor} "
              f"({100.0 * n_poor / max(1, int(good.sum())):.1f}%)   [advisory]")

    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--project", help="project UID (else --manifest, "
                                      "else CRYOSPARC_PROJECT from .env)")
    ap.add_argument("--job", help="patch_ctf job UID (else read from --manifest)")
    ap.add_argument("--manifest", help="a _shared manifest.json to read project/job from")
    ap.add_argument("--expected", type=int, help="expected micrograph count")
    ap.add_argument("--fit-cutoff-A", type=float, default=DEFAULT_FIT_CUTOFF_A)
    args = ap.parse_args(argv)

    env = config.load_env(args.env)

    project, job = args.project, args.job
    if args.manifest:
        rec = json.loads(Path(args.manifest).read_text())
        project = project or rec.get("project_uid")
        job = job or (rec.get("shared", {}).get("patch_ctf") or {}).get("uid")
    project = project or env.get("CRYOSPARC_PROJECT")
    if not project or not job:
        sys.exit("need a project (--project or CRYOSPARC_PROJECT in .env) and a "
                 "--job, or a --manifest that records both")

    api = CryoSPARCApi(config.ConnectionConfig.from_env(env))
    api.connect()
    api.use_project(project)

    doc = api.find_job(project, job).doc
    status, job_type = doc.get("status"), doc.get("job_type")
    print(f"{project}/{job}  {job_type}  status={status}")
    failures = []
    if job_type != "patch_ctf_estimation_multi":
        failures.append(f"{job} is a {job_type!r}, not patch_ctf_estimation_multi")
    if status != "completed":
        failures.append(f"{job} is {status!r}, not completed")

    if not failures:
        dset = api.cs.find_job(project, job).load_output("exposures")
        failures += report(dset, args.expected, args.fit_cutoff_A)

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
