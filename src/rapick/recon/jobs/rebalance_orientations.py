"""Rebalance Orientations (net-new; subsets an existing refine's particles).

  job_type : rebalance_3D   (CONFIRMED — cs.print_job_types(): "Rebalance Orientations")
  inputs   : particles <- refine.particles   (carries alignments3D poses + blob)
  params   : rebalance3D_N_bins      Fibonacci orientation bins (default 192)
             rebalance3D_reject_pct  overpopulated-bin percentile T (default 80)
             rebalance3D_reject_by    intra-bin exclusion order (default "random")
  outputs  : particles -> "particles_rebalanced" group (kept subset, alignments3D + passthrough)
             (also emits "particles_excluded", not wired downstream)
  compute  : CPU only (rebalance_3D/build.py: set_resources_needed(1, 0, 24000, False))

Bins particles by viewing direction over the unit sphere and trims over-populated
bins down to the T-th percentile, flattening the orientation distribution. The kept
subset feeds a fresh Homogeneous Refinement so the re-refined map is built from a
more isotropic set of views. See guide.cryosparc.com Rebalance Orientations.

NOT USED BY ANY CONDITION OF THE PAPER. It is also not exported from jobs/__init__.py,
and configs/cryosparc_v47.yaml declares no `rebalance_orient` step -- so calling `run`
below raises KeyError until such an entry is added, using the job type and port names
documented above.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "rebalance_orient"


def run(api, profile, workspace_uid, params, inputs, gpu=None, wait=True) -> JobResult:
    refine = inputs["particles"]                     # source refine JobResult (particles port)
    slots = profile.inputs(STEP)
    connections = {slots["particles"]: (refine.job_uid, refine.outputs["particles"])}

    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
        wait=wait,
    )
