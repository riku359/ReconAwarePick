"""Homogeneous Refinement (forked per seed; GSFSC 0.143 selects best-of-3).

  job_type : homo_refine
  inputs   : particles <- abinit.particles_all_classes
             volume    <- abinit.volume_class_0
  params   : random_seed=<seed>
  outputs  : volume -> "volume"           (group carries map_half_A/B + mask_fsc_auto)
             mask   -> "mask_fsc_auto"    (used by local_resolution)
  compute  : GPU

Confirmed against a live CryoSPARC v4.7.1 server; the output ports are declared in the
CryoSPARC master source, refine/build.py:119-126.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "refine"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    parts = inputs["particles"]                      # abinit (particles_all_classes)
    vol = inputs["volume"]                           # abinit (volume_class_0)
    slots = profile.inputs(STEP)
    connections = {
        slots["particles"]: (parts.job_uid, parts.outputs["particles"]),
        slots["volume"]: (vol.job_uid, vol.outputs["volume"]),
    }
    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
    )
