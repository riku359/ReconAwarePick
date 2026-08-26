"""Ab-Initio Reconstruction (forked per seed; the main source of trial variance).

  job_type : homo_abinit
  inputs   : particles <- class2d.particles
  params   : abinit_K=1, random_seed=<seed>, abinit_seed_init=<seed>
  outputs  : particles -> "particles_all_classes"
             volume    -> "volume_class_0"
  compute  : GPU

Confirmed against a live CryoSPARC v4.7.1 server.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "abinit"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    parts = inputs["particles"]                      # JobResult from class2d
    slot = profile.inputs(STEP)["particles"]
    connections = {slot: (parts.job_uid, parts.outputs["particles"])}
    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
    )
