"""2D Classification (run once per source; not forked over seeds).

  job_type : class_2D        (CONFIRMED — jobregister.py:141; class_2D_new also exists)
  inputs   : particles <- extract.particles
  params   : class2D_K, class2D_num_full_iter, random_seed   (caller-provided)
  outputs  : particles -> "particles"    (every accepted particle; the conditions that
                                          select 2D classes hang a Select 2D Classes job
                                          off this output instead of using it directly)
  compute  : GPU

Confirmed against a live CryoSPARC v4.7.1 server.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "class2d"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    ext = inputs["particles"]                        # JobResult from extract
    slot = profile.inputs(STEP)["particles"]
    connections = {slot: (ext.job_uid, ext.outputs["particles"])}
    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
    )
