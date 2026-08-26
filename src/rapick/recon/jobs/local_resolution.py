"""Local Resolution Estimation (net-new; runs on the best-of-3 refine only).

  job_type : local_resolution    (CONFIRMED — jobregister.py:222)
  inputs   : volume <- refine.volume   (OUTPUT group; carries map_half_A/B + masks)
             mask   <- (optional) static mask; defaults to the refinement mask
  params   : locres_use_mask, locres_step_size, locres_box_width, ...  (defaults ok)
  outputs  : volume -> "volume" group; the local-res map is result "map_locres"
  compute  : GPU

Contract confirmed from the master job spec (local_resolution/build.py:19-60,
refine/build.py:119-126).
"""
from __future__ import annotations

from ._base import JobResult


STEP = "local_res"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    refine = inputs["volume"]                         # refine JobResult
    slot = profile.inputs(STEP)["volume"]             # connect refine's volume GROUP
    connections = {slot: (refine.job_uid, refine.outputs["volume"])}
    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
    )
