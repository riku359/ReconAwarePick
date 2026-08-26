"""Import Micrographs (shared across all sources for a dataset+setting).

  job_type : import_micrographs
  inputs   : (none)
  params   : blob_paths, psize_A, accel_kv, cs_mm, total_dose_e_per_A2   (dataset optics)
  outputs  : micrographs -> "imported_micrographs"
  compute  : CPU

Confirmed against a live CryoSPARC v4.7.1 server.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "import_micrographs"


def run(api, profile, workspace_uid, params, inputs=None, gpu=None) -> JobResult:
    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections={},                 # no inputs
        expose=profile.outputs(STEP),
        gpu=gpu,                         # CPU job, but pin the worker (pipeline._worker_pin)
    )
