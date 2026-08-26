"""Import Particles — coordinates only (built-in job, "Ignore raw data").

Imports the source's pre-unified GT-aligned star as particle *locations* (no
image blobs), connected to the imported micrographs so Extract can cut them.
This is the confirmed coordinate->extract path; no external job / loader needed.

  job_type : import_particles
  inputs   : micrographs <- import_micrographs.imported_micrographs
  params   : particle_meta_path=<source star>,
             ignore_pose=True, ignore_blob=True,        (fixed in profile)
             + source.import_params (dataset yaml)
  outputs  : particles -> "imported_particles"
  compute  : CPU

Name-matching (imports/run.py:231-254): the job strips a leading "<token>_" from
the SOURCE (imported micrograph) names iff remove_leading_uid=True, and slices
the STAR (query) names by query_cut_prefix. Every star micrograph MUST match an
imported one — an unmatched name FAILS the job (it does not skip). So the
imported micrograph set must cover every micrograph the star references.
  - CryoPPP GT star carries a 22-char CryoSPARC "<uid>_" prefix on the STAR side
    -> query_cut_prefix=22 (NOT remove_leading_uid, which strips the source).
  - Picker unified stars use raw basenames -> no cutting.

Confirmed against a live CryoSPARC v4.7.1 server; the param semantics are in the
CryoSPARC master source, imports/run.py:231-268.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "import_particles"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    mic = inputs["micrographs"]                       # JobResult from import_micrographs
    slot = profile.inputs(STEP)["micrographs"]
    connections = {slot: (mic.job_uid, mic.outputs["micrographs"])}
    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,                         # CPU job, but pin the worker (pipeline._worker_pin)
    )
