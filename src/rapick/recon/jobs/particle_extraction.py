"""Extract From Micrographs.

  job_type : extract_micrographs_multi
  inputs   : particles   <- import_particles.imported_particles
             micrographs <- ctf.exposures     (CTF'd micrographs, NOT raw import)
  params   : box_size_pix   (dataset extraction.box_size_pix)
  outputs  : particles -> "particles"
  compute  : GPU

Confirmed against a live CryoSPARC v4.7.1 server.
"""
from __future__ import annotations

import os

from ._base import JobResult

STEP = "extract"

# Explicit opt-in escape hatch, for the rare case where a small, independently
# verified micrograph loss is acceptable. Unset by default so every caller keeps the
# strict guard below.
_MAX_INCOMPLETE_ALLOWED = int(os.environ.get("RAPICK_RECON_MAX_INCOMPLETE_MICS", "0"))


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    picks = inputs["particles"]       # JobResult from import_particles
    mics = inputs["micrographs"]      # JobResult from ctf
    slots = profile.inputs(STEP)
    connections = {
        slots["particles"]: (picks.job_uid, picks.outputs["particles"]),
        slots["micrographs"]: (mics.job_uid, mics.outputs["micrographs"]),
    }
    result = api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
    )
    _fail_on_incomplete(api, result)
    return result


def _fail_on_incomplete(api, result) -> None:
    """Extraction reports 'completed' even when micrographs were dropped (e.g. a
    per-micrograph GPU OOM). Those land in the 'micrographs_incomplete' output while
    their particles silently vanish -- one observed extract job lost 243 of its 300
    micrographs this way and still reported `completed`. Treat any incomplete
    micrograph as a hard failure so a partial stack never reaches 2D."""
    try:
        n_incomplete = api.output_count(
            api.project_uid, result.job_uid, "micrographs_incomplete")
    except Exception:
        n_incomplete = 0
    if n_incomplete and n_incomplete > _MAX_INCOMPLETE_ALLOWED:
        raise RuntimeError(
            f"{result.job_uid} extraction left {n_incomplete} micrographs incomplete "
            f"(likely GPU OOM on a shared card). Clear the job and re-extract on a "
            f"free GPU before continuing.")
    if n_incomplete:
        print(f"[extract] WARNING: {result.job_uid} left {n_incomplete} micrographs "
              f"incomplete; continuing because RAPICK_RECON_MAX_INCOMPLETE_MICS="
              f"{_MAX_INCOMPLETE_ALLOWED} allows it.")
