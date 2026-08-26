"""Patch CTF Estimation (shared across all sources for a dataset+setting).

  job_type : patch_ctf_estimation_multi
  inputs   : exposures <- import_micrographs.imported_micrographs
  params   : (defaults)
  outputs  : micrographs -> "exposures"   (CTF'd micrographs, feed Extract)
  compute  : GPU

Confirmed against a live CryoSPARC v4.7.1 server.
"""
from __future__ import annotations

import os

from ._base import JobResult

STEP = "patch_ctf"

# Patch CTF reports 'completed' even when it silently drops micrographs to
# 'exposures_incomplete' (e.g. a per-micrograph GPU OOM on a shared card). A
# `completed` entry in a CryoSPARC manifest is therefore NOT proof that the job was
# correct: this project hit a real instance where Patch CTF finished `completed`
# while dropping micrographs. Downstream steps (extract, class_2D, ...) only ever
# see the shrunken 'exposures' output, with no signal that anything was dropped, so
# treat any incomplete micrograph as a hard failure by default -- same pattern as
# particle_extraction.py's _fail_on_incomplete. Opt-in escape hatch, off by default.
_MAX_INCOMPLETE_ALLOWED = int(
    os.environ.get("RAPICK_RECON_MAX_INCOMPLETE_CTF_MICS", "0"))


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    mic = inputs["micrographs"]                       # JobResult from import_micrographs
    slot = profile.inputs(STEP)["exposures"]          # input slot name
    connections = {slot: (mic.job_uid, mic.outputs["micrographs"])}
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
    try:
        n_incomplete = api.output_count(
            api.project_uid, result.job_uid, "exposures_incomplete")
    except Exception:
        n_incomplete = 0
    if n_incomplete and n_incomplete > _MAX_INCOMPLETE_ALLOWED:
        raise RuntimeError(
            f"{result.job_uid} patch_ctf left {n_incomplete} micrographs incomplete "
            f"(likely GPU OOM on a shared card). Clear the job and re-run patch_ctf "
            f"on a free GPU before continuing.")
    if n_incomplete:
        print(f"[patch_ctf] WARNING: {result.job_uid} left {n_incomplete} micrographs "
              f"incomplete; continuing because RAPICK_RECON_MAX_INCOMPLETE_CTF_MICS="
              f"{_MAX_INCOMPLETE_ALLOWED} allows it.")
