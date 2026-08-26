"""Orientation Diagnostics (net-new; terminal diagnostic on a refined volume).

  job_type : orientation_diagnostics   (CONFIRMED — jobregister "Orientation Diagnostics", v4.4+)
  inputs   : volume    <- refine.volume      (OUTPUT group; carries map_half_A/B + mask_fsc_auto,
                                               which the job auto-uses as the conical-FSC mask)
             particles <- refine.particles    (OPTIONAL; enables the Fourier-sampling metric SCF*.
                                               cFAR is computed from the half-maps alone, so a
                                               volume-only run still yields the anisotropy score.)
  params   : defaults ok. Symmetry should match the upstream refinement (C1 here — the homo_refine
             default), so SCF*'s symmetry expansion agrees with the map.
  outputs  : volume_3dfsc -> "volume_3DFSC"   (interpolated 3DFSC volume, result map_3DFSC;
                                               terminal — nothing downstream consumes it)
  compute  : CPU  (resources_needed GPU:0, CPU:24 — route with a worker pin, do NOT reserve a GPU)

Verified end-to-end on CryoSPARC v4.7.1: orientation_diagnostics on refine J29 (cryolo/10093)
completed in ~24 s and rendered the GUI plots (viewing-direction distribution, conical FSCs,
Fourier sampling), reporting cFAR=0.063 and SCF*=0.850.

Purpose: read the anisotropy of a refinement independently of its resolution (Å). cFAR < 0.5 or
SCF* < 0.81 flag preferred orientation; used to compare a picker's baseline stack against its
MicrographCleaner-cleaned stack (same particles, contamination removed) and see whether cleaning
moved the orientation bias — the reconstruction-aware picking question.

Contract confirmed by build-only introspection on CryoSPARC v4.7.1: input_slot_groups are
volume / volumes_all_classes / particles / mask.
"""
from __future__ import annotations

from ._base import JobResult

STEP = "orient_diag"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    refine = inputs["volume"]                          # refine JobResult (its 'volume' group)
    slots = profile.inputs(STEP)
    connections = {slots["volume"]: (refine.job_uid, refine.outputs["volume"])}

    # SCF* (sampling-based anisotropy) is only computed when particles are connected. Wire them
    # whenever the caller supplies them so the diagnosis reports both signal (cFAR) and sampling
    # (SCF*) views; a volume-only call (no particles) still produces the cFAR score.
    parts = inputs.get("particles")
    if parts is not None and "particles" in slots:
        connections[slots["particles"]] = (parts.job_uid, parts.outputs["particles"])

    return api.create_and_wait(
        workspace_uid=workspace_uid,
        name=STEP,
        job_type=profile.type(STEP),
        params={**profile.fixed_params(STEP), **params},
        connections=connections,
        expose=profile.outputs(STEP),
        gpu=gpu,
    )
