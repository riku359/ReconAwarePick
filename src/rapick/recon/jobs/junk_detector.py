"""Micrograph Junk Detector (BETA) — reject picks that sit on contamination.

  job_type : junk_detector_v1   (CONFIRMED — jobregister.py:86, build_junk_detector.py.
                                 The same file also defines the older junk_detector_beta1 /
                                 junk_particle_rejector_beta1 builders; do not use those —
                                 they name their outputs particles_good/particles_bad.)
  inputs   : exposures  <- ctf.exposures (before extract) or extract.micrographs (after).
                           Slots: micrograph_blob (required) + annotation_blob (optional).
                           Micrographs that ALREADY carry an annotation are not re-analysed:
                           their stored labels are reused to filter the particles.
             particles  <- import_particles.imported_particles (before extract) or
                           extract.particles (after). Only the `location` slot is required,
                           so unextracted picks are a valid input. The group is count_min=0;
                           with nothing connected the job only labels micrographs and the
                           two particle outputs are not generated at all.
  params   : min_dist (Å, default 150) — reject a pick whose centre lies within this distance
             of any junk pixel. Advanced per-class overrides: enable_cat{1,2,3} (bool) and
             min_dist_cat{1,2,3} (Å) for carbon_gold / intrinsic_ice / extrinsic_ice.
             v5.0's `min_label_area` does NOT exist on v4.7 — do not set it.
  outputs  : particles          -> "particles_accepted"  (exposed under the logical name
                                                          `particles` so the result drops
                                                          into any downstream particle slot
                                                          exactly like extract's does)
             particles_rejected -> "particles_rejected"
             micrographs        -> "exposures"           (labelled micrographs, carrying
                                                          annotation_blob + junk_stats)
  compute  : GPU — set_resources_needed(CPU 4, GPU 1, RAM 16 GB, SSD False). No SSD cache,
             so this stage is unaffected by the shared-cache contention that bites class_2D.

A pretrained network labels every micrograph pixel as one of no_label / carbon_gold /
intrinsic_ice / extrinsic_ice (4 labels, 512×512 annotation, psize ≈ 8.2 Å on 10532), then
drops each pick lying within `min_dist` of a labelled pixel.

Reference run (GUI, CryoSPARC v4.7.1) — EMPIAR-10532 at full-deposition scale, all params
default, wired AFTER extract (exposures + particles both taken from the extract job): 1,548
micrographs and 234,295 extracted particles in, 101,399 accepted / 132,896 rejected (56.7 %
rejected), 18.5 min on one 24 GB card with `SSD: false`.

This module was verified against that run: driving it from Python reproduced all three
output counts (1,548 / 101,399 / 132,896). The output names above are therefore confirmed
against a live job, not just read off the builder source — which matters because a freshly
built job shows only `exposures`, the particle outputs appearing once picks are connected
(see the count_min=0 note above).

Why this stage exists: it is the CryoSPARC-native counterpart of the MicrographCleaner filter
behind the `cryotransformer_clean` source. Both drop picks that land on contamination, so
putting the same picker through both lets the study ask whether contamination-aware picking
improves the 3D map independently of which detector produced the mask.
"""
from __future__ import annotations

import sys

from ._base import JobResult

STEP = "junk_detector"


def run(api, profile, workspace_uid, params, inputs, gpu=None) -> JobResult:
    exposures = inputs["exposures"]   # ctf JobResult (pre-extract) or extract (post-extract)
    picks = inputs["particles"]       # import_particles (pre-extract) or extract JobResult
    slots = profile.inputs(STEP)
    connections = {
        slots["exposures"]: (exposures.job_uid, exposures.outputs["micrographs"]),
        slots["particles"]: (picks.job_uid, picks.outputs["particles"]),
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
    _report_split(api, result)
    return result


def _report_split(api, result) -> None:
    """Log the accept/reject split and refuse to hand an empty stack downstream.

    The rejection rate is this stage's headline number, so it belongs in the run log rather
    than only in the GUI. Rejecting *everything* means the detector labelled the whole
    micrograph set as junk (wrong optics, or a micrograph type the pretrained model has never
    seen); left alone that surfaces much later as a confusing failure inside extract or
    class_2D, so fail here where the cause is still obvious."""
    n_accepted = api.output_count(api.project_uid, result.job_uid, result.outputs["particles"])
    n_rejected = api.output_count(
        api.project_uid, result.job_uid, result.outputs["particles_rejected"])

    total = n_accepted + n_rejected
    rejected_frac = n_rejected / total if total else 0.0
    print(f"[rapick-recon] {result.job_uid} junk detector: {n_accepted} accepted, "
          f"{n_rejected} rejected ({rejected_frac:.1%} of {total})", file=sys.stderr)

    if not n_accepted:
        raise RuntimeError(
            f"{result.job_uid} rejected all {total} picks, leaving nothing to reconstruct. "
            f"Check that the connected micrographs are the ones this stack was picked from, "
            f"then lower min_dist before re-running.")
