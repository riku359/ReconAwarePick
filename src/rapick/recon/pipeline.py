"""Research workflow control: source branching, best-of-3, resume.

Shared import+ctf are computed once per (dataset, setting) and reused across every
source of that dataset. Per source: import_particles -> extract -> class_2D (one
seed), then abinit+refine forked over `seeds`, best-of-3 by GSFSC 0.143, then
local_resolution on the winner.

Three seeds, not one: a single-seed resolution is not trustworthy, because the
seed-to-seed spread can exceed the effect being measured. Measured spreads in this
project run from 0.011 A to 2.073 A depending on the dataset, so how many seeds are
needed is dataset-dependent and best-of-3 is the floor. `--seeds` must be passed
explicitly on every `run`/`collect` call; the condition YAML's reconstruction.seeds
is documentation, not enforced config.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import coords
from . import manifest as mf
from .config import ResolvedConfig
from .jobs import (
    abinit_reconstruction,
    classification_2d,
    ctf_estimation,
    homogeneous_refinement,
    import_micrographs,
    import_particles,
    local_resolution,
    particle_extraction,
)
from .jobs._base import JobResult
from .manifest import Manifest, TrialRecord


def _worker_pin(gpu):
    """Route a CPU-only job to the same worker as this run's GPU jobs, without reserving
    a GPU. Both worker targets share the 'default' lane; an unpinned CPU job can land on
    a stale/unreachable node and fail at launch (ssh exit 255), so carry just the host."""
    return {"hostname": gpu["hostname"]} if gpu and gpu.get("hostname") else None


def _extract_gpus(gpu):
    """GPU placement for the extract step. Extract, unlike the reconstruction stages, is
    I/O-bound not compute-bound (CryoSPARC: extraction is "bottlenecked by speed of reading
    and writing micrograph and particle stack files"). When the micrographs sit on a shared
    or networked disk the read IS the bottleneck, so with --extract-gpus N extract runs
    across N physically-free cards: CryoSPARC splits the micrographs between them, giving N
    parallel readers and a larger share of the storage -- without touching other jobs.

    Returns (gpu_for_extract, num_gpus). num_gpus is how many cards were actually free
    (>= 1), so the reserved-card count always matches the compute_num_gpus param. Falls
    back to the run's single card when the run didn't opt in or nothing extra is free."""
    want = gpu.get("extract_num_gpus") if gpu else None
    if not want or want <= 1 or not gpu.get("hostname"):
        return gpu, 1
    from .gpu_select import pick_free_gpus
    cards = pick_free_gpus(want, cryosparcm=gpu.get("cryosparcm"))
    if len(cards) <= 1:                      # nothing extra free -> no gain, keep one card
        return gpu, 1
    return {**gpu, "gpus": cards, "auto": False}, len(cards)


def _step(api, jobs: dict, name: str, build, fingerprint: str = None,
          checkpoint=None) -> JobResult:
    """Reuse the recorded job iff completed and its inputs are unchanged, else build it
    and record the result with the fingerprint of the inputs it ran on.

    `checkpoint` persists the manifest the moment a job is recorded. A run here lasts
    hours and the host it runs on can go down mid-step, which kills the process outright
    — no exception, no cleanup. Saving only at the end loses the record of every job
    above the interruption, and the next run then rebuilds them and orphans the
    originals. Checkpointing costs one small write per job and bounds that loss to the
    single job that was in flight."""
    existing = mf.reuse_or_none(api, jobs, name, fingerprint)
    if existing is not None:
        if fingerprint is not None and jobs.get(name, {}).get("input_fingerprint") is None:
            print(f"[rapick-recon] reusing {name} ({existing.job_uid}) recorded before input "
                  f"fingerprints existed — cannot verify its inputs are unchanged", file=sys.stderr)
        return existing
    result = build()
    mf.record(jobs, name, result, input_fingerprint=fingerprint)
    if checkpoint:
        checkpoint()
    return result


def ensure_shared(api, cfg: ResolvedConfig, setting: str, manifest: Manifest, gpu=None,
                  checkpoint=None) -> dict:
    """Create-or-reuse Import Micrographs + Patch CTF for this (dataset, setting).
    Returns {"micrographs": JobResult, "ctf": JobResult}. Reused across sources."""
    profile, ws = cfg.profile, manifest.workspace_uid

    mic = _step(api, manifest.shared, "import_micrographs", lambda: import_micrographs.run(
        api, profile, ws, params=cfg.dataset.import_params(setting), gpu=_worker_pin(gpu)),
        fingerprint=coords.micrograph_set_fingerprint(cfg.dataset.micrograph_glob(setting)),
        checkpoint=checkpoint)
    ctf = _step(api, manifest.shared, "patch_ctf", lambda: ctf_estimation.run(
        api, profile, ws, params={}, inputs={"micrographs": mic}, gpu=gpu),
        checkpoint=checkpoint)

    return {"micrographs": mic, "ctf": ctf}


def run_source(api, cfg, setting, source, seeds, shared_manifest, source_manifest, gpu=None,
               checkpoint=None) -> dict:
    """Run one source end to end: import_particles -> extract -> class_2D ->
    (abinit -> refine) x seeds -> best-of-N -> local_resolution."""
    profile, ws = cfg.profile, source_manifest.workspace_uid

    shared = ensure_shared(api, cfg, setting, shared_manifest, gpu=gpu, checkpoint=checkpoint)
    source_manifest.shared = shared_manifest.shared          # provenance copy

    src = cfg.dataset.source(setting, source)

    # Make the star CryoSPARC-ready (flip Y using the micrograph height) before import.
    ny = coords.dataset_micrograph_height(cfg.dataset.micrograph_glob(setting))
    out_dir = Path(cfg.work_root) / cfg.dataset.name / setting / source
    import_star = coords.normalize_star(src.star, out_dir / "normalized.star", ny, src.y_flip)
    source_manifest.input_star = {
        "path": src.star, "normalized": import_star, "y_flip": src.y_flip,
        "ny": ny, "sha256": mf.star_sha256(src.star)}

    picks = _step(api, source_manifest.jobs, "import_particles", lambda: import_particles.run(
        api, profile, ws,
        params={"particle_meta_path": import_star, **src.import_params},
        inputs={"micrographs": shared["micrographs"]}, gpu=_worker_pin(gpu)),
        fingerprint=source_manifest.input_star["sha256"], checkpoint=checkpoint)

    # Extract is read-bound; optionally fan it out over several free GPUs so micrographs
    # are read in parallel (see _extract_gpus). compute_num_gpus must equal the number of
    # reserved cards, so both are derived from the same pick.
    extract_gpu, n_extract_gpus = _extract_gpus(gpu)
    extract_params = {"box_size_pix": cfg.dataset.box_size_pix}
    if n_extract_gpus > 1:
        extract_params["compute_num_gpus"] = n_extract_gpus
    extracted = _step(api, source_manifest.jobs, "extract", lambda: particle_extraction.run(
        api, profile, ws, params=extract_params,
        inputs={"particles": picks, "micrographs": shared["ctf"]}, gpu=extract_gpu),
        checkpoint=checkpoint)

    class_params = dict(cfg.condition.class2d_params)
    seed_key = profile.seed_param("class2d")
    if seed_key:
        class_params[seed_key] = seeds[0]                    # 2D is shared: one seed
    classes = _step(api, source_manifest.jobs, "class2d", lambda: classification_2d.run(
        api, profile, ws, params=class_params, inputs={"particles": extracted}, gpu=gpu),
        checkpoint=checkpoint)

    # Fork abinit+refine over all seeds, pick the best by GSFSC 0.143.
    trials = {s: reconstruct_trial(api, cfg, classes, s, source_manifest, gpu=gpu,
                                   checkpoint=checkpoint) for s in seeds}
    best_seed = choose_best_trial(api, cfg, source_manifest, trials)
    best_refine = trials[best_seed]["refine"]

    # Local resolution on the best trial (its volume group carries the half maps).
    # Non-fatal: local-res is an optional extra output; if it fails (e.g. GPU OOM)
    # the reconstruction result still stands, so log and continue.
    out = {**shared, "import_particles": picks, "extract": extracted, "class2d": classes,
           "best_seed": best_seed, "refine": best_refine}
    if cfg.condition.local_res_enabled:
        try:
            local_res = _step(api, source_manifest.jobs, "local_res", lambda: local_resolution.run(
                api, profile, ws, params={}, inputs={"volume": best_refine}, gpu=gpu),
                checkpoint=checkpoint)
            source_manifest.local_res = local_res.job_uid
            out["local_res"] = local_res
        except Exception as exc:
            print(f"[rapick-recon] local_res failed for {source_manifest.source} "
                  f"(seed {best_seed}): {exc}; continuing without it", file=sys.stderr)

    source_manifest.status = "done"
    return out


# --- reconstruction ----------------------------------------------------
def _record_trial(manifest: Manifest, seed: int, abinit: JobResult, refine: JobResult) -> None:
    for t in manifest.trials:
        if t.seed == seed:
            t.abinit, t.refine = abinit.job_uid, refine.job_uid
            return
    manifest.trials.append(TrialRecord(seed=seed, abinit=abinit.job_uid, refine=refine.job_uid))


def reconstruct_trial(api, cfg, classes: JobResult, seed: int, manifest: Manifest, gpu=None,
                      checkpoint=None) -> dict:
    """One abinit + refine trial for a given seed. Resolution (res_0143) is filled
    by the collect/best-of-3 step (M4/M5)."""
    profile, ws = cfg.profile, manifest.workspace_uid

    abinit_params = {"abinit_K": 1}
    for sp in profile.seed_params("abinit"):
        abinit_params[sp] = seed
    abinit = _step(api, manifest.jobs, f"abinit_seed{seed}", lambda: abinit_reconstruction.run(
        api, profile, ws, params=abinit_params, inputs={"particles": classes}, gpu=gpu),
        checkpoint=checkpoint)

    refine_params = {}
    refine_seed = profile.seed_param("refine")
    if refine_seed:
        refine_params[refine_seed] = seed
    refine = _step(api, manifest.jobs, f"refine_seed{seed}", lambda: homogeneous_refinement.run(
        api, profile, ws, params=refine_params,
        inputs={"particles": abinit, "volume": abinit}, gpu=gpu),
        checkpoint=checkpoint)

    _record_trial(manifest, seed, abinit, refine)
    return {"abinit": abinit, "refine": refine}


def _set_trial_res(manifest: Manifest, seed: int, res: float) -> None:
    for t in manifest.trials:
        if t.seed == seed:
            t.res_0143 = res
            return


def choose_best_trial(api, cfg, manifest: Manifest, trials: dict) -> int:
    """Read each trial's refine GSFSC 0.143, record it, and return the best seed
    (lowest resolution in Å). Falls back to the first seed if none could be read."""
    best_seed, best_res = None, float("inf")
    for seed, tr in trials.items():
        res = api.read_gsfsc(manifest.project_uid, tr["refine"].job_uid).get("res_0143")
        _set_trial_res(manifest, seed, res)
        if res is not None and res < best_res:
            best_res, best_seed = res, seed
    if best_seed is None:
        best_seed = next(iter(trials))
    manifest.best_seed = best_seed
    return best_seed
