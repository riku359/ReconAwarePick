"""collect: derive light summaries and references from CryoSPARC job outputs.

The CryoSPARC job directory is the source of truth. This module never copies
particle stacks / half maps / large volumes / logs. It writes small derived
files under ${RAPICK_WORK}/empiar_<id>/<setting>/<source>/ and records
project-relative paths as references. `collect` is re-runnable and never
re-runs CryoSPARC jobs.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .manifest import Manifest


def particle_counts(api, manifest: Manifest) -> dict:
    """How many particles survive each stage, including what ab-initio actually used.

    Asked for a single class, `homo_abinit` reconstructs a randomly chosen subset large
    enough to converge and leaves the rest in its `particles_unused` output, so
    `homo_refine` never sees the whole stack. That cap is a per-ID constant (measured:
    219,900 on 10081, 176,700 on 10532) and does not move with the input size, so the
    used/unused pair is the only way to tell a clamped arm from an unclamped one -- and
    two arms are comparable on particle count only when neither is clamped.
    """
    counts = {}

    def record(key: str, uid: str, output: str) -> None:
        try:
            counts[key] = api.output_count(manifest.project_uid, uid, output)
        except Exception as exc:
            counts[key] = f"error: {exc}"

    # Each job's particle output is named in its own manifest record, so this follows the
    # DAG rather than restating it: `select_2D` emits `particles_selected`, `homo_abinit`
    # emits `particles_all_classes`, and the rest emit `particles`.
    steps = ["import_particles", "extract", "class2d", "select2d"]
    steps += [f"abinit_seed{trial.seed}" for trial in manifest.trials]
    for step in steps:
        rec = manifest.jobs.get(step)
        output = (rec or {}).get("outputs", {}).get("particles")
        if not output:
            continue
        record(f"{step}_used" if step.startswith("abinit_") else step, rec["uid"], output)
        if step.startswith("abinit_"):
            # Absent from the manifest's output map: the profile records only what the
            # pipeline connects downstream, and nothing consumes the discarded particles.
            record(f"{step}_unused", rec["uid"], "particles_unused")

    return counts


def map_references(api, manifest: Manifest) -> dict:
    """Project-relative job dirs of the best refine and local-res volumes (no copy)."""
    refs = {}
    best = manifest.best_seed
    if best is not None:
        rec = manifest.jobs.get(f"refine_seed{best}")
        if rec:
            refs["refine"] = {"uid": rec["uid"], "dir": api.job_dir(manifest.project_uid, rec["uid"])}
    if manifest.local_res:
        refs["local_res"] = {"uid": manifest.local_res,
                             "dir": api.job_dir(manifest.project_uid, manifest.local_res)}
    return refs


# Per-micrograph CTF / quality covariates from the shared Patch-CTF `exposures`
# output. Each entry is (csv column, how to derive it from the exposure dataset).
# Defocus = mean(df1, df2) in um (readable scale); astigmatism = |df1 - df2| in A;
# ctf_fit = ctf/ctf_fit_to_A = the resolution (A) the CTF fit is reliable to
# (lower is a better micrograph); ice = ctf_stats relative ice thickness. These
# are IDENTICAL across every condition of a dataset (one shared import+CTF per
# dataset+scale), so they characterise DATA difficulty rather than a picker.
CTF_COVARIATES = (
    ("defocus_um",        lambda ds: 0.5 * (ds["ctf/df1_A"] + ds["ctf/df2_A"]) / 1e4),
    ("astig_A",           lambda ds: abs(ds["ctf/df1_A"] - ds["ctf/df2_A"])),
    ("df_angle_deg",      lambda ds: _degrees(ds["ctf/df_angle_rad"])),
    ("ctf_fit_A",         lambda ds: ds["ctf/ctf_fit_to_A"]),
    ("ice_thickness_rel", lambda ds: ds["ctf_stats/ice_thickness_rel"]),
)


def _degrees(radians):
    import numpy as np
    return np.degrees(np.asarray(radians, float))


def _micrograph_names(ds) -> list:
    """Micrograph basenames for the CTF rows, or '' where the blob path is absent."""
    if "micrograph_blob/path" not in ds.fields():
        return [""] * len(ds)
    names = []
    for path in ds["micrograph_blob/path"]:
        text = path.decode() if isinstance(path, bytes) else str(path)
        names.append(Path(text).name)
    return names


def ctf_covariates(api, manifest: Manifest, out_dir: str | Path) -> dict | str:
    """Write the shared micrographs' CTF/quality covariates to derived/ctf.csv
    and return a small median summary. Reads the shared Patch-CTF `exposures` output
    recorded in manifest.shared; because that job is reused across conditions, every
    condition's ctf.csv is the same distribution (data difficulty, not picker). Degrades to a string on any failure so it never
    breaks collect; never re-runs a job (job dir stays the source of truth)."""
    import numpy as np

    rec = manifest.shared.get("patch_ctf")
    if not rec:
        return "no patch_ctf job in manifest.shared"
    try:
        ds = api.load_output(manifest.project_uid, rec["uid"], "exposures")
    except Exception as exc:
        return f"load exposures ({rec['uid']}): {exc}"

    columns = {name: np.asarray(derive(ds), float) for name, derive in CTF_COVARIATES}
    names = _micrograph_names(ds)

    derived = Path(out_dir) / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    csv_path = derived / "ctf.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["micrograph", *columns])
        for i in range(len(ds)):
            writer.writerow([names[i], *(f"{columns[c][i]:.6g}" for c in columns)])

    return {"job": rec["uid"], "csv": str(csv_path), "n_micrographs": int(len(ds)),
            "median": {c: float(np.median(v)) for c, v in columns.items()}}


# CryoSPARC renders these plots per iteration and stores them in GridFS as both
# PDF (vector, for figures) and PNG. Each spec is (plot name, manifest job key,
# filename stem). The best refine holds the FSC (A4) and viewing-direction (A5)
# plots; class_2D holds the 50-class average montage (A7/V8). The stem sits in
# `J<uid>_<stem>[_for]_iteration_<N>.<ext>` -- class_2D inserts a `_for` before
# `_iteration_` (e.g. J87_2d_classes_for_iteration_39.png) that refine omits.
PLOT_SPECS = (
    ("fsc",               "refine_best", "fsc"),                             # A4 GSFSC curve
    ("viewing_direction", "refine_best", "viewing_direction_distribution"),  # A5 elevation x azimuth
    ("class_averages",    "class2d",     "2d_classes"),                      # A7/V8 50-class montage
)


def _plot_job_uid(manifest: Manifest, job_key: str) -> str | None:
    """Resolve a PLOT_SPECS job key to a job UID. `refine_best` means the best-seed
    refinement; any other key indexes manifest.jobs directly. None if absent."""
    if job_key == "refine_best":
        best = manifest.best_seed
        job_key = f"refine_seed{best}" if best is not None else ""
    rec = manifest.jobs.get(job_key)
    return rec["uid"] if rec else None


def _final_iteration_assets(assets: list, stem: str) -> dict:
    """Pick the highest-iteration PDF and PNG for one plot family (e.g. 'fsc').
    Returns {ext: asset}; empty if the family is absent. The `_iteration_` anchor
    (with class_2D's optional `_for`) keeps 'fsc' from also matching
    'fsc_before_fsc_mask_auto_tightening'."""
    latest: dict = {}  # ext -> (iteration, asset)
    pattern = re.compile(rf"_{re.escape(stem)}(?:_for)?_iteration_(\d+)\.(pdf|png)$")
    for asset in assets:
        match = pattern.search(asset.get("filename", ""))
        if not match:
            continue
        iteration, ext = int(match.group(1)), match.group(2)
        if ext not in latest or iteration > latest[ext][0]:
            latest[ext] = (iteration, asset)
    return {ext: asset for ext, (_, asset) in latest.items()}


def cryosparc_plots(api, manifest: Manifest, out_dir: str | Path) -> dict:
    """Download CryoSPARC's own GUI-rendered final-iteration plots -- PDF + PNG --
    into <out_dir>/derived/cryosparc_plots/: FSC (A4) + viewing-direction (A5) from
    the best refine, and the 50-class 2D average montage (A7/V8) from class_2D.
    Fetched verbatim (no recomputation); the 5-source overlay/panel figures are
    built separately from data. Every failure degrades to a string
    so it never breaks collect; never re-runs a job."""
    plot_dir = Path(out_dir) / "derived" / "cryosparc_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    assets_cache: dict = {}  # job_uid -> asset list, or an error string if listing failed

    def assets_for(job_uid: str):
        if job_uid not in assets_cache:
            try:
                assets_cache[job_uid] = api.list_job_assets(manifest.project_uid, job_uid)
            except Exception as exc:
                assets_cache[job_uid] = f"list_assets({job_uid}): {exc}"
        return assets_cache[job_uid]

    plots, jobs = {}, {}
    for name, job_key, stem in PLOT_SPECS:
        job_uid = _plot_job_uid(manifest, job_key)
        jobs[name] = job_uid
        if not job_uid:
            plots[name] = f"no {job_key} job in manifest"
            continue
        assets = assets_for(job_uid)
        if isinstance(assets, str):          # list_assets failed for this job
            plots[name] = assets
            continue
        chosen = _final_iteration_assets(assets, stem)
        if not chosen:
            plots[name] = "not found"
            continue
        saved = {}
        for ext, asset in chosen.items():
            try:
                api.download_asset(asset["_id"], plot_dir / asset["filename"])
                saved[ext] = asset["filename"]
            except Exception as exc:
                saved[ext] = f"error: {exc}"
        plots[name] = saved
    return {"dir": str(plot_dir), "jobs": jobs, "plots": plots}


# Orientation Diagnostics assets that carry the Fourier-space / directional views.
# These are single (no `_iteration_N`), so they need a keyword match rather than the
# highest-iteration picker used for refine/class_2D plots.
OD_PLOT_KEYWORDS = (
    "summary_of",                          # cFSC summary (reports cFAR)
    "3dfsc_volume_central_slices",         # 3DFSC volume central slices (directional res)
    "fourier_sampling",                    # Fourier sampling — the missing-cone view
    "relative_signal_amount_vs_viewing",   # relative signal (tFSC) by viewing direction
    "avg_relative_signal_amount_within",   # relative signal in az/el regions (missing views)
    "viewing_direction_distribution",      # viewing-direction heatmap
    "posterior_precision_directional",     # CTF-modulated Fourier sampling
)


def _od_quadrant(cfar, scf_star) -> str | None:
    """The official cFAR/SCF* 2x2 diagnostic (guide.cryosparc.com Orientation
    Diagnostics): cFAR<0.5 and/or SCF*<0.81 flag preferred orientation."""
    if cfar is None or scf_star is None:
        return None
    lo_c, lo_s = cfar < 0.5, scf_star < 0.81
    if not lo_c and not lo_s:
        return "no orientation bias"
    if lo_c and not lo_s:
        return "anisotropy from junk/contaminants (sampling ok, signal anisotropic)"
    if not lo_c and lo_s:
        return "inconclusive / pathological alignment distribution"
    return "anisotropy from junk and/or preferred orientation"


def orientation_diagnostics(api, manifest: Manifest, out_dir: str | Path) -> dict:
    """Read cFAR/SCF* off the recorded Orientation Diagnostics job and download its
    Fourier-space plots. cFAR<0.5 or SCF*<0.81 => preferred orientation. Every
    failure degrades to a string so it never breaks collect; never re-runs a job."""
    # scripts/run_orientation_diagnostics.py --record writes this under "orient"; the
    # older "orient_diag" is still read so manifests written before the rename collect.
    rec = manifest.jobs.get("orient") or manifest.jobs.get("orient_diag")
    if not rec:
        return {"status": "no orientation diagnostics job recorded"}
    uid = rec["uid"]

    cfar = scf_star = None
    try:
        doc = api.find_job(manifest.project_uid, uid).doc
        for group in doc.get("output_result_groups", []):
            stats = group.get("latest_summary_stats") or {}
            if stats.get("cfar") is not None:
                cfar = float(stats["cfar"])
            if stats.get("scf_star") is not None:
                scf_star = float(stats["scf_star"])
    except Exception as exc:
        return {"job": uid, "error": f"read stats: {exc}"}

    plot_dir = Path(out_dir) / "derived" / "cryosparc_plots" / "orientation_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: dict = {}
    try:
        assets = api.list_job_assets(manifest.project_uid, uid)
    except Exception as exc:
        assets, saved["_list_error"] = [], str(exc)
    for asset in assets:
        filename = asset.get("filename", "")
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        if ext in ("png", "pdf") and any(k in filename for k in OD_PLOT_KEYWORDS):
            try:
                api.download_asset(asset["_id"], plot_dir / filename)
                saved[filename] = filename
            except Exception as exc:
                saved[filename] = f"error: {exc}"

    return {"job": uid, "cFAR": cfar, "SCF_star": scf_star,
            "quadrant": _od_quadrant(cfar, scf_star),
            "dir": str(plot_dir), "plots": saved}


def collect(api, cfg, manifest: Manifest, out_dir: str | Path) -> dict:
    """Derive metrics.json for one completed condition run: particle counts through
    every stage, the per-seed resolutions and which seed won, references to the maps in
    the CryoSPARC job directory, the CTF covariates of the micrographs, and the plots
    CryoSPARC rendered for its own jobs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": manifest.dataset,
        "setting": manifest.setting,
        "source": manifest.source,
        "status": manifest.status,
        "particle_counts": particle_counts(api, manifest),
        "best_seed": manifest.best_seed,
        "trials": [{"seed": t.seed, "refine": t.refine, "res_0143": t.res_0143}
                   for t in manifest.trials],
        "maps": map_references(api, manifest),
        "ctf_covariates": ctf_covariates(api, manifest, out),
        "cryosparc_plots": cryosparc_plots(api, manifest, out),
        "orientation_diagnostics": orientation_diagnostics(api, manifest, out),
        "input_star": manifest.input_star,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
