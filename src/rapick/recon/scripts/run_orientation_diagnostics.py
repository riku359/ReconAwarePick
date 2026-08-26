#!/usr/bin/env python3
"""Run CryoSPARC Orientation Diagnostics on an existing homo_refine output.

This exercises rapick/recon/jobs/orientation_diagnosis.py against a refinement that
already ran (e.g. one recorded in a source manifest's `trials[].refine`), then confirms
the GUI plots CryoSPARC rendered for it. It creates the real CryoSPARC job, so cFAR /
SCF* / the viewing-direction sphere appear in the web interface.

The diagnostic is a CPU job (GPU:0, CPU:24), so it only pins the worker — no GPU is reserved.

Usage (job UIDs are your own; read them out of the arm's manifest.json):
  python run_orientation_diagnostics.py --refine J29
  python run_orientation_diagnostics.py --refine J119 \
      --workspace W5 --assets-dir ./orient_J119          # download the rendered plots

Reads CRYOSPARC_* from --env (default .env at the repository root); the project uid comes
from CRYOSPARC_PROJECT unless --project overrides it. --profile default is the committed
v4.7 profile.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Import the package from the src/ layout without needing an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rapick.recon import config                       # noqa: E402
from rapick.recon import manifest as mf               # noqa: E402
from rapick.recon.api import CryoSPARCApi             # noqa: E402
from rapick.recon.jobs import orientation_diagnosis   # noqa: E402
from rapick.recon.jobs._base import JobResult         # noqa: E402


def _refine_result(api, project_uid: str, refine_uid: str) -> tuple[JobResult, str]:
    """Wrap an existing homo_refine job as a JobResult and return (result, its workspace).

    The refine job carries output groups 'volume' (map_half_A/B + mask_fsc_auto) and
    'particles' (alignments3D) — exactly the ports orientation_diagnosis wires. We reuse
    the port names verbatim so the module connects the live outputs, not fabricated ones.
    """
    job = api.find_job(project_uid, refine_uid)
    status = job.doc.get("status")
    if status != "completed":
        sys.exit(f"{refine_uid} is {status!r}, not completed — pick a finished refinement")
    groups = {g.get("name") for g in job.doc.get("output_result_groups", []) or []}
    if "volume" not in groups:
        sys.exit(f"{refine_uid} has no 'volume' output group (is it a homo_refine?): {sorted(groups)}")
    workspaces = job.doc.get("workspace_uids") or []
    result = JobResult(name="refine", job_uid=refine_uid, job_type=job.doc.get("job_type", "homo_refine"),
                       status=status, outputs={"volume": "volume", "particles": "particles"})
    return result, (workspaces[0] if workspaces else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--profile", default="configs/cryosparc_v47.yaml")
    ap.add_argument("--project", help="project UID holding the refine "
                                      "(default: CRYOSPARC_PROJECT from .env)")
    ap.add_argument("--refine", required=True, help="completed homo_refine job UID, e.g. J29")
    ap.add_argument("--workspace", help="workspace for the diagnostic (default: the refine's own)")
    ap.add_argument("--no-particles", action="store_true",
                    help="cFAR only — do not wire particles (skips the SCF* sampling metric)")
    ap.add_argument("--worker", help="CryoSPARC worker lane (CPU job pin); "
                    "default: CRYOSPARC_WORKER from .env, else this host's name")
    ap.add_argument("--assets-dir", help="download the rendered GUI plots (PNG/PDF) into this dir")
    ap.add_argument("--record", help="manifest.json to record the job in (as jobs['orient']); "
                                     "an already-recorded completed job is reused, not rebuilt")
    args = ap.parse_args(argv)

    env = config.load_env(args.env)
    profile = config.Profile.load(args.profile, env)
    project_uid = args.project or env.get("CRYOSPARC_PROJECT")
    if not project_uid:
        sys.exit("no project UID: set CRYOSPARC_PROJECT in .env or pass --project")
    api = CryoSPARCApi(config.ConnectionConfig.from_env(env))
    api.connect()
    api.use_project(project_uid)

    refine, refine_ws = _refine_result(api, project_uid, args.refine)
    ws = args.workspace or refine_ws
    if not ws:
        sys.exit(f"could not resolve a workspace for {args.refine}; pass --workspace")

    worker = args.worker or env.get("CRYOSPARC_WORKER") or os.uname().nodename
    inputs = {"volume": refine}
    if not args.no_particles:
        inputs["particles"] = refine
    print(f"[orient-diag] project={project_uid} workspace={ws} refine={args.refine} "
          f"particles={'no' if args.no_particles else 'yes'} worker={worker}")

    # Reuse a recorded diagnostic rather than making a second one. Without this, every
    # rerun of a driver over an already-finished arm — which is what a host reboot forces
    # — leaves another duplicate diagnostic behind on the same refinement.
    m = mf.load(args.record) if args.record else None
    result = mf.reuse_or_none(api, m.jobs, "orient") if m else None
    if result is not None:
        print(f"[orient-diag] reusing recorded {result.job_uid}")
    else:
        result = orientation_diagnosis.run(
            api, profile, ws, params={}, inputs=inputs, gpu={"hostname": worker})
        print(f"[orient-diag] {result.job_uid} ({result.job_type}) -> {result.status}")
        if m is not None:
            mf.record(m.jobs, "orient", result)
            mf.save(m, args.record)
            print(f"[orient-diag] recorded in {args.record}")
        elif args.record:
            print(f"[orient-diag] no manifest at {args.record}; not recorded", file=sys.stderr)

    # GUI proof: CryoSPARC renders the diagnostic plots (viewing-direction sphere, cFSC curves,
    # SCF*) as job assets. If they exist and are viewable, the GUI works for this job.
    assets = api.list_job_assets(project_uid, result.job_uid)
    plots = [a for a in assets if str(a.get("contentType", "")).startswith("image")]
    print(f"[orient-diag] CryoSPARC rendered {len(assets)} assets ({len(plots)} images) for the GUI:")
    for a in assets:
        print(f"    - {a.get('filename')}  ({a.get('contentType')}, {a.get('length')} bytes)")

    if args.assets_dir:
        out = Path(args.assets_dir)
        out.mkdir(parents=True, exist_ok=True)
        for a in assets:
            api.download_asset(a["_id"], out / a["filename"])
        print(f"[orient-diag] downloaded {len(assets)} assets to {out}")

    print(f"\nOpen in the GUI: {env.get('CRYOSPARC_HOST', 'localhost')}:{env.get('CRYOSPARC_PORT', '')}"
          f"  ->  project {project_uid} / workspace {ws} / job {result.job_uid}")
    return 0 if result.status == "completed" and assets else 1


if __name__ == "__main__":
    sys.exit(main())
