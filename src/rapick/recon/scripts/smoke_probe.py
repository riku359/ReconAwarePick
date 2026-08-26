#!/usr/bin/env python3
"""Re-verify configs/cryosparc_v47.yaml against a live CryoSPARC server, without
running any jobs.

Run this on a new site, and after any CryoSPARC upgrade: the profile records the job
types and port names this pipeline wires together, and a release that renames one of
them would otherwise surface as a confusing failure hours into a run.

It creates each candidate job *build-only* (create_job, never queue), reads the
input/output/param names off its `job.doc`, prints them, and deletes it. Nothing
is queued, so this costs no GPU time.

What it answers:
  - Does `local_resolution` exist? What are `homo_refine`'s volume/mask output ports
    (and `homo_abinit`'s particles/volume ports)?
  - `class_2D` vs `class_2D_new`; `homo_refine` vs `homo_refine_new`.
  - `import_particles` param names (ignore_blob / ignore_pose / remove_leading_uid).

The probe writes jobs into whatever project you point it at, so create a THROWAWAY
project first and pass that uid -- never one holding results.

Usage:
  python smoke_probe.py --env .env --project <throwaway uid> [--workspace W5] [--keep]

Reads CRYOSPARC_HOST / CRYOSPARC_PORT / CRYOSPARC_LICENSE_ID / CRYOSPARC_EMAIL /
CRYOSPARC_PASSWORD from the --env file (falling back to the environment).
"""
from __future__ import annotations

import argparse
import os
import sys

# Candidate job types to probe. Alternatives are grouped so the report can say
# which of each pair the server actually registers.
PROBE_TYPES = [
    "import_micrographs",
    "patch_ctf_estimation_multi",
    "import_particles",
    "extract_micrographs_multi",
    "class_2D",
    "class_2D_new",
    "homo_abinit",
    "homo_refine",
    "homo_refine_new",
    "nonuniform_refine_new",
    "local_resolution",
]

# Params worth confirming by name, per job type.
PARAM_INTEREST = {
    "import_particles": ("ignore_blob", "ignore_pose", "remove_leading_uid", "particle_meta_path"),
    "class_2D": ("class2D_K", "class2D_num_full_iter", "random_seed"),
    "class_2D_new": ("class2D_K", "class2D_num_full_iter", "random_seed"),
    "homo_abinit": ("abinit_K", "random_seed", "abinit_seed_init"),
    "homo_refine": ("random_seed",),
    "homo_refine_new": ("random_seed",),
}


def read_env(path: str) -> dict:
    """Parse a KEY=VALUE .env file, layered under os.environ (env wins)."""
    values: dict[str, str] = {}
    if path and os.path.isfile(path):
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.startswith("CRYOSPARC_")})
    return values


def connect(env: dict):
    from cryosparc.tools import CryoSPARC

    missing = [k for k in ("CRYOSPARC_LICENSE_ID", "CRYOSPARC_EMAIL", "CRYOSPARC_PASSWORD") if not env.get(k)]
    if missing:
        sys.exit(f"missing in env: {', '.join(missing)}")
    cs = CryoSPARC(
        license=env["CRYOSPARC_LICENSE_ID"],
        email=env["CRYOSPARC_EMAIL"],
        password=env["CRYOSPARC_PASSWORD"],
        host=env.get("CRYOSPARC_HOST", "localhost"),
        base_port=int(env.get("CRYOSPARC_PORT", "39000")),
    )
    if not cs.test_connection():
        sys.exit("could not reach CryoSPARC (check host/port and that the master is up)")
    return cs


def _names(groups) -> list:
    """Best-effort extraction of slot/result names from a job.doc group list.
    The doc schema varies by release, so probe several common shapes."""
    out = []
    for g in groups or []:
        gname = g.get("name") if isinstance(g, dict) else str(g)
        subs = []
        if isinstance(g, dict):
            for key in ("slots", "contains", "results"):
                for s in g.get(key, []) or []:
                    subs.append(s.get("name") if isinstance(s, dict) else str(s))
        out.append(f"{gname}({', '.join(n for n in subs if n)})" if subs else str(gname))
    return out


def probe_one(project, workspace_uid: str, job_type: str, keep: bool) -> dict:
    """Build one job (no queue), read its ports/params, delete it. Returns a report."""
    try:
        job = project.create_job(workspace_uid, job_type)
    except Exception as exc:  # type not registered / not creatable
        return {"type": job_type, "exists": False, "error": str(exc).splitlines()[0][:200]}

    doc = getattr(job, "doc", {}) or {}
    # params_base is the job type's full param schema. params_spec holds only the values a
    # user overrode, so it is empty on a default job and reports every param as missing.
    params_base = doc.get("params_base", {}) or {}
    interest = PARAM_INTEREST.get(job_type, ())
    report = {
        "type": job_type,
        "exists": True,
        "uid": job.uid,
        "inputs": _names(doc.get("input_slot_groups")),
        "outputs": _names(doc.get("output_result_groups")),
        "params_present": [p for p in interest if p in params_base],
        "params_missing": [p for p in interest if p not in params_base],
    }
    if not keep:
        try:
            job.delete()
        except Exception:
            try:
                project.cs.cli.delete_job(project.uid, job.uid)  # fallback
            except Exception:
                report["cleanup"] = f"could not delete {job.uid}; remove manually"
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--project", required=True,
                    help="existing THROWAWAY project UID -- this creates and deletes "
                         "jobs in it, so never point it at a project holding results")
    ap.add_argument("--workspace", help="existing workspace UID; if omitted, a probe workspace is created")
    ap.add_argument("--keep", action="store_true", help="do not delete the built probe jobs")
    args = ap.parse_args(argv)

    env = read_env(args.env)
    cs = connect(env)
    print("connected:", env.get("CRYOSPARC_HOST", "localhost"),
          env.get("CRYOSPARC_PORT", "39000"))

    print("\n=== job types registered on this server ===")
    cs.print_job_types()

    project = cs.find_project(args.project)
    workspace_uid = args.workspace
    if not workspace_uid:
        ws = project.create_workspace(title="rapick_recon_probe")
        workspace_uid = ws.uid
        print(f"\ncreated probe workspace {workspace_uid}")

    print("\n=== per-job-type introspection (build-only, no queue) ===")
    reports = [probe_one(project, workspace_uid, jt, args.keep) for jt in PROBE_TYPES]
    for r in reports:
        if not r["exists"]:
            print(f"\n[{r['type']}]  MISSING  ({r.get('error','')})")
            continue
        print(f"\n[{r['type']}]  EXISTS  ({r['uid']})")
        print(f"  inputs : {r['inputs']}")
        print(f"  outputs: {r['outputs']}")
        if r["params_present"] or r["params_missing"]:
            print(f"  params : present={r['params_present']} missing={r['params_missing']}")

    # --- decision summary ----------------------------------------------
    exists = {r["type"] for r in reports if r["exists"]}
    print("\n=== summary — check against configs/cryosparc_v47.yaml ===")
    print(f"  class2d : use {'class_2D' if 'class_2D' in exists else 'class_2D_new'}  "
          f"(class_2D={'Y' if 'class_2D' in exists else 'n'}, class_2D_new={'Y' if 'class_2D_new' in exists else 'n'})")
    print(f"  refine  : use {'homo_refine' if 'homo_refine' in exists else 'homo_refine_new'}  "
          f"(homo_refine={'Y' if 'homo_refine' in exists else 'n'}, homo_refine_new={'Y' if 'homo_refine_new' in exists else 'n'})")
    print(f"  local_res: local_resolution={'Y' if 'local_resolution' in exists else 'MISSING — find the real key in the list above'}")
    print("  refine volume/mask output ports: see the [homo_refine*] outputs line above")
    print("  import_particles params: see the [import_particles] params line above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
