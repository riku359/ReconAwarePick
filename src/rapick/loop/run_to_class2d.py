#!/usr/bin/env python3
"""Run one source from import_particles through class_2D, stopping before reconstruction.

`rapick.recon`'s own `run` continues straight from class_2D into ab-initio x N +
refinement x N, and there is no stopping point in between. The feedback loop evaluates
only the 2D-selected stack, so the un-selected reconstruction under each round's class_2D
would be ~27 GPU-minutes per round that nothing reads. This driver produces the class_2D
that the 2D selection hangs off, and stops.

The steps mirror `rapick.recon.pipeline.run_source` up to class_2D exactly -- same step
reuse, same manifest keys, same class_2D seed rule -- so a manifest written here is the
one the reconstruction drivers expect to find. Keep it that way: if `run_source` changes
above class_2D, this has to change with it.

Run it with the interpreter that has cryosparc-tools (the `recon` environment):

  PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.loop.run_to_class2d \\
      --env .env --profile configs/cryosparc_v47.yaml \\
      --condition configs/recon.yaml \\
      --dataset $RAPICK_WORK/loop/10081/round0/dataset.yaml \\
      --setting annot --source fb_r0 --gpus 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from rapick.recon import coords, manifest as mf
from rapick.recon import pipeline, setup_check
from rapick.recon.api import CryoSPARCApi
from rapick.recon.config import resolve
from rapick.recon.jobs import classification_2d, import_particles, particle_extraction


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env",
                    help="repository-root .env holding CRYOSPARC_* (default: .env)")
    ap.add_argument("--profile", default="configs/cryosparc_v47.yaml")
    ap.add_argument("--condition", required=True,
                    help="the config carrying the class_2D parameters "
                         "(configs/recon.yaml)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--setting", default="annot",
                    help="which micrograph set of the dataset config to use: 'annot' "
                         "(the 300 CryoPPP-annotated micrographs) or 'full' (the whole "
                         "deposition). Default: annot")
    ap.add_argument("--source", required=True,
                    help="which picks to run, keyed in the dataset config")
    ap.add_argument("--star",
                    help="the STAR to classify, when the dataset config does not name it "
                         "(or to override the one it does). The source key above then "
                         "only names the manifest and the output directory.")
    ap.add_argument("--project", help="project UID (default: CRYOSPARC_PROJECT from .env)")
    ap.add_argument("--workspace", help="workspace UID; else resolved by title")
    ap.add_argument("--workspace-title",
                    help="workspace title to get-or-create, overriding the default "
                         "<id>_<setting>. The loop's arms use it to keep one arm's "
                         "rounds out of another's workspace")
    ap.add_argument("--seeds", default="0",
                    help="only seeds[0] is used here (the class_2D seed); reconstruction "
                         "seeds are the reconstruction driver's business")
    ap.add_argument("--gpus", help="comma-separated GPU ids to pin (else auto)")
    ap.add_argument("--worker", help="CryoSPARC worker lane name "
                                     "(default: CRYOSPARC_WORKER from .env)")
    ap.add_argument("--force", action="store_true", help="continue past a preflight FAIL")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = resolve(args.env, args.profile, args.condition, args.dataset)
    # Declare the picks before the preflight: it hashes every star the config names,
    # so a star supplied here has to be in the config by the time it runs.
    cfg.dataset.ensure_source(args.setting, args.source, args.star)

    preflight = setup_check.data_preflight(cfg, args.setting)
    for r in preflight:
        print(f"  [{'OK' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
    if not all(r.ok for r in preflight) and not args.force:
        print("preflight failed - fix the inputs above before running", file=sys.stderr)
        return 1
    if args.dry_run:
        print("--dry-run: nothing created")
        return 0

    project_uid = args.project or cfg.project_uid
    if not project_uid:
        raise SystemExit("no project UID: set CRYOSPARC_PROJECT in your .env "
                         "or pass --project")
    api = CryoSPARCApi(cfg.connection)
    api.connect()
    api.use_project(project_uid)
    ws = args.workspace or api.get_or_create_workspace(
        project_uid,
        args.workspace_title or f"{cfg.dataset.empiar_id}_{args.setting}")

    shared_path = mf.path_for(cfg.work_root, cfg.dataset.name, args.setting, "_shared")
    shared_manifest = mf.load(shared_path) or mf.Manifest(
        dataset=cfg.dataset.name, setting=args.setting, source="_shared")
    source_path = mf.path_for(cfg.work_root, cfg.dataset.name, args.setting, args.source)
    source_manifest = mf.load(source_path) or mf.Manifest(
        dataset=cfg.dataset.name, setting=args.setting, source=args.source)
    for m in (shared_manifest, source_manifest):
        m.project_uid, m.workspace_uid = project_uid, ws

    import os

    hostname = args.worker or cfg.worker or os.uname().nodename
    gpu = ({"gpus": [int(g) for g in args.gpus.split(",")], "hostname": hostname}
           if args.gpus else {"gpus": None, "hostname": hostname, "auto": True})
    seeds = [int(s) for s in args.seeds.split(",")]

    # Persist after every job: a round runs for over an hour, so an abrupt death (host
    # fault, pre-emption) must not lose the record of jobs that already completed.
    def checkpoint():
        mf.save(shared_manifest, shared_path)
        mf.save(source_manifest, source_path)

    try:
        profile = cfg.profile
        shared = pipeline.ensure_shared(api, cfg, args.setting, shared_manifest, gpu=gpu,
                                        checkpoint=checkpoint)
        source_manifest.shared = shared_manifest.shared

        src = cfg.dataset.source(args.setting, args.source)
        ny = coords.dataset_micrograph_height(cfg.dataset.micrograph_glob(args.setting))
        out_dir = Path(cfg.work_root) / cfg.dataset.name / args.setting / args.source
        import_star = coords.normalize_star(src.star, out_dir / "normalized.star", ny,
                                            src.y_flip)
        source_manifest.input_star = {
            "path": src.star, "normalized": import_star, "y_flip": src.y_flip,
            "ny": ny, "sha256": mf.star_sha256(src.star)}

        picks = pipeline._step(
            api, source_manifest.jobs, "import_particles", lambda: import_particles.run(
                api, profile, ws,
                params={"particle_meta_path": import_star, **src.import_params},
                inputs={"micrographs": shared["micrographs"]},
                gpu=pipeline._worker_pin(gpu)),
            fingerprint=source_manifest.input_star["sha256"], checkpoint=checkpoint)

        extract_gpu, n_extract_gpus = pipeline._extract_gpus(gpu)
        extract_params = {"box_size_pix": cfg.dataset.box_size_pix}
        if n_extract_gpus > 1:
            extract_params["compute_num_gpus"] = n_extract_gpus
        extracted = pipeline._step(
            api, source_manifest.jobs, "extract", lambda: particle_extraction.run(
                api, profile, ws, params=extract_params,
                inputs={"particles": picks, "micrographs": shared["ctf"]}, gpu=extract_gpu),
            checkpoint=checkpoint)

        class_params = dict(cfg.condition.class2d_params)
        seed_key = profile.seed_param("class2d")
        if seed_key:
            class_params[seed_key] = seeds[0]
        classes = pipeline._step(
            api, source_manifest.jobs, "class2d", lambda: classification_2d.run(
                api, profile, ws, params=class_params, inputs={"particles": extracted},
                gpu=gpu),
            checkpoint=checkpoint)
    finally:
        checkpoint()

    for step in ("import_particles", "extract", "class2d"):
        rec = source_manifest.jobs.get(step, {})
        print(f"  {step:18s} {rec.get('uid')} ({rec.get('status')})")
    print(f"manifest: {source_path}")
    print(f"CLASS2D={classes.job_uid}")
    return 0 if classes.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
