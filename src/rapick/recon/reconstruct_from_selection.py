#!/usr/bin/env python3
"""Reconstruct from an existing Select 2D Classes job.

`rapick-recon run` wires class_2D's accepted particles straight into ab-initio, so there
is nowhere in it to inject a class selection. This driver starts one step lower instead:
it wraps an EXISTING `select_2D` job as the particle source and reuses
`pipeline.reconstruct_trial` / `pipeline.choose_best_trial` verbatim, so the arm runs the
same `abinit x3 -> homo_refine x3 -> best-of-3 by GSFSC 0.143 -> local_resolution` chain
as the condition it is read against.

Four conditions need it, because their particles come from a 2D class selection rather
than from a STAR of their own:

    --condition       --parent      appears as
    select            baseline      Table 4 row 3 (+select)
    both              mask          Table 4 row 4 (+both), Table 8
    cryosegnet_both   cryosegnet    Table 8
    fb                fb            Table 2 (Ours), Table 4 row 5, Table 7

`--parent` names the condition whose `class_2D` the selection sits on. `fb` is its own
parent: it imports, extracts and classifies its own stack under that same name, and only
the reconstruction starts at the selection.

The manifest is seeded from the parent condition's manifest, because the shared
import + CTF and the condition's `import_particles` / `extract` / `class_2D` genuinely
are the same physical jobs -- only the particle subset differs. That also lets
`rapick-recon collect` report particle counts for the arm without a special case.

THIS DRIVER DOES NOT COLLECT. Run `rapick-recon collect` afterwards with the same
`--condition` / `--dataset` / `--setting`, or the arm ends up with a manifest and no
`metrics.json`.

  rapick-recon reconstruct-from-selection --entry 10081 --select2d J212 \
      --condition select --parent baseline --setting full --seeds 0,1,2

  rapick-recon collect --condition configs/conditions/select.yaml \
      --dataset configs/datasets/empiar_10081.yaml --setting full

Every run re-verifies that the named `select_2D` descends from the `class_2D` the
parent's manifest recorded, walking up through 2D jobs only. A mismatch aborts:
rebuilding an arm on the wrong parent silently produces a comparison between two
different particle stacks, and the same check is what catches a `--parent` that does not
match the selection.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Same defaults as `rapick-recon run`; both are relative to the repository root.
DEFAULT_ENV = ".env"
DEFAULT_PROFILE = "configs/cryosparc_v47.yaml"
CONDITIONS_DIR = Path("configs/conditions")
DATASETS_DIR = Path("configs/datasets")

# The `select_2D` output that carries the kept classes. `particles_excluded` is the
# discarded half; wiring that by accident would reconstruct exactly the junk we removed.
SELECTED_PARTICLES = "particles_selected"

# Which condition's class_2D each pre-selected condition's selection sits on. Only a
# default for --parent, which always wins; the ancestry walk below verifies whichever
# one is used, so a wrong parent is caught rather than believed.
DEFAULT_PARENT = {
    "select": "baseline",
    "both": "mask",
    "cryosegnet_both": "cryosegnet",
    "fb": "fb",
}

# The only job types a selection chain may pass through on its way down from the parent
# condition's class_2D. A single-cutoff selection is one hop; CryoSift's iterative
# workflow (src/rapick/select2d/iterate_class2d.py) alternates select_2D and class_2D for
# several cycles, and the paper's conditions all use that one.
CHAIN_JOB_TYPES = ("select_2D", "class_2D", "class_2D_new")


def _parent_uids(api, project_uid: str, job_uid: str) -> list:
    doc = api.find_job(project_uid, job_uid).doc
    return [c["job_uid"]
            for g in doc.get("input_slot_groups") or []
            for c in g.get("connections") or [] if c.get("job_uid")]


def trace_to_class2d(api, project_uid: str, select_uid: str, expected_class2d: str,
                     max_visits: int = 64) -> list:
    """Walk up from a select_2D and require every path to end at `expected_class2d`.

    A direct-parent check would reject the iterative selection, whose final select_2D
    sits several select_2D/class_2D hops above the parent condition's class_2D. Widening
    it to an ancestry walk keeps the guard that matters: the walk refuses to leave the 2D
    jobs, so a chain rooted in a different extract (i.e. a different particle stack) is
    caught the moment it reaches that stack's own class_2D and tries to climb past it.
    """
    chain: list = []
    queue, seen = [select_uid], set()
    while queue:
        uid = queue.pop(0)
        if uid in seen:
            continue
        seen.add(uid)
        chain.append(uid)

        if uid == expected_class2d:
            continue                       # stop here; above it lies extract/import
        if len(seen) > max_visits:
            sys.exit(f"chain from {select_uid} exceeded {max_visits} jobs without "
                     f"reaching {expected_class2d}")

        doc = api.find_job(project_uid, uid).doc
        job_type = doc.get("job_type") or doc.get("type")
        if job_type not in CHAIN_JOB_TYPES:
            sys.exit(f"{uid} is a {job_type!r}: the chain from {select_uid} leaves the "
                     f"2D jobs before reaching {expected_class2d} (wrong particle stack)")

        parents = _parent_uids(api, project_uid, uid)
        if not parents:
            sys.exit(f"{uid} has no inputs: the chain from {select_uid} never reaches "
                     f"{expected_class2d}")
        queue.extend(parents)

    if expected_class2d not in chain:
        sys.exit(f"{select_uid} does not descend from {expected_class2d} "
                 f"(the class_2D recorded in the parent condition's manifest)")
    return chain


def verify_select2d(api, project_uid: str, select_uid: str,
                    expected_class2d: str) -> tuple:
    """Check the select_2D job is usable and really sits on `expected_class2d`.

    Returns its counts and the job chain back to the parent's class_2D. Exits with a
    message on any mismatch rather than letting a wrong-parent arm reach the GPU.
    """
    doc = api.find_job(project_uid, select_uid).doc

    if doc.get("job_type") != "select_2D":
        sys.exit(f"{select_uid} is a {doc.get('job_type')!r}, not select_2D")
    if doc.get("status") != "completed":
        sys.exit(f"{select_uid} is {doc.get('status')!r}, not completed")
    if doc.get("deleted"):
        sys.exit(f"{select_uid} is deleted in CryoSPARC")

    chain = trace_to_class2d(api, project_uid, select_uid, expected_class2d)

    counts = {g["name"]: g.get("num_items")
              for g in doc.get("output_result_groups") or []
              if g.get("type") in ("particle", "template")}
    if not counts.get(SELECTED_PARTICLES):
        sys.exit(f"{select_uid} has no {SELECTED_PARTICLES} particles: {counts}")
    return counts, chain


def _count(counts: dict, key: str) -> str:
    """A count for the log line, or '?' where the job declares no such output group."""
    value = counts.get(key)
    return f"{value:,}" if isinstance(value, int) else "?"


def seed_manifest(parent, select_uid: str, counts: dict, condition: str,
                  chain=None):
    """A manifest for this arm that inherits every job it shares with its parent.

    Everything up to and including class_2D is the same physical job, so it is copied
    rather than re-recorded; `select2d` is the only new upstream step.
    """
    from . import manifest as mf

    m = mf.Manifest(
        dataset=parent.dataset, setting=parent.setting, source=condition,
        project_uid=parent.project_uid, workspace_uid=parent.workspace_uid,
        shared=dict(parent.shared),
        input_star=dict(parent.input_star),
    )
    for step in ("import_particles", "extract", "class2d"):
        if step in parent.jobs:
            m.jobs[step] = dict(parent.jobs[step])
    m.jobs["select2d"] = {
        "uid": select_uid, "job_type": "select_2D", "status": "completed",
        "outputs": {"particles": SELECTED_PARTICLES},
        "selected_particles": counts.get(SELECTED_PARTICLES),
        "excluded_particles": counts.get("particles_excluded"),
        "selected_classes": counts.get("templates_selected"),
        "excluded_classes": counts.get("templates_excluded"),
        # Provenance for the iterative selection: which jobs sit between the selection
        # and the parent's class_2D. One hop for a single-cutoff selection.
        "chain": chain or [select_uid],
    }
    return m


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare this driver's flags. Shared with `rapick-recon`'s subparser."""
    parser.add_argument("--entry", required=True, metavar="ID",
                        help="EMPIAR entry, e.g. 10081")
    parser.add_argument("--select2d", required=True, metavar="UID",
                        help="uid of the completed Select 2D Classes job to reconstruct "
                             "from. For CryoSift's iterative workflow this is the FINAL "
                             "selection, cutoff 3.5: read it out of the cycle's "
                             "state.json (see src/rapick/select2d/README.md). Never "
                             "guessed and never defaulted")
    parser.add_argument("--condition", required=True, metavar="NAME",
                        help="condition this run is recorded as: select, both, "
                             "cryosegnet_both or fb. Also names the manifest directory "
                             "and, unless --condition-config says otherwise, the config "
                             f"read from {CONDITIONS_DIR}/<NAME>.yaml")
    parser.add_argument("--parent", metavar="NAME", default=None,
                        help="condition whose class_2D this selection sits on: baseline "
                             "for select, mask for both, cryosegnet for cryosegnet_both, "
                             "fb for fb (the default for those four). Its manifest "
                             "supplies the shared jobs and the class_2D the ancestry "
                             "check walks up to")
    parser.add_argument("--setting", default="full", choices=("annot", "full"),
                        help="which micrograph set of the dataset config the parent ran "
                             "on: 'annot' (the 300 CryoPPP-annotated micrographs) or "
                             "'full' (the whole deposition). Default: full, the scale "
                             "every reconstruction-level result of the paper uses")
    parser.add_argument("--dataset", default=None, metavar="PATH",
                        help=f"dataset config (default {DATASETS_DIR}/empiar_<entry>.yaml)")
    parser.add_argument("--condition-config", default=None, metavar="PATH",
                        dest="condition_config",
                        help=f"condition config (default {CONDITIONS_DIR}/"
                             "<condition>.yaml). Pass it only for an ad-hoc arm whose "
                             "name is not one of the repository's conditions")
    parser.add_argument("--seeds", default="0",
                        help="comma-separated seeds to fork ab-initio and refinement "
                             "over. PASS 0,1,2: a single-seed resolution is not "
                             "trustworthy, and the default is one seed deliberately so "
                             "that a one-seed run is never accidental")
    parser.add_argument("--gpu", default=None, metavar="N",
                        help="GPU index to pin every job to (default: RAPICK_GPU)")
    parser.add_argument("--worker", default=None,
                        help="CryoSPARC worker lane name (default: CRYOSPARC_WORKER "
                             "from .env)")
    parser.add_argument("--project", default=None,
                        help="project UID (default: CRYOSPARC_PROJECT from .env)")
    parser.add_argument("--env", default=DEFAULT_ENV,
                        help="repository-root .env holding CRYOSPARC_* (default: .env)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="verify the selection and print what would run, create no "
                             "jobs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapick-recon reconstruct-from-selection", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return parser


def run(args) -> int:
    """Verify the selection, then run the same chain the parent condition ran."""
    from . import config, manifest as mf, pipeline
    from .api import CryoSPARCApi
    from .jobs import local_resolution
    from .jobs._base import JobResult

    condition_path = (args.condition_config
                      or str(CONDITIONS_DIR / f"{args.condition}.yaml"))
    dataset_path = args.dataset or str(DATASETS_DIR / f"empiar_{args.entry}.yaml")
    cfg = config.resolve(args.env, args.profile, condition_path, dataset_path)

    # Every path and credential is named when it is missing, rather than defaulted to
    # something that happens to exist here (docs/CONFIGURATION.md).
    if not cfg.work_root:
        sys.exit("RAPICK_WORK is not set: it is where the manifests live "
                 "(see docs/CONFIGURATION.md)")
    project_uid = args.project or cfg.project_uid
    if not project_uid:
        sys.exit("no project UID: set CRYOSPARC_PROJECT in your .env "
                 "(see .env.example) or pass --project")
    worker = args.worker or cfg.worker
    if not worker:
        sys.exit("no worker lane: set CRYOSPARC_WORKER in your .env "
                 '(`cryosparcm cli "get_scheduler_targets()"` prints it) or pass --worker')
    gpu_index = args.gpu if args.gpu not in (None, "") else os.environ.get("RAPICK_GPU", "")
    gpu_index = str(gpu_index).strip()
    if not gpu_index:
        sys.exit("no GPU index: set RAPICK_GPU (see docs/CONFIGURATION.md) or pass --gpu")
    if not gpu_index.isdigit():
        sys.exit(f"--gpu/RAPICK_GPU must be a card index, got {gpu_index!r}")

    parent = args.parent or DEFAULT_PARENT.get(args.condition)
    if not parent:
        sys.exit(f"--parent is required for condition {args.condition!r}: name the "
                 f"condition whose class_2D this selection sits on")

    parent_path = mf.path_for(cfg.work_root, cfg.dataset.name, args.setting, parent)
    parent_manifest = mf.load(parent_path)
    if parent_manifest is None:
        sys.exit(f"no manifest for parent condition {parent!r} at {parent_path}: run "
                 f"`rapick-recon run --condition {CONDITIONS_DIR}/{parent}.yaml "
                 f"--dataset {dataset_path} --setting {args.setting}` first")
    expected_class2d = (parent_manifest.jobs.get("class2d") or {}).get("uid")
    if not expected_class2d:
        sys.exit(f"the manifest of parent condition {parent!r} ({parent_path}) records "
                 f"no class2d job, so there is nothing to check the selection against")
    if not parent_manifest.workspace_uid:
        sys.exit(f"the manifest of parent condition {parent!r} ({parent_path}) records "
                 f"no workspace; this arm's jobs belong in the same one")

    api = CryoSPARCApi(cfg.connection)
    api.connect()
    api.use_project(project_uid)

    counts, chain = verify_select2d(api, project_uid, args.select2d, expected_class2d)
    print(f"[select2d] {args.entry}  {args.select2d} <- class_2D {expected_class2d} "
          f"({parent})  selected {_count(counts, SELECTED_PARTICLES)} particles / "
          f"{counts.get('templates_selected')} classes "
          f"(excluded {_count(counts, 'particles_excluded')} / "
          f"{counts.get('templates_excluded')})")
    if len(chain) > 2:
        print(f"[select2d] chain: {' <- '.join(chain)}")

    out_path = mf.path_for(cfg.work_root, cfg.dataset.name, args.setting, args.condition)
    m = mf.load(out_path) or seed_manifest(parent_manifest, args.select2d, counts,
                                           args.condition, chain)
    m.project_uid = project_uid
    m.workspace_uid = parent_manifest.workspace_uid

    # Resuming into the wrong arm (a mistyped --condition) would reuse that arm's
    # recorded seeds while feeding a different particle subset, so the trials would mix
    # two stacks.
    recorded_select = (m.jobs.get("select2d") or {}).get("uid")
    if recorded_select and recorded_select != args.select2d:
        sys.exit(f"{out_path} already records select_2D {recorded_select}, "
                 f"not {args.select2d}: --condition {args.condition} points at "
                 f"another arm")

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.dry_run:
        print(f"[select2d] dry run: would run seeds {seeds} on GPU {gpu_index} "
              f"in {m.workspace_uid}, manifest {out_path}")
        return 0

    gpu = {"gpus": [int(gpu_index)], "hostname": worker}
    selected = JobResult(name="select2d", job_uid=args.select2d, job_type="select_2D",
                         status="completed", outputs={"particles": SELECTED_PARTICLES})

    # Checkpoint after every job, not just every seed: a seed is an abinit plus a refine
    # and each runs for hours, so a host going down between them would otherwise lose the
    # abinit's record and rebuild it.
    def checkpoint():
        mf.save(m, out_path)

    trials = {}
    for seed in seeds:
        trials[seed] = pipeline.reconstruct_trial(api, cfg, selected, seed, m, gpu=gpu,
                                                  checkpoint=checkpoint)
        checkpoint()

    best_seed = pipeline.choose_best_trial(api, cfg, m, trials)
    print(f"[select2d] {args.entry} best seed {best_seed}: "
          f"{[(t.seed, t.res_0143) for t in m.trials]}")

    if cfg.condition.local_res_enabled:
        try:
            res = pipeline._step(api, m.jobs, "local_res", lambda: local_resolution.run(
                api, cfg.profile, m.workspace_uid, params={},
                inputs={"volume": trials[best_seed]["refine"]}, gpu=gpu))
            m.local_res = res.job_uid
        except Exception as exc:
            print(f"[select2d] local_res failed for {args.entry}: {exc}; continuing",
                  file=sys.stderr)

    m.status = "done"
    mf.save(m, out_path)
    print(f"[select2d] {args.entry} done -> {out_path}")
    print(f"[select2d] this driver writes no metrics.json. Run: rapick-recon collect "
          f"--condition {condition_path} --dataset {dataset_path} "
          f"--setting {args.setting}")
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
