"""Command-line entrypoint: `rapick-recon <command>`.

Commands parse args, load config, and dispatch. No CryoSPARC / STAR / FSC logic
lives here.

  rapick-recon check-setup --dataset configs/datasets/empiar_10081.yaml --setting full
  rapick-recon prepare     --dataset ... --setting full
  rapick-recon run         --dataset ... --setting full --source cryotransformer_mask \
                           --star $RAPICK_WORK/picks/10081/cryotransformer_mask.star \
                           --seeds 0,1,2
  rapick-recon collect     --dataset ... --setting full --source cryotransformer_mask

An arm whose particles come from a 2D class selection has no chain for `run` to start --
`run` wires class_2D's accepted particles straight into ab-initio. Those start one step
lower, from the Select 2D Classes job that src/rapick/select2d/ produced:

  rapick-recon reconstruct-from-selection --entry 10081 --select2d J212 \
      --condition cryotransformer_mask_select --parent cryotransformer_mask \
      --setting full --seeds 0,1,2

--condition is the class_2D and reconstruction parameters, one file shared by every arm
(configs/recon.yaml); what distinguishes an arm is the STAR handed to it and the
--source it is recorded under. --star declares that STAR whether or not the dataset
config names it. scripts/ drives all of this; these are what it runs.

The CryoSPARC project uid comes from CRYOSPARC_PROJECT in the repository-root .env
(--project overrides it for a one-off); the worker lane comes from CRYOSPARC_WORKER
(--worker overrides it). Neither is ever baked into a config file.
"""
from __future__ import annotations

import argparse
import sys

DEFAULT_ENV = ".env"
DEFAULT_PROFILE = "configs/cryosparc_v47.yaml"
DEFAULT_RECON_CONFIG = "configs/recon.yaml"


def _resolve(args):
    from .config import resolve
    cfg = resolve(args.env, args.profile, args.condition, args.dataset)
    if getattr(args, "no_local_res", False):
        cfg.condition.local_res_enabled = False
    return cfg


def _project_uid(args, cfg) -> str:
    """The CryoSPARC project to write into: CRYOSPARC_PROJECT in .env, or --project."""
    uid = args.project or cfg.project_uid
    if not uid:
        raise SystemExit(
            "no project UID: set CRYOSPARC_PROJECT in your .env (see .env.example) "
            "or pass --project")
    return uid


def _source(args, cfg) -> str:
    """Which picks to run: the condition's own name, since the dataset configs key
    their picks by condition. --source overrides it for an ad-hoc run."""
    return getattr(args, "source", None) or cfg.condition.name


def _workspace_title(cfg, setting: str) -> str:
    """Workspace title = "<empiar id>_<setting>", one workspace per (dataset, scale)
    inside the project named by CRYOSPARC_PROJECT. The setting is part of the title so
    a full-deposition run (~1,000-1,900 micrographs) branches into its own workspace
    instead of colliding with the 300-micrograph annotated run:
    annot -> "10081_annot", full -> "10081_full"."""
    return f"{cfg.dataset.empiar_id}_{setting}"


def _gpu(args, cfg):
    """GPU placement context for GPU jobs.

    With --gpus, pin exactly those cards. Without, return an 'auto' context so each
    GPU job picks a physically-free card at queue time -- this is what stops a job
    from landing on a shared GPU another user has filled and dying with CUDA OOM.
    CPU-only jobs are unaffected (their modules pass gpu=None explicitly).

    The worker lane is CRYOSPARC_WORKER from .env (`cryosparcm cli
    "get_scheduler_targets()"` prints the name your server uses); --worker overrides
    it, and this host's name is the last resort for a single-box install."""
    import os

    hostname = args.worker or cfg.worker or os.uname().nodename
    if getattr(args, "gpus", None):
        return {"gpus": [int(g) for g in args.gpus.split(",")], "hostname": hostname}
    return {"gpus": None, "hostname": hostname, "auto": True}


def _cmd_check_setup(args) -> int:
    from .api import CryoSPARCApi
    from . import setup_check

    cfg = _resolve(args)
    # Declare the arm about to run, so the preflight checks its STAR and not only the
    # ones the dataset config happens to name.
    if args.source or args.star:
        cfg.dataset.ensure_source(args.setting, _source(args, cfg), args.star)
    if args.project:
        cfg.project_uid = args.project
    api = CryoSPARCApi(cfg.connection)
    results = setup_check.check_setup(api, cfg, args.setting)
    for r in results:
        print(f"  [{'OK' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
    return 0 if all(r.ok for r in results) else 1


def _cmd_prepare(args) -> int:
    from .api import CryoSPARCApi

    cfg = _resolve(args)
    api = CryoSPARCApi(cfg.connection)
    api.connect()
    project_uid = _project_uid(args, cfg)
    api.use_project(project_uid)
    ws = args.workspace or api.get_or_create_workspace(
        project_uid, _workspace_title(cfg, args.setting))
    print(f"project={project_uid} workspace={ws}")
    return 0


def _load_manifest(mf, cfg, args, project_uid, ws, source):
    path = mf.path_for(cfg.work_root, cfg.dataset.name, args.setting, source)
    manifest = mf.load(path) or mf.Manifest(
        dataset=cfg.dataset.name, setting=args.setting, source=source)
    manifest.project_uid = project_uid
    manifest.workspace_uid = ws
    return manifest, path


def _cmd_run(args) -> int:
    from .api import CryoSPARCApi
    from . import manifest as mf
    from . import pipeline
    from . import setup_check

    cfg = _resolve(args)
    if args.micrographs:                      # optional subset glob for a fast smoke
        cfg.dataset.setting(args.setting).micrograph_glob = args.micrographs
    source = _source(args, cfg)
    # Declares the source when the config does not name it, so a re-pick or a smoke can
    # be run from a STAR path without editing a committed dataset config.
    cfg.dataset.ensure_source(args.setting, source, args.star)

    # Preflight the inputs before creating any job (data integrity, no server needed).
    # Checks run against the overrides applied above, so a smoke sees its own inputs.
    preflight = setup_check.data_preflight(cfg, args.setting)
    for r in preflight:
        print(f"  [{'OK' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
    if not all(r.ok for r in preflight) and not args.force:
        print("preflight failed — fix the inputs above, or pass --force to override "
              "(e.g. broken micrographs already dropped by a reused shared import)", file=sys.stderr)
        return 1

    api = CryoSPARCApi(cfg.connection)
    api.connect()
    project_uid = _project_uid(args, cfg)
    api.use_project(project_uid)
    ws = args.workspace or api.get_or_create_workspace(
        project_uid, _workspace_title(cfg, args.setting))

    shared_manifest, shared_path = _load_manifest(mf, cfg, args, project_uid, ws, "_shared")
    source_manifest, source_path = _load_manifest(mf, cfg, args, project_uid, ws, source)

    seeds = [int(s) for s in args.seeds.split(",")]
    gpu = _gpu(args, cfg)
    if args.extract_gpus:                     # opt-in: fan the extract step over free cards
        gpu["extract_num_gpus"] = args.extract_gpus
    # Persist after every job, and again on the way out whether or not the run finished.
    # A run lasts hours; a failed step raises (the `finally` catches that) but a host
    # going down kills the process outright, and then only what was already written
    # survives. Either way the next run reuses the completed steps instead of rebuilding
    # an hour of extract and class_2D and orphaning the originals.
    def checkpoint():
        mf.save(shared_manifest, shared_path)
        mf.save(source_manifest, source_path)

    try:
        results = pipeline.run_source(
            api, cfg, args.setting, source, seeds,
            shared_manifest, source_manifest, gpu=gpu, checkpoint=checkpoint)
    finally:
        checkpoint()

    for label, key in [("import", "micrographs"), ("ctf", "ctf"),
                       ("import_particles", "import_particles"),
                       ("extract", "extract"), ("class2d", "class2d"),
                       ("abinit", "abinit"), ("refine", "refine")]:
        r = results.get(key)
        if r:
            print(f"  {label:16s} {r.job_uid} ({r.status})")
    print(f"manifest: {source_path}")
    return 0


def _cmd_reconstruct_from_selection(args) -> int:
    from . import reconstruct_from_selection
    return reconstruct_from_selection.run(args)


def _cmd_collect(args) -> int:
    from .api import CryoSPARCApi
    from . import artifacts, manifest as mf

    cfg = _resolve(args)
    api = CryoSPARCApi(cfg.connection)
    api.connect()
    api.use_project(_project_uid(args, cfg))

    source_path = mf.path_for(cfg.work_root, cfg.dataset.name, args.setting,
                              _source(args, cfg))
    manifest = mf.load(source_path)
    if manifest is None:
        print(f"no manifest at {source_path}; run first", file=sys.stderr)
        return 1
    metrics = artifacts.collect(api, cfg, manifest, source_path.parent)
    print(f"metrics: {source_path.parent / 'metrics.json'}")
    print(f"  counts={metrics['particle_counts']} best_seed={metrics['best_seed']}")
    print(f"  trials={[(t['seed'], t['res_0143']) for t in metrics['trials']]}")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--env", default=DEFAULT_ENV,
                   help="repository-root .env holding CRYOSPARC_* (default: .env)")
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--condition", default=DEFAULT_RECON_CONFIG, metavar="PATH",
                   help="the class_2D and reconstruction parameters, shared by every "
                        f"arm (default: {DEFAULT_RECON_CONFIG})")
    p.add_argument("--dataset", required=True, help="configs/datasets/empiar_<id>.yaml")
    p.add_argument("--setting", default="annot", metavar="SETTING",
                   help="which micrograph set of the dataset config to use: "
                        "'annot' (the 300 CryoPPP-annotated micrographs) or 'full' "
                        "(the whole deposition). Default: annot")
    p.add_argument("--project", help="project UID (default: CRYOSPARC_PROJECT from .env)")
    p.add_argument("--workspace", help="workspace UID (else get/create by title)")


def build_parser() -> argparse.ArgumentParser:
    # The module docstring is the usage text, and its worked examples only read as
    # commands if argparse leaves the line breaks alone.
    p = argparse.ArgumentParser(prog="rapick-recon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("check-setup", help="preflight checks (read-only)")
    _add_common(pc)
    pc.add_argument("--source", default=None,
                    help="the arm about to run, so its STAR is checked too")
    pc.add_argument("--star", help="that arm's STAR, when the dataset config does not "
                                   "name it")
    pc.set_defaults(func=_cmd_check_setup)

    pp = sub.add_parser("prepare", help="get/create the dataset workspace")
    _add_common(pp)
    pp.set_defaults(func=_cmd_prepare)

    pr = sub.add_parser("run", help="run one source's whole chain")
    _add_common(pr)
    pr.add_argument("--source", default=None,
                    help="picks to run, keyed in the dataset config "
                         "(default: the condition's own name)")
    pr.add_argument("--seeds", default="0", help="comma-separated seeds")
    pr.add_argument("--gpus", help="comma-separated GPU IDs for GPU jobs (else scheduler picks)")
    pr.add_argument("--extract-gpus", type=int, default=None, metavar="N",
                    help="run only the (I/O-bound) extract step on up to N physically-free "
                         "GPUs -- parallel micrograph readers win a bigger share of a "
                         "shared or networked disk; other stages keep the run's single card")
    pr.add_argument("--worker", help="CryoSPARC worker lane name "
                                     "(default: CRYOSPARC_WORKER from .env)")
    pr.add_argument("--micrographs", help="override the micrograph glob (subset smoke)")
    pr.add_argument("--star", help="the STAR to run, when the dataset config does not "
                                   "name this source (or to override the one it does)")
    pr.add_argument("--no-local-res", action="store_true", dest="no_local_res",
                    help="skip the local-resolution estimate on the best-of-3 winner")
    pr.add_argument("--force", action="store_true",
                    help="run even if preflight (micrograph health / star distinctness) fails")
    pr.set_defaults(func=_cmd_run)

    # Conditions whose particles come from a 2D class selection cannot be started by
    # `run`, which wires class_2D's accepted particles straight into ab-initio. This
    # subcommand starts one step lower, from an existing select_2D job; it declares its
    # own flags (the module documents them) and does not collect afterwards.
    from .reconstruct_from_selection import add_arguments as _add_from_selection
    ps = sub.add_parser("reconstruct-from-selection",
                        help="reconstruct from an existing Select 2D Classes job "
                             "(conditions select / both / cryosegnet_both / fb)")
    _add_from_selection(ps)
    ps.set_defaults(func=_cmd_reconstruct_from_selection)

    pl = sub.add_parser("collect", help="rebuild metrics.json from finished jobs, "
                                        "without re-running any of them")
    _add_common(pl)
    pl.add_argument("--source", default=None,
                    help="picks to collect (default: the condition's own name)")
    pl.set_defaults(func=_cmd_collect)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NotImplementedError:
        print(f"rapick-recon: '{args.command}' is not implemented.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
