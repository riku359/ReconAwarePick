#!/usr/bin/env python
"""Hang a select_2D off an existing class_2D and keep only the good classes.

This creates one new select_2D taking the existing class_2D's outputs (`particles` /
`class_averages`) as its inputs, so nothing downstream (ab-initio and beyond) is
touched. Class selection follows upstream (Magellon's `auto_select`):
`interact("get_class_info")` -> `set_class_selected` -> `interact("finish")`.

This is the single-shot cutoff. The paper's iterative workflow is iterate_class2d.py.

The default cutoff 3.5 matches upstream `settings.ini`'s `abinitio_cutoff`.

`--cutoff` is interpreted independently by the two scripts. This one applies its own
`--cutoff` to the raw `cryosift_score` column of scores.csv, so what enters the job is
always the threshold passed here, while the `keep` column of scores.csv and the colours
of the montage reflect the threshold passed to score_class2d.py. Running the two at
different values makes the figure disagree with the job, so change the threshold at the
scoring step and let it flow through.

Usage:
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.purify_class2d --job J15 --cutoff 3.5
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.purify_class2d --job J15 --cutoff 3.5 --dry-run
"""

import argparse
from pathlib import Path

from .cryosift_env import (
    DEFAULT_ENV_FILE,
    connect,
    read_env,
    require,
    resolve_work_dir,
)
from .cryosift_jobs import (
    create_select,
    find_completed_class2d,
    finish_select,
    select_classes,
)
from .score_class2d import read_scores_csv as parse_scores_csv


def read_scores_csv(csv_path):
    """Return score_class2d.py's scores.csv as {class_idx: score}."""
    if not Path(csv_path).is_file():
        raise SystemExit(f"{csv_path} not found. Run score_class2d.py first")
    return parse_scores_csv(csv_path)


def purify(cs, project_uid, job_uid, class_scores, cutoff, dry_run):
    project = cs.find_project(project_uid)
    source = find_completed_class2d(project, job_uid)
    workspace_uid = source.doc["workspace_uids"][0]

    workspace = project.find_workspace(workspace_uid)
    print(f"source  : {project_uid}/{workspace_uid}/{job_uid} ({source.doc.get('status')})")

    select_job = create_select(workspace, job_uid, f"CryoSift purify (cutoff {cutoff})")
    print(f"created : {select_job.uid}")

    selection = select_classes(select_job, class_scores, lambda score: score < cutoff)

    if dry_run:
        print(f"dry-run : leaving {select_job.uid} waiting, without finishing it")
    else:
        print(f"status  : {finish_select(select_job)}")

    print()
    print(f"keep    : {selection.summary()}")
    print(f"drop    : {selection.dropped_particles:,} particles")
    print(f"classes : {sorted(selection.kept_classes)}")

    return select_job, selection.kept_particles


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="class_2D job uid to purify")
    parser.add_argument("--project", help="CryoSPARC project uid (default: CRYOSPARC_PROJECT)")
    parser.add_argument("--cutoff", type=float, default=3.5,
                        help="keep classes with score < cutoff (default 3.5)")
    parser.add_argument("--scores-csv", help="scores.csv, given directly")
    parser.add_argument("--out-root",
                        help="parent to look for scores.csv under (default $RAPICK_WORK/select2d)")
    parser.add_argument("--env", default=None,
                        help=f"CryoSPARC credentials .env (default {DEFAULT_ENV_FILE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="create the select_2D and select classes, but do not finish it")
    args = parser.parse_args()

    env = read_env(args.env)
    project = args.project or require(env, "CRYOSPARC_PROJECT", args.env)
    if args.scores_csv:
        csv_path = Path(args.scores_csv)
    else:
        csv_path = resolve_work_dir(args.out_root) / f"{project}_{args.job}" / "scores.csv"

    class_scores = read_scores_csv(csv_path)
    print(f"scores  : {csv_path} ({len(class_scores)} classes)")

    purify(connect(env, args.env), project, args.job, class_scores, args.cutoff, args.dry_run)


if __name__ == "__main__":
    main()
