#!/usr/bin/env python
"""Run CryoSift's iterative 2D selection, starting from a completed class_2D.

This is the procedure of Fig. 2 of the CryoSift paper (Schaefer et al., bioRxiv
2025.07.28.667259), and the non-interactive equivalent of upstream Magellon's
`main.py`. Rather than creating one select_2D at a single cutoff, this repeats
selection and re-classification in order to cancel the attractor effect -- a few
well-aligning classes pull in the particles of rare viewing angles, so those particles
end up discarded with them.

  1. Score the initial classification and set aside the best `keep_fraction` of the
     classes scoring <= attract_threshold, holding them out of the loop (the attractor
     set)
  2. On the rest, repeat "discard score >= reject_threshold -> re-classify" N times.
     N follows from the extract box (>300 px: 2, 200-300 px: 3, <200 px: 5).
     Exit early if a re-classification's worst score falls below reject_threshold
  3. Merge the loop survivors with the set-aside attractor classes and classify once more
  4. Score that final classification and create one select_2D per cutoff (the
     reconstruction takes 3.5)

Particles that fall out at reject_threshold never come back: the final classification
takes only the loop survivors and the set-aside classes.

There is one structural deviation from upstream. Upstream feeds the subset that the
attractor holdout select_2D emits into the next select_2D; here two select_2D jobs hang
off the initial classification to make the same split. The composition is identical
(attract < score < reject enters the loop), and it avoids relying on how class_idx is
numbered when the input is a subset -- upstream looks its score dictionary up by the
original numbering, so an implementation that renumbers from 0 would break the first
selection silently.

Usage:
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.iterate_class2d --class2d J59 --dry-run
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.iterate_class2d --class2d J59 --gpu 0
"""

import argparse
import json
import math

import numpy as np

from .cryosift_env import (
    DEFAULT_ENV_FILE,
    connect,
    default_gpu,
    read_env,
    require,
    resolve_job_dir,
    resolve_work_dir,
)
from .cryosift_jobs import (
    create_class2d,
    create_select,
    find_completed_class2d,
    finish_select,
    parent_uids,
    queue_and_wait,
    select_classes,
)
from .score_class2d import read_scores_csv, score_job

# Upstream settings.ini defaults; the paper's Discussion argues over the same set.
ATTRACT_THRESHOLD = 2.5     # classes at or below this are candidates to set aside
KEEP_FRACTION = 0.7         # the fraction of those candidates held out, counted in classes
REJECT_THRESHOLD = 4.5      # at or above this a class is discarded permanently in the loop
DEFAULT_CUTOFFS = (2.5, 3.5, 4.5)
RECON_CUTOFF = 3.5          # the final cutoff the paper found best

CRYOSPARC_DEFAULT_BOX_PIX = 256   # extract box when params_spec has no record of one


def plan_for_box(box_size_pix):
    """Pick the classification job type and the cycle count from the extract box.

    The paper's body text contradicts itself on the number of cycles. The Fig. 2 caption
    and the upstream code agree with each other, so follow the code: 2 above 300 px,
    5 below 200 px.
    """
    if box_size_pix > 300:
        return "class_2D", 2
    if box_size_pix < 200:
        return "class_2D_new", 5   # 2D Classification (Small Particle)
    return "class_2D", 3


def find_extract_box(project, class2d_uid, max_visits=24):
    """Read box_size_pix off the extract job among the class_2D's ancestors.

    CryoSPARC does not record a parameter in params_spec when it equals the default, so
    an absent record means the default 256 px was used.
    """
    seen, queue = set(), [class2d_uid]
    while queue and len(seen) < max_visits:
        uid = queue.pop(0)
        if uid in seen:
            continue
        seen.add(uid)

        doc = project.find_job(uid).doc
        job_type = doc.get("type") or doc.get("job_type") or ""
        if job_type.startswith("extract"):
            params = {k: v.get("value") for k, v in (doc.get("params_spec") or {}).items()}
            return int(params.get("box_size_pix", CRYOSPARC_DEFAULT_BOX_PIX)), uid

        queue.extend(parent_uids(project, uid))

    raise SystemExit(f"no extract job among {class2d_uid}'s ancestors; pass --box")


def attractor_threshold(scores, attract_threshold, keep_fraction):
    """Return the score bound of the classes to set aside.

    Same quantile as upstream's `select_high_quality_classes`. Upstream narrows the
    candidates to 0 <= s <= attract_threshold; the lower bound is dropped here because a
    negative score is the BEST class, and losing it from the candidate list shifts the
    quantile. Negative scores are exactly the values upstream's star parser fails to
    read, and this repository reads them with their sign intact.
    """
    candidates = sorted(s for s in scores if s <= attract_threshold)
    index = math.floor(len(candidates) * keep_fraction)
    if index == 0:
        return attract_threshold   # no candidates: use the bound itself, as upstream does
    return candidates[index - 1]


def inherited_params(job_doc):
    """Carry the parent class_2D's params_spec through unchanged (K is one of them).

    Neither the paper nor upstream specifies the number of classes, so the starter job's
    value runs through every cycle.
    """
    params = {k: v.get("value") for k, v in (job_doc.get("params_spec") or {}).items()}
    if params.get("compute_use_ssd") is False:
        raise SystemExit("parent job has compute_use_ssd=false: class_2D dies with SIGFPE")
    return params


def load_state(path):
    return json.loads(path.read_text()) if path.is_file() else {"steps": {}, "rounds": []}


def save_state(path, state):
    path.write_text(json.dumps(state, indent=2))


def resume_or_create(project, state, state_path, name, create_fn, log=print):
    """Reuse a recorded job if there is one, otherwise create it with create_fn.

    A single class_2D runs for hours, so an interrupted run must never rebuild one.
    """
    entry = state["steps"].get(name)
    if entry:
        job = project.find_job(entry["uid"])
        status = job.doc.get("status")
        if status != "completed":
            raise SystemExit(f"[{name}] recorded job {job.uid} is {status}. "
                             f"Clean it up in CryoSPARC, then delete that entry "
                             f"from state.json")
        log(f"[{name}] reuse {job.uid}")
        return job, entry

    job, entry = create_fn()
    state["steps"][name] = entry
    save_state(state_path, state)
    return job, entry


def score_round(cs, project_uid, job_uid, round_dir, rescore=False):
    """Score one cycle's class averages, or read the CSV if it was scored already."""
    csv_path = round_dir / "scores.csv"
    if csv_path.is_file() and not rescore:
        class_scores = read_scores_csv(csv_path)
        return np.array([class_scores[i] for i in sorted(class_scores)]), csv_path

    job_dir = resolve_job_dir(cs, project_uid, job_uid)
    scores, _meta, csv_path = score_job(
        job_dir, round_dir, job_uid=job_uid, cutoff=RECON_CUTOFF,
        montage_name=f"{project_uid}_{job_uid}_montage.png",
    )
    return scores, csv_path


def draw_convergence(png_path, rows, title):
    """Stacked bars of the particle counts over the cycles (Fig. S1's three colours)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    held = np.array([r["held"] for r in rows])
    recycled = np.array([r["recycled"] for r in rows])
    rejected = np.array([r["rejected"] for r in rows])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 5))
    ax.bar(x, held, color="tab:green", label="held out (attractor)")
    ax.bar(x, recycled, bottom=held, color="gold", label="recycled / accepted")
    ax.bar(x, rejected, bottom=held + recycled, color="tab:red", label="rejected (cumulative)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, (held + recycled + rejected).max() * 1.2)   # keep the legend off the bars
    ax.set_ylabel("particles")
    # Text drawn into the figure stays ASCII: matplotlib's default font carries no CJK
    # glyphs and renders them as tofu boxes.
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def run(cs, args):
    project = cs.find_project(args.project)
    parent = find_completed_class2d(project, args.class2d)
    workspace_uid = parent.doc["workspace_uids"][0]
    workspace = project.find_workspace(workspace_uid)

    class_params = inherited_params(parent.doc)
    box_pix, extract_uid = ((args.box, "(--box)") if args.box
                            else find_extract_box(project, args.class2d))
    job_type, iterations = plan_for_box(box_pix)
    if args.iterations:
        iterations = args.iterations

    out_dir = resolve_work_dir(args.out_root) / f"{args.project}_{args.class2d}_iter"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = load_state(state_path)

    print(f"parent  : {args.project}/{workspace_uid}/{args.class2d}")
    print(f"box     : {box_pix} px (from {extract_uid}) -> {job_type} x {iterations} iterations")
    print(f"params  : {class_params}")
    print(f"out dir : {out_dir}")
    print()

    scores, csv_path = score_round(cs, args.project, args.class2d,
                                   out_dir / "round0", rescore=args.rescore)
    class_scores = {i: float(s) for i, s in enumerate(scores)}
    lower = attractor_threshold(scores, args.attract_threshold, args.keep_fraction)

    is_attractor = scores <= lower
    is_rejected = scores >= args.reject_threshold
    is_pool = ~is_attractor & ~is_rejected
    print(f"scores  : min {scores.min():.3f} / med {np.median(scores):.3f} / "
          f"max {scores.max():.3f}  ({csv_path})")
    if (scores < 0).any():
        print(f"note    : {int((scores < 0).sum())} negative scores "
              "(values the upstream parser would drop to 5.0)")
    print(f"attract : score <= {lower:.3f} -> set aside {int(is_attractor.sum())} classes")
    print(f"pool    : {lower:.3f} < score < {args.reject_threshold} -> "
          f"{int(is_pool.sum())} classes into the loop")
    print(f"reject  : score >= {args.reject_threshold} -> "
          f"discard {int(is_rejected.sum())} classes")
    print()

    if args.dry_run:
        print("dry-run : no jobs created. A real run would create, in order:")
        print(f"  select_2D  attractor holdout (score <= {lower:.3f})")
        print(f"  select_2D  round 0           ({lower:.3f} < score < {args.reject_threshold})")
        for i in range(1, iterations):
            print(f"  {job_type}   round {i} classify")
            print(f"  select_2D  round {i}           (score < {args.reject_threshold})"
                  f"   [stops earlier if the worst score is below {args.reject_threshold}]")
        print(f"  {job_type}   final classify (loop + attractor)")
        for cutoff in args.cutoffs:
            print(f"  select_2D  final select (score < {cutoff})")
        return 0

    state.update({
        "project": args.project, "workspace": workspace_uid,
        "parent_class2d": args.class2d, "box_size_pix": box_pix,
        "job_type": job_type, "iterations": iterations, "class2d_params": class_params,
        "thresholds": {"attract": args.attract_threshold, "keep_fraction": args.keep_fraction,
                       "reject": args.reject_threshold, "lower": lower,
                       "cutoffs": list(args.cutoffs)},
    })
    save_state(state_path, state)

    def make_select(name, class2d_uid, scores_map, keep, title):
        def create():
            job = create_select(workspace, class2d_uid, title)
            print(f"[{name}] created {job.uid}: {title}")
            selection = select_classes(job, scores_map, keep)
            finish_select(job)
            print(f"[{name}] {job.uid} keep {selection.summary()}")
            return job, {"uid": job.uid, "type": "select_2D",
                         "kept_classes": selection.kept_classes,
                         "kept_particles": selection.kept_particles,
                         "dropped_particles": selection.dropped_particles}
        return resume_or_create(project, state, state_path, name, create)

    def make_class2d(name, sources, title):
        def create():
            job = create_class2d(workspace, sources, class_params, title, job_type=job_type)
            print(f"[{name}] created {job.uid}: {title}")
            queue_and_wait(job, lane=args.lane, hostname=args.worker, gpus=[args.gpu])
            return job, {"uid": job.uid, "type": job_type, "sources": list(sources)}
        return resume_or_create(project, state, state_path, name, create)

    # 1. Cut the attractor holdout and the loop's initial pool out of the same initial
    #    classification, with two select_2D jobs.
    attractor, attractor_entry = make_select(
        "attractor", args.class2d, class_scores, lambda s: s <= lower,
        f"CryoSift iter {args.class2d}: attractor holdout (score <= {lower:.3f})")
    pool, pool_entry = make_select(
        "round0", args.class2d, class_scores,
        lambda s: lower < s < args.reject_threshold,
        f"CryoSift iter {args.class2d}: round 0 "
        f"({lower:.3f} < score < {args.reject_threshold})")

    # The round 0 select sees every particle of the initial classification, so its
    # dropped count also contains the set-aside classes. Only the classes at or above
    # reject_threshold were really discarded, so subtract the holdout.
    rejected_total = pool_entry["dropped_particles"] - attractor_entry["kept_particles"]
    rounds = [{"label": "round 0", "held": attractor_entry["kept_particles"],
               "recycled": pool_entry["kept_particles"],
               "rejected": rejected_total}]

    # 2. The "discard -> re-classify" cycles. Exit early once re-classification leaves
    #    nothing to discard.
    last_select, last_entry = pool, pool_entry
    for i in range(1, iterations):
        classify, _ = make_class2d(
            f"class2d_round{i}", [(last_select.uid, "particles_selected")],
            f"CryoSift iter {args.class2d}: round {i} classify")
        round_scores, _ = score_round(cs, args.project, classify.uid,
                                      out_dir / f"round{i}", rescore=args.rescore)

        if round_scores.max() < args.reject_threshold:
            print(f"[round{i}] worst score {round_scores.max():.3f} < "
                  f"{args.reject_threshold}, so the loop stops here")
            state["early_exit_round"] = i
            save_state(state_path, state)
            break

        round_map = {j: float(s) for j, s in enumerate(round_scores)}
        last_select, last_entry = make_select(
            f"round{i}", classify.uid, round_map, lambda s: s < args.reject_threshold,
            f"CryoSift iter {args.class2d}: round {i} (score < {args.reject_threshold})")

        rejected_total += last_entry["dropped_particles"]
        rounds.append({"label": f"round {i}", "held": attractor_entry["kept_particles"],
                       "recycled": last_entry["kept_particles"],
                       "rejected": rejected_total})

    # 3. Merge the loop survivors with the set-aside attractor classes and classify again.
    final_classify, _ = make_class2d(
        "class2d_final",
        [(last_select.uid, "particles_selected"), (attractor.uid, "particles_selected")],
        f"CryoSift iter {args.class2d}: final classify (loop + attractor)")
    final_scores, _ = score_round(cs, args.project, final_classify.uid,
                                  out_dir / "final", rescore=args.rescore)
    final_map = {j: float(s) for j, s in enumerate(final_scores)}

    # 4. The final selection, one per cutoff. The reconstruction takes 3.5; the others
    #    are there for the particle-count comparison.
    final_selects = {}
    for cutoff in args.cutoffs:
        job, entry = make_select(
            f"final_{cutoff}", final_classify.uid, final_map,
            lambda s, c=cutoff: s < c,
            f"CryoSift iter {args.class2d}: final select (score < {cutoff})")
        final_selects[str(cutoff)] = entry
        rounds.append({"label": f"cutoff {cutoff}", "held": 0,
                       "recycled": entry["kept_particles"],
                       "rejected": entry["dropped_particles"] + rejected_total})

    state["rounds"] = rounds
    state["final_selects"] = final_selects
    save_state(state_path, state)

    draw_convergence(out_dir / "convergence.png", rounds,
                     f"CryoSift iterative selection: {args.project} {args.class2d}")

    recon = final_selects.get(str(RECON_CUTOFF))
    print()
    print(f"state   : {state_path}")
    print(f"figure  : {out_dir / 'convergence.png'}")
    if recon:
        print(f"final select_2D at cutoff {RECON_CUTOFF}: {recon['uid']} "
              f"({recon['kept_particles']:,} particles) -- this is the job the "
              f"reconstruction stage takes as its input")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class2d", required=True,
                        help="uid of the completed class_2D to start from")
    parser.add_argument("--project", help="CryoSPARC project uid (default: CRYOSPARC_PROJECT)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index for the re-classifications (default: $RAPICK_GPU)")
    parser.add_argument("--worker",
                        help="CryoSPARC worker/lane hostname (default: CRYOSPARC_WORKER)")
    parser.add_argument("--lane", default="default", help="queue lane")
    parser.add_argument("--box", type=int,
                        help="extract box, given directly (default: read from the extract job)")
    parser.add_argument("--iterations", type=int,
                        help="cycle count, given directly (default: derived from the box)")
    parser.add_argument("--attract-threshold", type=float, default=ATTRACT_THRESHOLD)
    parser.add_argument("--keep-fraction", type=float, default=KEEP_FRACTION)
    parser.add_argument("--reject-threshold", type=float, default=REJECT_THRESHOLD)
    parser.add_argument("--cutoffs", default=",".join(str(c) for c in DEFAULT_CUTOFFS),
                        help=f"final selection thresholds "
                             f"(default {','.join(str(c) for c in DEFAULT_CUTOFFS)})")
    parser.add_argument("--out-root", help="output parent (default $RAPICK_WORK/select2d)")
    parser.add_argument("--env", default=None,
                        help=f"CryoSPARC credentials .env (default {DEFAULT_ENV_FILE})")
    parser.add_argument("--rescore", action="store_true",
                        help="re-run the CNN even where a cycle is already scored")
    parser.add_argument("--dry-run", action="store_true",
                        help="score and split only, creating no jobs")
    args = parser.parse_args()
    args.cutoffs = [float(c) for c in args.cutoffs.split(",")]

    env = read_env(args.env)
    args.project = args.project or require(env, "CRYOSPARC_PROJECT", args.env)
    if args.gpu is None:
        args.gpu = default_gpu()
    if not args.dry_run:
        args.worker = args.worker or require(env, "CRYOSPARC_WORKER", args.env)

    return run(connect(env, args.env), args)


if __name__ == "__main__":
    raise SystemExit(main())
