#!/usr/bin/env python3
"""Drive the reconstruction-aware feedback loop on one EMPIAR entry (paper Sec. 3.5).

One round is

  pick the 300 annotated micrographs with the current checkpoint
    -> score the picks against the CryoPPP annotation
    -> discard the picks that land on contamination (stored masks, no inference)
    -> import_particles / extract / class_2D                    (run_to_class2d.py)
    -> iterative 2D class selection                             (the select_2D tool)
    -> teacher labels from the surviving particles              (export_teacher_star.py)
    -> fine-tune theta_0 on them                                (finetune.py)
    -> promote the result to the next round's picking checkpoint

Every round fine-tunes theta_0 again, never the checkpoint that just picked:

    theta_{n+1} = FineTune(theta_0; S_n)                                    (Eq. 1)

which is what TranSPHIRE's own implementation does -- its `--weights_old` is assigned
once at session start and never updated, while only the picking weights advance. Round 0
does no fine-tuning of its own; round n's fine-tune produces the model round n+1 picks
with, so three rounds give two trained checkpoints and the paper reports the first.

The loop reconstructs nothing. At 300 micrographs the seed-to-seed spread of GSFSC 0.143
is the size of the effect being looked for, so a per-round trio of ab-initio + refinement
would cost an hour a round and answer nothing. What each round reports instead are the
GT-free diagnostics of the selection -- pick count, permanent-reject fraction, final
survival fraction -- which are deterministic given the picks. 3D happens once, at fullset
scale, on the checkpoint the loop produced: repick_fullset.py.

Every step records itself in state.json and is skipped when already done, because a round
runs for hours and long unattended runs are exactly where a host fault or a pre-emption
costs the most progress. Re-running the driver resumes.

Round 0's score is a hard gate: the same checkpoint under the same operating point has to
reproduce this entry's theta_0 row (Sec. S2, `entries.ENTRIES[...].gate`). A mismatch
means the checkpoint, the images or the preprocessing differ, and every downstream number
would inherit that -- so the loop stops there rather than guessing.

`--teacher gt` replaces the round's labels with the CryoPPP annotations of the same
micrographs, holding everything else fixed. That is the perfect-teacher upper bound of
Table 7's lower row rather than a feedback loop, so it runs for one round; it writes into
its own arm (`fb_gt`) so it never overwrites the arm it is read against.

  python -m rapick.loop.run_loop --id 10081 --rounds 0-2
  python -m rapick.loop.run_loop --id 10081 --rounds 0 --stop-after score
  python -m rapick.loop.run_loop --id 10532 --rounds 0-2 --gpu 1 --teacher-mics all
  python -m rapick.loop.run_loop --id 10081 --rounds 0 --last-round 1 --teacher gt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import entries, make_gt_teacher, paths, star
from .common import (LOCK_DIR, State, acquire_lock, log, parse_rounds, run,
                     wait_for_free_gpu)

# The operating point every round and every entry picks at: the top 75% of 600 queries
# per micrograph, NMS 0.7. It is relative on purpose. An absolute score threshold cannot
# be held fixed across rounds, because fine-tuning moves the scale the scores live on, so
# the same number would mean a different operating point each round.
PICK_ARGS = ["--backbone", "resnet152", "--num_queries", "600",
             "--quartile_threshold", "0.25", "--nms_threshold", "0.7",
             "--selection", "legacy_idxfix", "--gt-format",
             "--save_micrographs_with_encircled_proteins", "N"]

# What the contamination filter names its outputs. Passed explicitly rather than left to
# the filter's default, so that renaming it there cannot silently break the wiring.
STAR_PREFIX = "cryotransformer"
MASK_SUFFIX = "_tri"
CLEAN_STAR = f"{STAR_PREFIX}_clean{MASK_SUFFIX}.star"
FILTER_SUMMARY = f"summary{MASK_SUFFIX}.json"

# Round 0 has to land this close to theta_0's recorded row before the loop continues.
GATE_METRIC_TOL = 0.02    # absolute, on P/R/F1
GATE_COUNT_TOL = 0.03     # relative, on the pick count

STEPS = ("pick", "score", "filter", "class2d", "select2d", "teacher", "finetune",
         "promote")

# The 40/10 split of the 50 teacher micrographs, as a fraction handed to finetune.py.
# The 10 validation micrographs monitor the loss; they select nothing.
VAL_FRACTION = "0.2"

# A fine-tune started on a card somebody else has filled OOMs minutes in, and the round
# has to be redone from its `pick` step, so the driver waits for the card instead.
FT_MIN_FREE_MB = int(os.environ.get("RAPICK_FT_MIN_FREE_MB", "20000"))
FT_MAX_WAIT_S = int(os.environ.get("RAPICK_FT_MAX_WAIT_S", "7200"))


@dataclass
class Run:
    """One invocation: the entry, the arm and everything chosen on the command line."""

    entry: entries.Entry
    arm: entries.Arm
    gpu: str
    worker: str
    project: str
    last_round: int
    teacher_mics: str
    teacher_mics_from: str

    @property
    def empiar(self) -> str:
        return self.entry.empiar

    def source(self, n: int) -> str:
        return entries.source_name(self.arm.name, n)

    def model(self, n: int) -> Path:
        return entries.model_path(self.empiar, n, self.arm.name)

    def teacher_star(self, round_dir: Path) -> Path:
        """The labels this round fine-tunes on: the surviving picks, or the annotation."""
        return round_dir / ("teacher_gt.star" if self.arm.teacher == entries.TEACHER_GT
                            else "teacher.star")


def write_round_dataset(n: int, ctx: Run, round_dir: Path) -> Path:
    """A dataset config listing rounds 0..n only.

    The reconstruction pipeline's preflight hashes *every* declared star and fails on a
    missing one, so a round cannot run against a config naming rounds that have not
    picked yet. Listing exactly the finished rounds also turns the distinctness check
    into a live guard: if a round's model reproduces an earlier round's picks exactly,
    the identical-star check catches it instead of the run quietly comparing a stack
    against a copy of itself.
    """
    entry = ctx.entry
    sources = "\n".join(
        f'      {ctx.source(i)}:\n'
        f'        star: "{entries.round_dir(ctx.empiar, i, ctx.arm.name) / CLEAN_STAR}"\n'
        f'        import_params: {{}}\n'
        f'        y_flip: true'
        for i in range(n + 1))
    text = f"""# GENERATED by rapick.loop.run_loop for {entry.empiar} round {n} -- do not edit.
# Restricted to the rounds whose stars exist; see write_round_dataset for why.
name: empiar_{entry.empiar}
empiar_id: "{entry.empiar}"

optics:
  psize_A: {entry.psize_A}
  accel_kv: 300
  cs_mm: 2.7
  total_dose_e_per_A2: 50.0

extraction:
  box_size_pix: {entry.box_size_pix}

settings:
  {entries.SETTING_ANNOT}:
    micrographs: "{paths.annotated_micrographs(entry.empiar)}/*.mrc"
    expected_micrograph_count: {entries.SUBSET_MICROGRAPHS}
    # Every round's picks carry the GT-aligned top-left Y origin, so all need the flip.
    sources:
{sources}
"""
    path = round_dir / "dataset.yaml"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def step_pick(n, st, rd, ctx: Run):
    """Pick the 300 annotated micrographs with round n's checkpoint."""
    ckpt = ctx.model(n)
    if not Path(ckpt).is_file():
        raise RuntimeError(f"round {n} has no checkpoint at {ckpt}")
    remarks = ctx.source(n)
    run(paths.tool_cmd("predict") +
        ["--empiar", ctx.empiar, "--data_root", str(paths.annotated_data_root()),
         "--resume", str(ckpt), *PICK_ARGS,
         "--device", f"cuda:{ctx.gpu}", "--remarks", remarks],
        cwd=paths.tool_cwd("predict"), log_path=rd / "logs" / "pick.log",
        env_extra=paths.tool_env("predict"))

    pred_root = paths.tool_cwd("predict") / "output" / "predictions"
    candidates = sorted(
        pred_root.glob(f"predictions_EMPIAR_{ctx.empiar}_remarks_{remarks}_timestamp_*"),
        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"no prediction dir for {remarks} under {pred_root}")
    produced = candidates[-1] / f"EMPIAR_{ctx.empiar}_remarks_{remarks}_star_file.star"
    if not produced.is_file():
        raise RuntimeError(f"combined star missing: {produced}")
    shutil.copyfile(produced, rd / "picks.star")
    st.mark("pick", checkpoint=str(ckpt), prediction_dir=str(candidates[-1]),
            picks_star=str(rd / "picks.star"))


def step_score(n, st, rd, ctx: Run):
    """Score the raw picks against the CryoPPP annotation, and gate round 0 on it.

    The 50 micrographs a round trains on are drawn from the same 300 this scores, so
    round 1 onward has ~17% of its evaluation set inside its training set. That is
    deliberate rather than an oversight: the loop's purpose is specialisation to one
    entry, not generalisation to unseen data, so fitting the data is the mechanism. It
    is reported as such wherever these numbers appear.
    """
    scorer = paths.tool_script("scorer")
    out = run(paths.tool_cmd("scorer") +
              ["--id", ctx.empiar, "--pred", str(rd / "picks.star"),
               "--gt", str(ctx.entry.gt_star), "--json"],
              cwd=scorer.parent, log_path=rd / "logs" / "score.log")
    payload = json.loads(next(l for l in out.splitlines() if l.startswith("JSON "))[5:])
    log(f"round {n} picks={payload['n_pred_eval']} macro P/R/F1 = "
        f"{payload['macro_P']:.3f}/{payload['macro_R']:.3f}/{payload['macro_F1']:.3f}")

    if n == 0:
        gate = ctx.entry.gate
        drift = {k: payload[k] - gate[k] for k in ("macro_P", "macro_R", "macro_F1")}
        count_rel = abs(payload["n_pred_eval"] - gate["n_pred_eval"]) / gate["n_pred_eval"]
        bad = [k for k, d in drift.items() if abs(d) > GATE_METRIC_TOL]
        if count_rel > GATE_COUNT_TOL:
            bad.append("n_pred_eval")
        if bad:
            raise RuntimeError(
                f"round 0 gate FAILED on {', '.join(bad)}.\n"
                f"  got      P/R/F1 = {payload['macro_P']:.3f}/{payload['macro_R']:.3f}/"
                f"{payload['macro_F1']:.3f}  picks = {payload['n_pred_eval']}\n"
                f"  expected P/R/F1 = {gate['macro_P']}/{gate['macro_R']}/{gate['macro_F1']}"
                f"  picks = {gate['n_pred_eval']}  (theta_0, Sec. S2)\n"
                f"The checkpoint, the input images or the preprocessing differ from the "
                f"run that produced the reference. Stopping before anything downstream.")
        log(f"round 0 gate PASSED (max |drift| {max(abs(d) for d in drift.values()):.3f}, "
            f"pick count {count_rel:.2%} off)")
    st.mark("score", **payload)


def step_filter(n, st, rd, ctx: Run):
    """Discard the picks whose centre lands on contamination.

    The masks are read from the store rather than recomputed: they depend on the
    micrograph and not on the picks, so every round of every arm applies the same ones.
    """
    if not ctx.arm.masked:
        # Copy rather than delete the stage: everything downstream is wired to the
        # cleaned star's name, so passing the picks through unchanged keeps both the
        # wiring and the shape of state.json identical between the arms.
        source = rd / "picks.star"
        shutil.copyfile(source, rd / CLEAN_STAR)
        total = star.count_star_particles(source)
        log(f"round {n} contamination filter SKIPPED ({ctx.arm.name}): "
            f"{total} picks pass through")
        st.mark("filter", picks_total=total, picks_kept=total, picks_removed=0,
                removed_fraction=0.0, masked=False)
        return

    mask_dir = ctx.entry.mask_dir
    if not mask_dir.is_dir():
        raise RuntimeError(f"no stored masks for {ctx.empiar} at {mask_dir}")
    run(paths.tool_cmd("mask_filter") +
        ["--star", str(rd / "picks.star"), "--empiar-id", ctx.empiar,
         "--mask-dir", str(mask_dir), "--out-dir", str(rd),
         "--star-prefix", STAR_PREFIX, "--suffix", MASK_SUFFIX, "--overwrite"],
        cwd=paths.tool_cwd("mask_filter"), log_path=rd / "logs" / "filter.log")
    summary = json.loads((rd / FILTER_SUMMARY).read_text())
    log(f"round {n} contamination filter kept {summary['picks_kept']} of "
        f"{summary['picks_total']} ({summary['removed_fraction']:.2%} removed)")
    st.mark("filter", masked=True, **{k: summary[k] for k in
                                      ("picks_total", "picks_kept", "picks_removed",
                                       "removed_fraction")})


def step_class2d(n, st, rd, ctx: Run):
    """import_particles -> extract -> class_2D, stopping before any reconstruction."""
    dataset = write_round_dataset(n, ctx, rd)
    # The workspace is addressed by title, not uid, so a new entry lands in its own
    # workspace without a lookup table of uids that only one CryoSPARC database can
    # confirm.
    out = run([paths.recon_python(), "-m", "rapick.loop.run_to_class2d",
               "--env", str(paths.env_file()),
               "--profile", str(paths.recon_profile()),
               "--condition", str(paths.condition(entries.LOOP_CONDITION)),
               "--dataset", str(dataset), "--setting", entries.SETTING_ANNOT,
               "--project", ctx.project, "--source", ctx.source(n),
               "--gpus", str(ctx.gpu), "--worker", ctx.worker,
               "--workspace-title", f"{ctx.empiar}_{entries.SETTING_ANNOT}"
                                    f"{ctx.arm.workspace_suffix}"],
              cwd=paths.REPO_ROOT, log_path=rd / "logs" / "class2d.log",
              env_extra=paths.recon_env())
    uid = next(l.split("=", 1)[1].strip() for l in out.splitlines()
               if l.startswith("CLASS2D="))
    log(f"round {n} class_2D = {uid}")
    st.mark("class2d", uid=uid, dataset=str(dataset))


def step_select2d(n, st, rd, ctx: Run):
    """Run the iterative 2D class selection and record its GT-free diagnostics."""
    class2d = st.get("class2d", "uid")
    # --out-root explicitly: the tool would otherwise infer a root of its own, and this
    # driver has to know the same path to read the cycle's state back.
    run(paths.tool_cmd("select_2d") +
        ["--class2d", class2d, "--project", ctx.project, "--gpu", str(ctx.gpu),
         "--worker", ctx.worker, "--out-root", str(paths.select2d_root()),
         "--env", str(paths.env_file())],
        cwd=paths.tool_cwd("select_2d"), log_path=rd / "logs" / "select2d.log",
        env_extra=paths.tool_env("select_2d"))

    state_path = paths.select2d_root() / f"{ctx.project}_{class2d}_iter" / "state.json"
    cycle = json.loads(state_path.read_text())
    final = cycle["steps"]["final_3.5"]
    log(f"round {n} final select_2D = {final['uid']} ({final['kept_particles']} particles, "
        f"{len(final['kept_classes'])} classes)")

    # The permanently discarded count is not a field of its own: each select_2D's
    # dropped_particles is "everything this select did not keep", so the first select's
    # dropped count includes the attractor set that was held out of the cycle, not just
    # the rejects. Subtract both kept sets from the class_2D total instead. This is the
    # GT-free diagnostic the design leans on, so it is derived once here rather than left
    # to be re-derived, wrongly, at report time.
    attractor, first = cycle["steps"]["attractor"], cycle["steps"]["round0"]
    n_class2d = attractor["kept_particles"] + attractor["dropped_particles"]
    permanent_reject = n_class2d - attractor["kept_particles"] - first["kept_particles"]
    log(f"round {n} permanently rejected {permanent_reject}/{n_class2d} "
        f"({permanent_reject / n_class2d:.1%}); final survival "
        f"{final['kept_particles'] / n_class2d:.1%}")
    st.mark("select2d", select2d=final["uid"], kept_particles=final["kept_particles"],
            kept_classes=len(final["kept_classes"]), n_class2d=n_class2d,
            attractor_kept=attractor["kept_particles"], loop_kept=first["kept_particles"],
            permanent_reject=permanent_reject,
            permanent_reject_frac=round(permanent_reject / n_class2d, 5),
            final_survival_frac=round(final["kept_particles"] / n_class2d, 5),
            state=str(state_path))


def step_teacher(n, st, rd, ctx: Run):
    """Sample the teacher micrographs and write the surviving particles as labels."""
    if n >= ctx.last_round:
        st.mark("teacher", skipped="last round trains nothing")
        return
    manifest = json.loads(
        (paths.manifest_dir(ctx.empiar, entries.SETTING_ANNOT, ctx.source(n))
         / "manifest.json").read_text())
    if ctx.teacher_mics_from:
        # Training two arms on the same micrographs needs the list, not the seed: the
        # sampling pool differs per arm, so the same seed draws a different 50.
        mics_args = ["--mics-from", ctx.teacher_mics_from.replace("{round}", str(n))]
    elif ctx.teacher_mics != "all":
        mics_args = ["--num-mics", ctx.teacher_mics]
    else:
        mics_args = ["--all-mics"]
    run([paths.recon_python(), "-m", "rapick.loop.export_teacher_star",
         "--project", ctx.project, "--select2d", st.get("select2d", "select2d"),
         "--extract", manifest["jobs"]["extract"]["uid"], "--empiar", ctx.empiar,
         "--input-star", str(rd / CLEAN_STAR),
         "--seed", str(n + 1), *mics_args, "--out-dir", str(rd)],
        cwd=paths.REPO_ROOT, log_path=rd / "logs" / "teacher.log",
        env_extra=paths.recon_env())
    summary = json.loads((rd / "summary.json").read_text())
    marks = {k: summary[k] for k in
             ("n_teacher_particles", "particles_per_mic", "seed",
              "n_micrographs_with_survivors")}

    if ctx.arm.teacher == entries.TEACHER_GT:
        # The sampling above still decides WHICH micrographs are trained on -- that is
        # what is held fixed between the two rows of Table 7 -- but the labels on them
        # become the annotation. The counts recorded then describe the GT teacher, not
        # the survivors it replaced.
        gt = make_gt_teacher.build_teacher(ctx.empiar, rd / "train_mics.txt", rd)
        log(f"round {n} GT teacher: {gt['n_teacher_particles']} annotated particles "
            f"over {gt['n_micrographs_with_gt']} of {gt['n_micrographs_listed']} "
            f"micrographs")
        if gt["dropped_empty_gt_mics"]:
            log(f"round {n} {len(gt['dropped_empty_gt_mics'])} of those micrographs "
                f"carry no annotated particle and cannot appear in a star: "
                f"{', '.join(gt['dropped_empty_gt_mics'])}")
        marks = {"survivors_" + k: v for k, v in marks.items()}
        marks.update({k: gt[k] for k in
                      ("n_teacher_particles", "particles_per_mic",
                       "n_micrographs_with_gt", "dropped_empty_gt_mics")})

    st.mark("teacher", teacher_mics=ctx.teacher_mics, teacher=ctx.arm.teacher,
            teacher_star=str(ctx.teacher_star(rd)), **marks)


def step_finetune(n, st, rd, ctx: Run):
    """Fine-tune theta_0 on this round's teacher labels (Eq. 1)."""
    if n >= ctx.last_round:
        st.mark("finetune", skipped="last round trains nothing")
        return
    # The fine-tuner writes a ~914 MB checkpoint every epoch. On network storage that
    # alone takes 10-25 minutes per epoch against 1-2 minutes of computation, so point
    # RAPICK_WORK at a local disk for this stage, or run it against one and copy the
    # round directory back afterwards.
    out_dir = rd / "finetune"
    # theta_0 every round, never the checkpoint that just picked. Resuming from the
    # picking model instead would let the picker's own bias accumulate: it would be
    # trained on the particles it chose, having chosen them because it was trained on
    # them.
    resume = paths.base_checkpoint()
    wait_for_free_gpu(ctx.gpu, FT_MIN_FREE_MB, FT_MAX_WAIT_S)
    # --resume must be theta_0 and is always passed: the fine-tuner loads every weight
    # as-is and reinitialises nothing, and its own default points at the released
    # checkpoint, whose head is the degenerate one theta_0 exists to repair.
    run(paths.tool_cmd("finetune") +
        ["--images_dir", str(paths.annotated_micrographs(ctx.empiar)),
         "--star", str(ctx.teacher_star(rd)),
         "--box_size", str(ctx.entry.diameter_px),
         "--val_fraction", VAL_FRACTION,
         "--finetune_mode", ctx.arm.finetune_mode,
         "--resume", str(resume),
         "--device", f"cuda:{ctx.gpu}",
         "--output_dir", str(out_dir)],
        cwd=paths.tool_cwd("finetune"), log_path=rd / "logs" / "finetune.log",
        env_extra=paths.tool_env("finetune"))
    checkpoint = out_dir / "checkpoint.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"fine-tuning produced no checkpoint at {checkpoint}")
    stats = [json.loads(l) for l in (out_dir / "log.txt").read_text().splitlines()
             if l.strip()]
    st.mark("finetune", checkpoint=str(checkpoint), init=str(resume), arm=ctx.arm.name,
            teacher=ctx.arm.teacher, star=str(ctx.teacher_star(rd)),
            finetune_mode=ctx.arm.finetune_mode, epochs=len(stats),
            first_train_loss=stats[0].get("train_loss") if stats else None,
            last_train_loss=stats[-1].get("train_loss") if stats else None,
            last_val_loss=stats[-1].get("val_loss") if stats else None)


def step_promote(n, st, rd, ctx: Run):
    """Publish the fine-tuned checkpoint as the model round n+1 picks with."""
    if n >= ctx.last_round:
        st.mark("promote", skipped="last round trains nothing")
        return
    out = ctx.model(n + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(st.get("finetune", "checkpoint"), out)
    log(f"round {n} -> {out}")
    st.mark("promote", model=str(out))


HANDLERS = {"pick": step_pick, "score": step_score, "filter": step_filter,
            "class2d": step_class2d, "select2d": step_select2d,
            "teacher": step_teacher, "finetune": step_finetune,
            "promote": step_promote}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", dest="empiar", choices=sorted(entries.ENTRIES),
                    required=True, help="EMPIAR entry to run the loop on")
    ap.add_argument("--arm", choices=sorted(entries.ARMS), default=entries.DEFAULT_ARM,
                    help=f"which arm to run (default {entries.DEFAULT_ARM}, the paper's; "
                         f"see entries.ARMS for what the others are)")
    ap.add_argument("--rounds", default="0-2",
                    help='rounds to run, e.g. "0-2" (default), "2", "0,1"')
    ap.add_argument("--last-round", type=int, default=None,
                    help="the round that trains nothing (default: the last of --rounds). "
                         "Pass it explicitly when resuming a run in pieces, or a partial "
                         "--rounds would skip a fine-tune that is not actually last")
    ap.add_argument("--gpu", default=None, help="GPU index (default $RAPICK_GPU)")
    ap.add_argument("--worker", default=None,
                    help="CryoSPARC worker lane (default CRYOSPARC_WORKER from .env)")
    ap.add_argument("--project", default=None,
                    help="CryoSPARC project uid (default CRYOSPARC_PROJECT from .env)")
    ap.add_argument("--teacher", default=None,
                    choices=(entries.TEACHER_PICKS, entries.TEACHER_GT),
                    help="which labels the fine-tune trains on (default: the arm's own, "
                         "which for every arm but fb_gt is 'picks'). 'picks' is the "
                         "particles that survived this round's own cleanup; 'gt' is the "
                         "CryoPPP annotations of the same micrographs, the "
                         "perfect-teacher upper bound of Table 7. 'gt' switches to the "
                         "arm's GT counterpart (fb -> fb_gt) so it writes none of the "
                         "pseudo-label arm's outputs")
    ap.add_argument("--teacher-mics", default="50", choices=("50", "all"),
                    help='"50" (the paper\'s sample size per round) or "all" (every '
                         "micrograph with surviving particles that round)")
    ap.add_argument("--teacher-mics-from", default="", dest="teacher_mics_from",
                    help="train on exactly the micrographs listed in this file instead "
                         "of sampling; {round} in the path is replaced by the round "
                         "number. For running a second arm on the first arm's draw")
    ap.add_argument("--stop-after", choices=STEPS, help="run up to this step and stop")
    ap.add_argument("--redo", help="comma-separated steps to re-run even if recorded")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    rounds = parse_rounds(args.rounds)

    try:
        arm = entries.arm_for(args.arm, args.teacher)
    except ValueError as exc:
        sys.exit(str(exc))

    ctx = Run(entry=entries.ENTRIES[args.empiar],
              arm=arm,
              gpu=paths.gpu(args.gpu),
              worker=paths.cryosparc_worker(args.worker),
              project=paths.cryosparc_project(args.project),
              last_round=args.last_round if args.last_round is not None else max(rounds),
              teacher_mics=args.teacher_mics,
              teacher_mics_from=args.teacher_mics_from)

    # One loop per entry, whatever the arm (common.acquire_lock explains what an overlap
    # costs). Two entries are a different matter: they touch different workspaces,
    # sources and output roots, so they may run side by side on separate cards.
    lock = acquire_lock(LOCK_DIR / f"rapick_loop_{ctx.empiar}.lock",   # noqa: F841
                        f"rapick.loop.run_loop --id {ctx.empiar}")

    # Fail before the first GPU minute rather than at the step that needs each of these:
    # a missing mask store or images directory only shows up rounds in, and the fine-tune
    # step is an hour past the point where it could have been caught. The `images` entry
    # is the link the picker appends to its data root; without it the picking step reads
    # an empty set and every count below is silently zero.
    images = paths.picker_images(paths.annotated_data_root(), ctx.empiar)
    for label, path in (("masks", ctx.entry.mask_dir), ("images", images),
                        ("micrographs", ctx.entry.micrographs),
                        ("annotation", ctx.entry.gt_star)):
        if not Path(path).exists():
            hint = (f"  (the picker reads <data root>/<id>/images; link it to "
                    f"{ctx.entry.micrographs})" if label == "images" else "")
            sys.exit(f"{ctx.empiar} {label} missing: {path}{hint}")

    if not ctx.arm.in_paper:
        log(f"NOTE: arm {ctx.arm.name!r} is not reported in the paper -- {ctx.arm.note}")
    if ctx.arm.teacher == entries.TEACHER_GT:
        log("NOTE: --teacher gt is a perfect-teacher upper bound, not the feedback loop. "
            "It is reimplemented from a written procedure -- the scripts that produced "
            "the published Table 7 numbers were never committed -- so it has not been "
            "run end to end in this form.")
    log(f"id={ctx.empiar}  arm={ctx.arm.name}  mode={ctx.arm.finetune_mode}  "
        f"teacher={ctx.arm.teacher}  "
        f"root={entries.loop_root(ctx.empiar, ctx.arm.name)}  "
        f"sources={ctx.arm.source_prefix}N  "
        f"workspace={ctx.empiar}_{entries.SETTING_ANNOT}{ctx.arm.workspace_suffix}  gpu={ctx.gpu}")

    redo = set(args.redo.split(",")) if args.redo else set()
    for n in rounds:
        rd = entries.round_dir(ctx.empiar, n, ctx.arm.name)
        rd.mkdir(parents=True, exist_ok=True)
        st = State(rd / "state.json")
        log(f"===== round {n} =====")
        for step in STEPS:
            if st.done(step) and step not in redo:
                log(f"round {n} {step}: already done, skipping")
            else:
                HANDLERS[step](n, st, rd, ctx)
            if args.stop_after == step:
                log(f"--stop-after {step}: stopping")
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
