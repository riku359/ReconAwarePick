#!/usr/bin/env python3
"""Re-pick a whole deposition with a loop checkpoint, and publish it as a condition's picks.

The loop trains on 300 micrographs and reconstructs nothing: at that scale the
seed-to-seed spread of GSFSC 0.143 is the size of the effect being looked for. Every
reconstruction-level number in the paper comes from the full deposition instead, and this
is the step between the two -- the only part of that path that needs to know about a
checkpoint the loop produced.

  1. pick the whole deposition with the checkpoint, at the loop's operating point
  2. discard the picks whose centre lands on contamination, per the stored masks
  3. publish the result as $RAPICK_WORK/picks/<id>/<condition>.star

Step 3 is the handoff. That path is what the dataset configs name, so once it exists the
reconstruction is `rapick.recon`'s ordinary business under `configs/conditions/fb.yaml` --
nothing about it is specific to a checkpoint any more. The command is printed at the end.

  python -m rapick.loop.repick_fullset --id 10081 \\
      --model "$RAPICK_WORK/loop/10081/models/model_1.pth"

Every step records itself in state.json and is skipped when already done, so re-running
resumes rather than rebuilds -- a deposition of 1,000 to 1,900 micrographs is a long pick.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

from . import entries, paths, star
from .common import LOCK_DIR, State, acquire_lock, log, run
from .run_loop import CLEAN_STAR, FILTER_SUMMARY, MASK_SUFFIX, PICK_ARGS, STAR_PREFIX

STEPS = ("pick", "filter", "publish")

# The condition whose picks a loop checkpoint produces. It is the paper's method, and the
# dataset configs declare its STAR at $RAPICK_WORK/picks/<id>/fb.star.
DEFAULT_CONDITION = "fb"


def work_dir(empiar: str, condition: str) -> Path:
    return entries.fullset_dir(empiar, condition)


def lock_path(empiar: str, condition: str) -> Path:
    return LOCK_DIR / f"rapick_loop_repick_{empiar}_{condition}.lock"


def warn_if_near_cap(entry: entries.Entry, n_particles: int, what: str) -> None:
    """Say so when a stack is about to be clamped by ab-initio reconstruction.

    Two conditions are comparable on particle count only when neither is clamped; a
    comparison straddling the cap is confounded, and the clamp is invisible unless
    somebody looks for it.
    """
    if entry.abinit_cap is None:
        return
    if n_particles >= entry.abinit_cap:
        log(f"WARNING {what} = {n_particles:,} is at or above the ab-initio "
            f"{entry.abinit_cap:,} cap for this entry: it will clamp, and this condition "
            f"loses the particle-count axis of the comparison")
    elif n_particles >= 0.95 * entry.abinit_cap:
        log(f"NOTE {what} = {n_particles:,} is within 5% of the ab-initio "
            f"{entry.abinit_cap:,} cap; the count entering it decides whether it clamps")


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def step_pick(st, td, args, entry):
    """Pick the whole deposition with the checkpoint under test."""
    ckpt = Path(args.model)
    if not ckpt.is_file():
        raise RuntimeError(f"no checkpoint at {ckpt}")

    data_root = paths.fullset_data_root()
    images = paths.picker_images(data_root, entry.empiar)
    if not images.exists():
        raise RuntimeError(
            f"{entry.empiar} images missing: {images} (the picker reads "
            f"<data root>/<id>/images; link it to "
            f"{paths.fullset_micrographs(entry.empiar)})")

    run(paths.tool_cmd("predict_fullset") +
        ["--empiar", entry.empiar, "--data_root", str(data_root),
         "--resume", str(ckpt), *PICK_ARGS,
         "--device", f"cuda:{args.gpu}", "--remarks", args.condition],
        cwd=paths.tool_cwd("predict_fullset"), log_path=td / "logs" / "pick.log",
        env_extra=paths.tool_env("predict_fullset"))

    pred_root = paths.tool_cwd("predict_fullset") / "output" / "predictions"
    candidates = sorted(
        pred_root.glob(f"predictions_EMPIAR_{entry.empiar}_remarks_{args.condition}"
                       f"_timestamp_*"),
        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"no prediction dir for {args.condition} under {pred_root}")
    produced = (candidates[-1] /
                f"EMPIAR_{entry.empiar}_remarks_{args.condition}_star_file.star")
    if not produced.is_file():
        raise RuntimeError(f"combined star missing: {produced}")
    picks = td / "picks.star"
    shutil.copyfile(produced, picks)

    # An inference that died partway still leaves a valid STAR of the micrographs it did
    # reach, and every count downstream would then be quietly short.
    mics = star.micrograph_names(picks)
    n_picks = star.count_star_particles(picks)
    if len(mics) != entry.fullset_micrographs:
        raise RuntimeError(f"{picks} covers {len(mics)} micrographs, expected "
                           f"{entry.fullset_micrographs}: the inference did not finish "
                           f"the set")
    log(f"pick: {n_picks:,} picks over {len(mics)} micrographs "
        f"({n_picks / len(mics):.1f} per micrograph)")
    warn_if_near_cap(entry, n_picks, "picks before the contamination filter")
    st.mark("pick", checkpoint=str(ckpt), prediction_dir=str(candidates[-1]),
            picks_star=str(picks), n_picks=n_picks, n_micrographs=len(mics))


def step_filter(st, td, args, entry):
    """Drop the picks whose centre lands on contamination, per the stored masks.

    Applying the saved masks needs neither TensorFlow, nor a GPU, nor the raw
    micrographs; the decision is identical to recomputing them up to float16 rounding
    (~0.0005 near the 0.5 threshold), which is why the masks can be reused across every
    checkpoint that re-picks the same micrographs.
    """
    mask_dir = paths.mask_dir(entry.empiar)
    if not mask_dir.is_dir():
        raise RuntimeError(f"no stored masks for {entry.empiar} at {mask_dir}")
    run(paths.tool_cmd("mask_filter") +
        ["--star", str(td / "picks.star"), "--mask-dir", str(mask_dir),
         "--empiar-id", entry.empiar, "--out-dir", str(td),
         "--star-prefix", STAR_PREFIX, "--suffix", MASK_SUFFIX, "--overwrite"],
        cwd=paths.tool_cwd("mask_filter"), log_path=td / "logs" / "filter.log")
    summary = json.loads((td / FILTER_SUMMARY).read_text())
    log(f"filter: kept {summary['picks_kept']:,} of {summary['picks_total']:,} "
        f"({summary['removed_fraction']:.2%} removed)")
    warn_if_near_cap(entry, summary["picks_kept"], "picks after the contamination filter")
    st.mark("filter", **{k: summary[k] for k in
                         ("picks_total", "picks_kept", "picks_removed", "removed_fraction",
                          "n_micrographs_anomaly", "n_micrographs_error")})


def step_publish(st, td, args, entry):
    """Copy the filtered picks to where the dataset configs say this condition's live."""
    out = paths.picks_star(entry.empiar, args.condition)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(td / CLEAN_STAR, out)
    n = star.count_star_particles(out)
    log(f"publish: {n:,} picks -> {out}")
    st.mark("publish", star=str(out), n_picks=n,
            checkpoint=st.get("pick", "checkpoint"))

    dataset = paths.REPO_ROOT / "configs" / "datasets" / f"empiar_{entry.empiar}.yaml"
    condition = paths.REPO_ROOT / "configs" / "conditions" / f"{args.condition}.yaml"
    log("")
    log("Reconstruct it with the reconstruction stage; nothing below is specific to the")
    log("checkpoint any more:")
    log("")
    log(f"  PYTHONPATH=src {paths.recon_python()} -m rapick.recon.cli run \\")
    log(f"      --condition {condition} \\")
    log(f"      --dataset {dataset} \\")
    log(f"      --setting {entries.SETTING_FULL} --seeds 0,1,2")
    log("")
    log("  # then build the 2D class selection over that class_2D with")
    log("  # src/rapick/select2d/, reconstruct from its final select_2D, and collect.")
    log("")
    log("Best-of-3, and say which seeds: when an ab-initio job dies with a SIGSEGV, retry")
    log("the same seed at most twice and then advance the seed number (--seeds 0,1,3).")
    log("Completed jobs are reused, so a retry resumes the trio rather than restarting it,")
    log("and the seeds actually used must be reported with the resolution -- a best-of-2")
    log("is not a best-of-3.")


HANDLERS = {"pick": step_pick, "filter": step_filter, "publish": step_publish}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, choices=sorted(entries.ENTRIES), dest="empiar",
                    help="EMPIAR entry whose full deposition to re-pick")
    ap.add_argument("--model", required=True,
                    help="the checkpoint to pick with; the paper's is round 1's, "
                         "$RAPICK_WORK/loop/<id>/models/model_1.pth")
    ap.add_argument("--condition", default=DEFAULT_CONDITION,
                    help=f"which condition these picks are (default {DEFAULT_CONDITION}, "
                         f"the paper's method). It names both the published STAR and the "
                         f"working directory")
    ap.add_argument("--gpu", default=None, help="GPU index (default $RAPICK_GPU)")
    ap.add_argument("--stop-after", choices=STEPS, help="run up to this step and stop")
    ap.add_argument("--redo", help="comma-separated steps to re-run even if recorded")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    entry = entries.ENTRIES[args.empiar]
    args.gpu = paths.gpu(args.gpu)

    lock = acquire_lock(lock_path(args.empiar, args.condition),   # noqa: F841
                        f"rapick.loop.repick_fullset --id {args.empiar}")

    td = work_dir(entry.empiar, args.condition)
    td.mkdir(parents=True, exist_ok=True)
    st = State(td / "state.json")

    # A condition names one checkpoint's picks. Pointing it at a second checkpoint would
    # leave the recorded picks, the published STAR and every job built on it describing
    # the first.
    recorded = st.get("pick", "checkpoint")
    if recorded and Path(recorded) != Path(args.model):
        sys.exit(f"{td}/state.json records checkpoint {recorded}, not {args.model}: "
                 f"--condition {args.condition} already names another model's picks. "
                 f"Use --redo pick if you mean to replace them.")

    log(f"id={entry.empiar}  condition={args.condition}  model={args.model}  "
        f"gpu={args.gpu}")
    log(f"out={td}  publishes to {paths.picks_star(entry.empiar, args.condition)}")

    redo = set(args.redo.split(",")) if args.redo else set()
    for step in STEPS:
        if st.done(step) and step not in redo:
            log(f"{step}: already done, skipping")
        else:
            HANDLERS[step](st, td, args, entry)
        if args.stop_after == step:
            log(f"--stop-after {step}: stopping")
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
