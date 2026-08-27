#!/usr/bin/env python3
"""Fine-tune theta_0 on one round's teacher labels (paper Sec. 3.5, Eq. 1).

    theta_{n+1} = FineTune(theta_0; S_n)

The initialisation is theta_0 every time, never the checkpoint that just picked.
Resuming from the picking model instead would let the picker's own bias accumulate:
it would be trained on the particles it chose, having chosen them because it was
trained on them. TranSPHIRE's own implementation does the same -- its `--weights_old`
is assigned once at session start and never advances.

This is one step of `run_loop.py`, which calls `finetune()` below rather than
duplicating it. Run standalone when you have a 2D selection already and want the
checkpoint it implies, without driving a whole round:

  python -m rapick.loop.finetune --id 10081 \\
      --star  $RAPICK_WORK/loop/10081/round0/teacher.star \\
      --out-dir $RAPICK_WORK/loop/10081/round0/finetune

Writes `checkpoint.pth` and `log.txt` into --out-dir, and prints the checkpoint path
as `CHECKPOINT=<path>` so a caller can read it out of the log.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from . import entries, paths
from .common import wait_for_free_gpu

# The 40/10 split of the 50 teacher micrographs, as a fraction handed to finetune.py.
# The 10 validation micrographs monitor the loss; they select nothing.
VAL_FRACTION = "0.2"

# The fine-tuner writes a ~914 MB checkpoint every epoch, so it waits for a card with
# room rather than dying part-way through one.
FT_MIN_FREE_MB = int(os.environ.get("RAPICK_FT_MIN_FREE_MB", "20000"))
FT_MAX_WAIT_S = int(os.environ.get("RAPICK_FT_MAX_WAIT_S", "7200"))


def finetune(empiar: str, teacher_star: Path, out_dir: Path, gpu: int,
             finetune_mode: str = entries.ARMS[entries.DEFAULT_ARM].finetune_mode,
             resume: Optional[Path] = None, log_path: Optional[Path] = None,
             runner=None) -> dict:
    """Run one fine-tune and return what it produced.

    `runner` is the subprocess helper to call (default `common.run`); run_loop passes
    its own so a round's output lands in that round's log directory.

    Returns {"checkpoint", "init", "epochs", "first_train_loss", "last_train_loss",
    "last_val_loss"} -- the fields the loop records in its per-round state.
    """
    from .common import run as default_run

    runner = runner or default_run
    entry = entries.ENTRIES[str(empiar)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # theta_0 unless the caller insists otherwise. --resume is always passed: the
    # fine-tuner loads every weight as-is and reinitialises nothing, and its own default
    # points at the released checkpoint, whose head is the degenerate one theta_0 exists
    # to repair.
    resume = Path(resume) if resume else paths.base_checkpoint()

    wait_for_free_gpu(gpu, FT_MIN_FREE_MB, FT_MAX_WAIT_S)
    runner(paths.tool_cmd("finetune") +
           ["--images_dir", str(paths.annotated_micrographs(entry.empiar)),
            "--star", str(teacher_star),
            "--box_size", str(entry.diameter_px),
            "--val_fraction", VAL_FRACTION,
            "--finetune_mode", finetune_mode,
            "--resume", str(resume),
            "--device", f"cuda:{gpu}",
            "--output_dir", str(out_dir)],
           cwd=paths.tool_cwd("finetune"),
           log_path=log_path or (out_dir / "finetune.log"),
           env_extra=paths.tool_env("finetune"))

    checkpoint = out_dir / "checkpoint.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"fine-tuning produced no checkpoint at {checkpoint}")
    stats = [json.loads(l) for l in (out_dir / "log.txt").read_text().splitlines()
             if l.strip()]
    return {"checkpoint": str(checkpoint), "init": str(resume),
            "finetune_mode": finetune_mode, "epochs": len(stats),
            "first_train_loss": stats[0].get("train_loss") if stats else None,
            "last_train_loss": stats[-1].get("train_loss") if stats else None,
            "last_val_loss": stats[-1].get("val_loss") if stats else None}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="EMPIAR entry, e.g. 10081")
    ap.add_argument("--star", required=True,
                    help="the teacher labels: surviving particles on the sampled "
                         "micrographs (rapick.loop.export_teacher_star writes them)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--finetune-mode",
                    default=entries.ARMS[entries.DEFAULT_ARM].finetune_mode,
                    help="which weights to update (default: the paper's)")
    ap.add_argument("--resume", help="initialisation (default: theta_0). The loop never "
                                     "changes this; see the module docstring for why.")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    result = finetune(args.id, Path(args.star), Path(args.out_dir), args.gpu,
                      finetune_mode=args.finetune_mode, resume=args.resume)
    for key in ("init", "finetune_mode", "epochs", "first_train_loss",
                "last_train_loss", "last_val_loss"):
        print(f"  {key:18s} {result[key]}")
    print(f"CHECKPOINT={result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
