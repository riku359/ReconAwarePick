#!/usr/bin/env python3
"""Fine-tuning time per epoch, from the checkpoint files' modification times.

The gap between the `teacher` and `finetune` timestamps in a round's state file includes
waiting for a free GPU, which on some rounds runs past twenty hours. The per-epoch
checkpoints are written while training runs, so the difference between the first and the
last checkpoint's mtime, divided by the epochs between them, is a training rate with no
waiting in it.

Backs: the fine-tuning time quoted in the compute-cost paragraph of the supplementary
material.

Reads `$RAPICK_WORK/loop/<id>/round<n>/finetune/checkpoint<N>.pth`. No CryoSPARC
connection. Note that it reads mtimes: a tree that was copied without preserving them
gives nothing useful.

    python finetune_rate.py [--epochs 50]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

from rapick.loop import entries                        # noqa: E402


def hms(seconds):
    return "%d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=list(analysis_env.CORE_IDS))
    ap.add_argument("--arm", default=entries.DEFAULT_ARM, choices=sorted(entries.ARMS))
    ap.add_argument("--epochs", type=int, default=50,
                    help="the epoch count a full fine-tune runs, for the estimate")
    args = ap.parse_args()

    print(" %-6s %-7s %7s %10s %12s %12s" % ("id", "round", "ckpts", "span_epochs",
                                             "s_per_epoch", f"{args.epochs}ep_est"))
    rates = {}
    for empiar in args.ids:
        root = entries.loop_root(empiar, args.arm)
        if not root.is_dir():
            continue
        for round_dir in sorted(d for d in root.iterdir() if d.name.startswith("round")):
            checkpoints = []
            for path in (round_dir / "finetune").glob("checkpoint[0-9]*.pth"):
                match = re.search(r"checkpoint(\d+)\.pth$", path.name)
                if match:
                    checkpoints.append((int(match.group(1)), path.stat().st_mtime))
            if len(checkpoints) < 2:
                continue
            checkpoints.sort()
            span_epochs = checkpoints[-1][0] - checkpoints[0][0]
            span_seconds = checkpoints[-1][1] - checkpoints[0][1]
            if span_epochs <= 0:
                continue
            per_epoch = span_seconds / span_epochs
            rates.setdefault(empiar, []).append(per_epoch)
            print(" %-6s %-7s %7d %11d %12.1f %12s"
                  % (empiar, round_dir.name, len(checkpoints), span_epochs, per_epoch,
                     hms(per_epoch * args.epochs)))

    print("\n== median per entry, scaled to %d epochs ==" % args.epochs)
    for empiar, values in rates.items():
        values = sorted(values)
        median = values[len(values) // 2]
        print("  %s  %.1f s/epoch  -> %s for %d epochs (n=%d rounds)"
              % (empiar, median, hms(median * args.epochs), args.epochs, len(values)))


if __name__ == "__main__":
    main()
