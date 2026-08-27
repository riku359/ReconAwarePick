#!/usr/bin/env python3
"""Print how far each arm of the loop has got, and the 2D diagnostics per round.

The loop measures the picker with GT-free, seed-free numbers rather than a
reconstruction, so "how is it going" is answered by this table and not by a resolution.
It reads only state.json, so it is safe to run against a live loop.

  python -m rapick.loop.status --id 10532
  python -m rapick.loop.status --id 10532 --arm fb
"""
from __future__ import annotations

import argparse

from . import entries
from .round_metrics import rows_for
from .run_loop import STEPS


def pct(frac):
    """A fraction as a percentage, or "-" when the round did not record it."""
    return "-" if frac is None else f"{frac * 100:.1f}%"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", dest="empiar", choices=sorted(entries.ENTRIES), required=True)
    ap.add_argument("--arm", choices=sorted(entries.ARMS) + ["all"], default="all")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    arms = sorted(entries.ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        root = entries.loop_root(args.empiar, arm)
        print(f"\n=== {args.empiar} / {arm} ({root}) ===")
        if not root.is_dir():
            print("  not started")
            continue
        print(f"  {'rd':>2} {'picks':>8} {'F1':>6} {'kept':>8} {'class2d':>8} "
              f"{'select2d':>8} {'reject%':>8} {'surv%':>8}  next")
        for n, d in rows_for(root):
            score, filt = d.get("score", {}), d.get("filter", {})
            class2d, selection = d.get("class2d", {}), d.get("select2d", {})
            todo = [s for s in STEPS if s not in d]
            print(f"  {n:>2} {score.get('n_pred_eval', ''):>8} "
                  f"{score.get('macro_F1', 0):>6.3f} {filt.get('picks_kept', ''):>8} "
                  f"{class2d.get('uid', ''):>8} {selection.get('select2d', ''):>8} "
                  f"{pct(selection.get('permanent_reject_frac')):>8} "
                  f"{pct(selection.get('final_survival_frac')):>8}  "
                  f"{todo[0] if todo else 'done'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
