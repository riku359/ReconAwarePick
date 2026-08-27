#!/usr/bin/env python3
"""Tabulate P/R/F1 and the pick-count funnel for every round of the loop (Table 6).

`status.py` answers "how far has this arm got"; this answers "what are the numbers",
across every entry and arm at once, straight from each round's state.json. It needs no
GPU and no CryoSPARC connection, so it is safe to run against a live loop.

Reading it: `raw` is what the picker proposed (filter.picks_total), `filter` is what
survived the contamination filter (filter.picks_kept), `class2d` is what class_2D
accepted (select2d.n_class2d), `select2d` is what the iterative selection kept for the
teacher (select2d.kept_particles). P/R/F1 score the round's *raw* picks against the
CryoPPP annotation at the loop's operating point -- the funnel columns are downstream of
that score, not an input to it. Of the 300 scored micrographs, the 50 the round trained
on are included; the loop specialises to one entry rather than generalising, so that
overlap is the mechanism and not a leak, and it is stated wherever these numbers appear.

  python -m rapick.loop.round_metrics
  python -m rapick.loop.round_metrics --id 10081 --id 10093
  python -m rapick.loop.round_metrics --csv results/tables/table6.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import entries

COLUMNS = ("id", "arm", "round", "n_gt", "raw_picks", "filter_kept", "class2d_particles",
           "select2d_kept", "select2d_reject_frac", "macro_P", "macro_R", "macro_F1",
           "checkpoint")


def rows_for(root: Path):
    """Every round dir under root that has a state.json, in order, however many exist."""
    n = 0
    while (root / f"round{n}" / "state.json").is_file():
        yield n, json.loads((root / f"round{n}" / "state.json").read_text())
        n += 1


def funnel_row(empiar: str, arm: str, n: int, d: dict) -> dict:
    score, filt = d.get("score", {}), d.get("filter", {})
    selection, pick = d.get("select2d", {}), d.get("pick", {})
    return {
        "id": empiar, "arm": arm, "round": n,
        "n_gt": score.get("n_gt"),
        "raw_picks": filt.get("picks_total", score.get("n_pred_eval")),
        "filter_kept": filt.get("picks_kept"),
        "class2d_particles": selection.get("n_class2d"),
        "select2d_kept": selection.get("kept_particles"),
        "select2d_reject_frac": selection.get("permanent_reject_frac"),
        "macro_P": score.get("macro_P"),
        "macro_R": score.get("macro_R"),
        "macro_F1": score.get("macro_F1"),
        "checkpoint": Path(pick["checkpoint"]).name if pick.get("checkpoint") else None,
    }


def fmt(row: dict) -> str:
    def f3(v):
        return "" if v is None else f"{v:.3f}"

    def pct(v):
        return "" if v is None else f"{v * 100:.1f}%"

    return (f"{row['id']:>7} {row['arm']:>10} {row['round']:>3} "
            f"{row['n_gt'] or '':>7} {row['raw_picks'] or '':>8} "
            f"{row['filter_kept'] or '':>8} {row['class2d_particles'] or '':>9} "
            f"{row['select2d_kept'] or '':>9} {pct(row['select2d_reject_frac']):>7} "
            f"{f3(row['macro_P']):>6} {f3(row['macro_R']):>6} {f3(row['macro_F1']):>6}  "
            f"{row['checkpoint'] or ''}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", action="append", dest="ids", choices=sorted(entries.ENTRIES),
                    default=None, help="repeatable; default = every entry")
    ap.add_argument("--arm", action="append", dest="arms", choices=sorted(entries.ARMS),
                    default=None, help="repeatable; default = every arm")
    ap.add_argument("--csv", default=None, help="also write the table here")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ids = args.ids or sorted(entries.ENTRIES)
    arms = args.arms or sorted(entries.ARMS)

    print(f"{'id':>7} {'arm':>10} {'rd':>3} {'n_gt':>7} {'raw':>8} {'filter':>8} "
          f"{'class2d':>9} {'select2d':>9} {'rej%':>7} {'P':>6} {'R':>6} {'F1':>6}  "
          f"checkpoint")

    all_rows = []
    for empiar in ids:
        for arm in arms:
            root = entries.loop_root(empiar, arm)
            if not root.is_dir():
                continue
            for n, d in rows_for(root):
                row = funnel_row(empiar, arm, n, d)
                all_rows.append(row)
                print(fmt(row))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\ncsv -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
