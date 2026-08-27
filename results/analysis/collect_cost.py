#!/usr/bin/env python3
"""Compute cost, collected from logs that already exist. Nothing is re-timed.

Three sources, all of them read-only:

  1. `$RAPICK_WORK/loop/<id>/round<n>/state.json` -- the completion time of each stage of
     one loop round on the 300 annotated micrographs. `pick` has no start time in the
     state file, so it is taken from the timestamp in the prediction directory's name.
  2. the full-set chain logs -- the completion time of each stage of one full-set arm,
     read off the `[HH:MM:SS]` at the start of each line. A run crossing midnight is
     carried forward by 24 h so the sequence stays monotonic.
  3. `<project>/J<n>/job.json` -- `started_at` / `completed_at` of every CryoSPARC job of
     the reconstruction chain, plus the instance's hardware description.

Backs: the compute-cost paragraph of the supplementary material.

    python collect_cost.py --project-dir <dir> --logs <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_env                                    # noqa: E402

from rapick.loop import entries                        # noqa: E402

N_SUBSET = 300

# The stage markers in a full-set chain log, in the order they appear.
STAGE_PATTERNS = [("pick", r"pick: "), ("filter", r"filter: "),
                  ("class2d", r"clean class_2D"),
                  ("cryosift", r"cryosift final select_2D"),
                  ("recon", r"cleaned\+selected best-of-3")]

# The refine job of the `fb` condition on each entry, used only to read the instance's
# hardware description out of one job.json. Any completed job would do.
HARDWARE_JOBS = ("J507", "J577", "J511", "J557")

LOOP_STAGES = ("pick", "score", "filter", "class2d", "cryosift", "teacher", "finetune")


def parse_time(text):
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def loop_rounds(ids, arm):
    """Per-stage seconds of each loop round, from the round state files."""
    out = {}
    for empiar in ids:
        root = entries.loop_root(empiar, arm)
        if not root.is_dir():
            continue
        for round_dir in sorted(d for d in root.iterdir() if d.name.startswith("round")):
            state_file = round_dir / "state.json"
            if not state_file.exists():
                continue
            state = json.loads(state_file.read_text())
            if not all(k in state and state[k].get("at") for k in LOOP_STAGES):
                continue
            at = {k: parse_time(state[k]["at"]) for k in LOOP_STAGES}
            prediction_dir = state["pick"].get("prediction_dir") or ""
            match = re.search(r"timestamp_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
                              prediction_dir)
            pick_s = (at["pick"] - parse_time(match.group(1))).total_seconds() if match else None
            out.setdefault(empiar, {})[round_dir.name] = {
                "pick_s": pick_s,
                "filter_s": (at["filter"] - at["score"]).total_seconds(),
                "class2d_s": (at["class2d"] - at["filter"]).total_seconds(),
                "cryosift_s": (at["cryosift"] - at["class2d"]).total_seconds(),
                "finetune_s": (at["finetune"] - at["teacher"]).total_seconds(),
                "n_picks": state["filter"].get("picks_total"),
            }
    return out


def fullset_chain(ids, logs: Path):
    """Per-stage seconds of one full-set arm, from its chain log."""
    out = {}
    if logs is None:
        return out
    for empiar in ids:
        candidates = sorted(logs.glob(f"chain_{empiar}_*.log"))
        if not candidates:
            continue
        marks, start = {}, None
        for line in candidates[0].read_text().splitlines():
            match = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\] (.*)", line)
            if not match:
                continue
            seconds = (int(match.group(1)) * 3600 + int(match.group(2)) * 60
                       + int(match.group(3)))
            if start is None:
                start = seconds
            body = match.group(4)
            for name, pattern in STAGE_PATTERNS:
                if re.match(pattern, body) and name not in marks:
                    marks[name] = seconds
        if len(marks) < len(STAGE_PATTERNS):
            continue
        sequence = [("pick", start)] + [(n, marks[n]) for n, _ in STAGE_PATTERNS]
        # Carry a midnight crossing forward, so the sequence stays monotonic.
        fixed, add, previous = [], 0, sequence[0][1]
        for name, value in sequence:
            if value < previous:
                add += 86400
            previous = value
            fixed.append((name, value + add))
        at = dict(fixed)
        out[empiar] = {
            "n_micrographs": entries.ENTRIES[empiar].fullset_micrographs,
            "pick_s": at["pick"] - fixed[0][1],
            "filter_s": at["filter"] - at["pick"],
            "class2d_s": at["class2d"] - at["filter"],
            "cryosift_s": at["cryosift"] - at["class2d"],
            "recon_s": at["recon"] - at["cryosift"],
        }
    return out


def cryosparc_jobs(ids, project: Path, setting: str, condition: str):
    """Wall-clock seconds of each job of one condition's reconstruction chain."""
    out = {}
    for empiar in ids:
        metrics_file = analysis_env.manifest_dir(empiar, setting, condition) / "metrics.json"
        if not metrics_file.exists():
            continue
        metrics = json.loads(metrics_file.read_text())
        uids = set()
        for trial in metrics.get("trials", []):
            uids.update(v for k, v in trial.items()
                        if isinstance(v, str) and v.startswith("J"))
        uids.add(metrics["maps"]["refine"]["uid"])
        rows = []
        for uid in sorted(uids, key=lambda u: int(u[1:])):
            job_file = project / uid / "job.json"
            if not job_file.exists():
                continue
            job = json.loads(job_file.read_text())
            started, completed = job.get("started_at"), job.get("completed_at")
            if not (started and completed):
                continue
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
            seconds = (dt.datetime.strptime(completed["$date"], fmt)
                       - dt.datetime.strptime(started["$date"], fmt)).total_seconds()
            rows.append({"uid": uid, "type": job.get("type"), "seconds": seconds})
        out[empiar] = rows
    return out


def hardware(project: Path, job_uids):
    """The instance description CryoSPARC records on a job."""
    for uid in job_uids:
        job_file = project / uid / "job.json"
        if job_file.exists():
            info = json.loads(job_file.read_text()).get("instance_information") or {}
            gpus = info.get("gpu_info") or []
            return {"cpu_model": info.get("cpu_model"),
                    "physical_cores": info.get("physical_cores"),
                    "total_memory": info.get("total_memory"),
                    "cuda": info.get("CUDA_version"), "driver": info.get("driver_version"),
                    "n_gpus": len(gpus),
                    "gpu_name": gpus[0].get("name") if gpus else None,
                    "gpu_mem_bytes": gpus[0].get("mem") if gpus else None}
    return {}


def hms(seconds):
    if seconds is None:
        return "--"
    return "%d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", default=None)
    ap.add_argument("--ids", nargs="+", default=list(analysis_env.CORE_IDS))
    ap.add_argument("--arm", default=entries.DEFAULT_ARM, choices=sorted(entries.ARMS))
    ap.add_argument("--logs", type=Path, default=None,
                    help="directory of full-set chain logs (chain_<id>_*.log)")
    ap.add_argument("--setting", default="full")
    ap.add_argument("--condition", default="fb",
                    help="which reconstruction arm's CryoSPARC jobs to time")
    ap.add_argument("--hardware-job", nargs="+", default=list(HARDWARE_JOBS),
                    help="any completed job whose job.json carries the instance "
                         "description; the first one that exists is used")
    ap.add_argument("--out", default=None,
                    help="default $RAPICK_WORK/analysis/cost.json")
    args = ap.parse_args()

    project = analysis_env.project_dir(args.project_dir)
    result = {
        "loop_rounds": loop_rounds(args.ids, args.arm),
        "fullset_chain": fullset_chain(args.ids, args.logs),
        "cryosparc_jobs": cryosparc_jobs(args.ids, project, args.setting, args.condition),
        "hardware": hardware(project, args.hardware_job),
    }
    out = analysis_env.out_path("cost.json", args.out)
    out.write_text(json.dumps(result, indent=1))

    print("== hardware ==")
    print(" ", result["hardware"])

    print("\n== full set, one arm ==")
    print(" %-6s %6s %9s %8s %9s %9s %9s" % ("id", "mics", "pick", "filter", "class2d",
                                             "cryosift", "recon"))
    for empiar, row in result["fullset_chain"].items():
        print(" %-6s %6d %9s %8s %9s %9s %9s"
              % (empiar, row["n_micrographs"], hms(row["pick_s"]), hms(row["filter_s"]),
                 hms(row["class2d_s"]), hms(row["cryosift_s"]), hms(row["recon_s"])))
        print("        pick %.2f s/micrograph" % (row["pick_s"] / row["n_micrographs"]))

    print("\n== loop, one round (%d micrographs) ==" % N_SUBSET)
    print(" %-6s %-7s %8s %8s %9s %9s %9s" % ("id", "round", "pick", "filter", "class2d",
                                              "cryosift", "finetune"))
    for empiar, rounds in result["loop_rounds"].items():
        for name, row in rounds.items():
            print(" %-6s %-7s %8s %8s %9s %9s %9s"
                  % (empiar, name, hms(row["pick_s"]), hms(row["filter_s"]),
                     hms(row["class2d_s"]), hms(row["cryosift_s"]), hms(row["finetune_s"])))

    print("\n== reconstruction chain, per CryoSPARC job (%s) ==" % args.condition)
    for empiar, rows in result["cryosparc_jobs"].items():
        total = sum(r["seconds"] for r in rows)
        kinds = {}
        for row in rows:
            kinds.setdefault(row["type"], []).append(row["seconds"])
        print(" %s total %s :" % (empiar, hms(total)),
              ", ".join("%s x%d avg %s" % (k, len(v), hms(sum(v) / len(v)))
                        for k, v in kinds.items()))
    print("\ndone ->", out)


if __name__ == "__main__":
    main()
