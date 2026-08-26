#!/usr/bin/env python3
"""count_gt_in_contamination.py -- aggregate how many annotated particles fell inside
contamination.

It aggregates the per-micrograph `n_gt` / `n_gt_in_contam` recorded in the manifest (=
how many ground-truth particle coordinates fell inside mask>=deep_thr, decided at full
resolution) by EMPIAR id and by distribution class, and appends the result to a markdown
file.

  <out-root>/manifest*.csv  ->  a <!-- GT-CONTAM --> section appended to <md>
                                (default: <out-root>/summary.md)

It answers whether "annotators pick particles away from contamination", and whether that
differs in and out of distribution, by the percentage of ground-truth particles that
landed on contamination. The ground truth concentrates on positive (anomaly) micrographs,
so both an all-micrograph and an anomaly-only version are reported.

The manifest is written by the contamination-detection driver of the research repository,
which is not part of this release; its columns are listed in this directory's README.
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env  # noqa: E402

BEGIN = "<!-- GT-CONTAM:BEGIN -->"
END = "<!-- GT-CONTAM:END -->"


def load_rows(out_root):
    shards = sorted(glob.glob(os.path.join(out_root, "manifest_gpu*.csv")))
    if not shards:
        single = os.path.join(out_root, "manifest.csv")
        shards = [single] if os.path.isfile(single) else []
    rows = []
    for s in shards:
        with open(s, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Tally:
    """A container for the ground-truth containment counts (all micrographs and anomaly-only)."""

    def __init__(self):
        self.gt = self.inc = 0          # all micrographs
        self.gt_pos = self.inc_pos = 0  # anomaly micrographs only
        self.mics = self.mics_gt = 0

    def add(self, n_gt, n_in, is_anom):
        self.mics += 1
        if n_gt > 0:
            self.mics_gt += 1
        self.gt += n_gt; self.inc += n_in
        if is_anom:
            self.gt_pos += n_gt; self.inc_pos += n_in


def pct(a, b):
    return "%.2f%%" % (100.0 * a / b) if b else "-"


def build_report(rows):
    per_id, per_dist = {}, {}
    id_dist = {}
    for r in rows:
        if r.get("status") not in ("ok", "anomaly"):
            continue
        n_gt, n_in = as_int(r.get("n_gt")), as_int(r.get("n_gt_in_contam"))
        if n_gt is None or n_in is None:
            continue
        eid, dist = r["empiar_id"], r["dist_class"]
        id_dist[eid] = dist
        is_anom = r.get("status") == "anomaly"
        per_id.setdefault(eid, Tally()).add(n_gt, n_in, is_anom)
        per_dist.setdefault(dist, Tally()).add(n_gt, n_in, is_anom)

    lines = [BEGIN, "", "## Ground-truth particles inside contamination", "",
             "How many ground-truth particle coordinates (CryoPPP selected.star) fell "
             "inside a contamination region (`mask>=deep_thr`), decided at full resolution. "
             "`in/all` is over all micrographs, `in/pos` only over the micrographs "
             "classified as anomaly.", ""]

    for dist in ("in_distribution", "out_of_distribution"):
        t = per_dist.get(dist)
        if not t:
            continue
        lines.append(
            "- **%s**: %d of %d ground-truth particles inside contamination (%s) "
            "/ restricted to anomaly micrographs, %d of %d (%s)"
            % (dist, t.inc, t.gt, pct(t.inc, t.gt),
               t.inc_pos, t.gt_pos, pct(t.inc_pos, t.gt_pos)))
    lines += ["", "| EMPIAR | dist | GT total | in-contam | in/all | GT(pos mics) | in/pos |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for eid in sorted(per_id):
        t = per_id[eid]
        short = "IN" if id_dist[eid] == "in_distribution" else "OOD"
        lines.append("| %s | %s | %d | %d | %s | %d | %s |"
                     % (eid, short, t.gt, t.inc, pct(t.inc, t.gt),
                        t.gt_pos, pct(t.inc_pos, t.gt_pos)))
    lines += ["", END, ""]
    return "\n".join(lines)


def upsert_section(md_path, section):
    """Append the GT-CONTAM section to the markdown (replacing it when it is already there)."""
    old = ""
    if os.path.isfile(md_path):
        with open(md_path) as f:
            old = f.read()
    if BEGIN in old and END in old:
        pre = old[:old.index(BEGIN)].rstrip("\n")
        post = old[old.index(END) + len(END):].lstrip("\n")
        new = (pre + "\n\n" + section + "\n" + post).rstrip("\n") + "\n"
    else:
        new = old.rstrip("\n") + ("\n\n" if old.strip() else "") + section + "\n"
    with open(md_path, "w") as f:
        f.write(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=str, default=None,
                    help="directory holding the manifest CSV (default: $RAPICK_WORK)")
    ap.add_argument("--md", type=str, default=None,
                    help="markdown to append to (default: <out-root>/summary.md)")
    args = ap.parse_args()

    out_root = args.out_root or str(env.work_root())
    rows = load_rows(out_root)
    if not rows:
        sys.exit("no manifest found: %s" % out_root)
    section = build_report(rows)
    md_path = args.md or os.path.join(out_root, "summary.md")
    upsert_section(md_path, section)
    print(section)
    print("appended -> %s" % md_path)


if __name__ == "__main__":
    main()
