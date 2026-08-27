#!/usr/bin/env python3
"""Export each round's extracted, accepted and surviving particles as GT-aligned STAR.

The loop already leaves the picker's own two stages on disk (`picks.star` and the
contamination filter's `cryotransformer_clean_tri.star` / `_removed_tri.star`), but what
the 2D selection discarded lives only inside CryoSPARC. `teacher.star` is not a
substitute: it covers the 50 sampled micrographs, not the 300 the round actually picked.

Three jobs per round are read, and all three are needed to attribute a pick correctly:

  extract    every particle that reached class_2D. Extraction drops picks whose box would
             cross the micrograph edge, so filter-kept minus extracted is an edge loss,
             not a selection rejection.
  class_2D   its `particles` output, which is not the whole input: class_2D rejects
             particles of its own accord (5,906 of 61,056 in one measured round, into
             `particles_rejected`). Those never reach the selector either.
  select_2D  the final iterative selection: the survivors.

Attributing either of those two losses to the selector would overstate it -- by 6,243
particles in that one round, a quarter of what the selection actually discarded.

All are written back in the loop's own GT-aligned (top-left origin, integer) convention,
so the five populations nest as plain set operations on (micrograph, x, y):

    picks >= filter-kept >= extracted >= class_2D-accepted >= survivors

The script asserts that nesting rather than assuming it. The coordinate inverse is the
one export_teacher_star.py derives and proves; the residual to the originating pick is
float32 rounding (measured 0.0001 px here), so integer keys match exactly.

Run it with the interpreter that has cryosparc-tools (the `recon` environment):

  python -m rapick.loop.export_stage_stars --id 10081
  python -m rapick.loop.export_stage_stars --id 10532 --rounds 1-2

Writes into each round dir: extracted.star, class2d_accepted.star, survivors.star,
stage_counts.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import entries, paths, star
from .common import connect_cryosparc, parse_rounds
from .run_loop import CLEAN_STAR


def load_particle_rows(job, output_name: str, slots=("location",)):
    """([(mic_key, x_int, y_int)] in input order, the loaded table) for one job output.

    The picks were imported with a Y flip (`ny - Y`) and Import Particles stored
    center_y_frac against that, so the inverse is `ny * (1 - center_y_frac)`.

    Set arithmetic over the five stages wants the keys and nothing else, but carrying a
    per-particle column along -- a 2D class number, say -- needs them in input order,
    which a set cannot give. Hence the two entry points over one coordinate inverse.
    """
    parts = job.load_output(output_name, slots=list(slots))
    mic_keys = [star.normalize_mic_name(str(p)) for p in parts["location/micrograph_path"]]
    shapes = np.array(parts["location/micrograph_shape"])          # (H, W) per particle
    ny = shapes[:, 0].astype(float)
    nx = shapes[:, 1].astype(float)
    x = np.array(parts["location/center_x_frac"], dtype=float) * nx
    y = ny * (1.0 - np.array(parts["location/center_y_frac"], dtype=float))

    rows = [(mic, int(round(px)), int(round(py))) for mic, px, py in zip(mic_keys, x, y)]
    return rows, parts


def load_particle_keys(job, output_name: str) -> set:
    """{(mic_key, x_int, y_int)} for one job output, in the GT-aligned convention."""
    rows, _ = load_particle_rows(job, output_name)
    keys = set(rows)
    if len(keys) != len(rows):
        sys.exit(f"error: {job.uid}/{output_name} has {len(rows) - len(keys)} particles "
                 f"sharing an integer coordinate; set arithmetic would drop them")
    return keys


def find_job_uid(empiar: str, arm: str, n: int, step: str) -> str:
    """One of the round's reconstruction jobs, from its manifest."""
    source = entries.source_name(arm, n)
    manifest = paths.manifest_dir(empiar, entries.SETTING_ANNOT, source) / "manifest.json"
    if manifest.is_file():
        uid = json.loads(manifest.read_text())["jobs"].get(step, {}).get("uid")
        if uid:
            return uid
    raise SystemExit(f"no {step} job recorded for {source} (EMPIAR-{empiar})")


def export_round(project, empiar: str, arm: str, n: int, force: bool) -> None:
    round_dir = entries.round_dir(empiar, n, arm)
    if not round_dir.is_dir():
        print(f"  round {n}: no such dir, skipped ({round_dir})")
        return

    out_paths = {"extracted": round_dir / "extracted.star",
                 "class2d_accepted": round_dir / "class2d_accepted.star",
                 "survivors": round_dir / "survivors.star"}
    if not force and all(p.is_file() for p in out_paths.values()):
        print(f"  round {n}: already exported, skipped (--force to redo)")
        return

    state = json.loads((round_dir / "state.json").read_text())
    select2d = state.get("select2d", {}).get("select2d")
    if not select2d:
        print(f"  round {n}: no final select_2D recorded, skipped")
        return
    extract = find_job_uid(empiar, arm, n, "extract")
    class2d = state.get("class2d", {}).get("uid") or find_job_uid(empiar, arm, n, "class2d")

    picks = star.star_keys(round_dir / "picks.star")
    cleaned = star.star_keys(round_dir / CLEAN_STAR)
    extracted = load_particle_keys(project.find_job(extract), "particles")
    accepted = load_particle_keys(project.find_job(class2d), "particles")
    survivors = load_particle_keys(project.find_job(select2d), "particles_selected")

    # The five populations must nest. A break here means a convention changed upstream,
    # and every attribution below it would be silently wrong rather than failing.
    for inner, outer, why in ((cleaned, picks, "filter-kept not within picks"),
                              (extracted, cleaned, "extracted not within filter-kept"),
                              (accepted, extracted, "class_2D-accepted not within extracted"),
                              (survivors, accepted, "survivors not within class_2D-accepted")):
        stray = inner - outer
        if stray:
            sys.exit(f"error: {why} for {arm} r{n} ({len(stray)} particles); "
                     f"example {next(iter(stray))}")

    for name, path in out_paths.items():
        star.write_star(path, {"extracted": extracted, "class2d_accepted": accepted,
                               "survivors": survivors}[name])

    counts = {"extract_job": extract, "class2d_job": class2d, "select2d_job": select2d,
              "picks": len(picks), "filter_kept": len(cleaned),
              "filter_removed": len(picks - cleaned),
              "extracted": len(extracted), "edge_dropped": len(cleaned - extracted),
              "class2d_accepted": len(accepted),
              "class2d_rejected": len(extracted - accepted),
              "survivors": len(survivors), "select2d_removed": len(accepted - survivors)}
    (round_dir / "stage_counts.json").write_text(json.dumps(counts, indent=2) + "\n")

    print(f"  round {n}: picks {counts['picks']:,} -> filter -{counts['filter_removed']:,} "
          f"-> edge -{counts['edge_dropped']:,} -> class_2D -{counts['class2d_rejected']:,} "
          f"-> selection -{counts['select2d_removed']:,} -> {counts['survivors']:,} survive "
          f"({counts['survivors'] / counts['picks']:.1%} of picks)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, choices=sorted(entries.ENTRIES),
                    dest="empiar", help="EMPIAR entry")
    ap.add_argument("--arm", choices=sorted(entries.ARMS), default=entries.DEFAULT_ARM)
    ap.add_argument("--rounds", default="0-2", help="'0-2' or '0,2'")
    ap.add_argument("--force", action="store_true",
                    help="re-export rounds already on disk")
    ap.add_argument("--project", default=None,
                    help="CryoSPARC project uid (default CRYOSPARC_PROJECT from .env)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cs = connect_cryosparc(paths.load_env())
    project = cs.find_project(paths.cryosparc_project(args.project))

    print(f"EMPIAR-{args.empiar} {args.arm} arm")
    for n in parse_rounds(args.rounds):
        export_round(project, args.empiar, args.arm, n, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
