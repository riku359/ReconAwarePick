#!/usr/bin/env python3
"""
Download the CryoPPP ground-truth `.star` files for all 34 target EMPIAR IDs.

Source: the light per-ID archives on the CryoPPP server
        https://calla.rnet.missouri.edu/cryoppp_lite/<ID>.tar.gz
(much smaller than the full archives: ~0.5-2.4 GB vs 15-88 GB, and they contain
 the full-resolution ground-truth coordinate files including the .star).

Per ID, the archive contains under <ID>/ground_truth/:
    empiar-<ID>_particles_selected.star      (main ground truth)
    empiar-<ID>_particles_excluded.star      (excluded particles)
    intermediate_data/<ID>_manual_picked_particles.star
(plus *.csv coordinates and JPG micrographs, which are NOT kept).

This tool, for each ID:
  1. downloads <ID>.tar.gz into a work-dir (curl --continue-at - resume),
  2. verifies gzip integrity,
  3. extracts ONLY <ID>/ground_truth (no JPG micrographs),
  4. moves ONLY the *.star files into
        <output-root>/<ID>/ground_truth/...   (relative layout preserved,
        incl. intermediate_data/),
     never overwriting an existing size>0 file (collisions logged + skipped),
  5. deletes the archive + staging, and writes a per-ID done-marker so reruns
     skip already-completed IDs without re-downloading.

Archives, staging, logs, and done-markers live under cryoppp_tools — never in
the data dir. Existing data is never deleted.

Sources:
  https://github.com/BioinfoMachineLearning/cryoppp
  http://calla.rnet.missouri.edu/cryoppp
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_cryoppp_micrographs_only import TARGET_IDS, human, ts  # noqa: E402

LITE_BASE = "https://calla.rnet.missouri.edu/cryoppp_lite"

# Set from --data-root at the start of main() via _init_paths().
TOOLS_ROOT = DEST_ROOT_DEFAULT = WORK_DIR_DEFAULT = None
STAR_LOG = STAR_FAIL_LOG = STAR_COLLISION_LOG = None


def _init_paths(data_root):
    global TOOLS_ROOT, DEST_ROOT_DEFAULT, WORK_DIR_DEFAULT
    global STAR_LOG, STAR_FAIL_LOG, STAR_COLLISION_LOG
    data_root = os.path.abspath(os.path.expanduser(str(data_root)))
    TOOLS_ROOT = os.path.join(data_root, "cryoppp_tools")
    DEST_ROOT_DEFAULT = os.path.join(data_root, "cryoppp")
    WORK_DIR_DEFAULT = os.path.join(TOOLS_ROOT, "star_download")
    STAR_LOG = os.path.join(TOOLS_ROOT, "cryoppp_star_manifest.tsv")
    STAR_FAIL_LOG = os.path.join(TOOLS_ROOT, "cryoppp_star_failures.tsv")
    STAR_COLLISION_LOG = os.path.join(TOOLS_ROOT, "cryoppp_star_collisions.tsv")
    os.makedirs(TOOLS_ROOT, exist_ok=True)


def append_tsv(path, row, header=None):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new and header:
            w.writerow(header)
        w.writerow(row)


def curl_content_length(url, timeout=60):
    try:
        out = subprocess.run(["curl", "-sIL", "--max-time", str(timeout), url],
                             capture_output=True, text=True)
        cl = None
        for line in out.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                cl = int(line.split(":", 1)[1].strip())
        return cl
    except Exception:
        return None


def download_archive(sid, work_dir, max_retries):
    url = f"{LITE_BASE}/{sid}.tar.gz"
    dest = os.path.join(work_dir, f"{sid}.tar.gz")
    expected = curl_content_length(url)
    for attempt in range(1, max_retries + 1):
        proc = subprocess.run(
            ["curl", "-fSL", "--continue-at", "-", "--retry", "0",
             "--connect-timeout", "30", "-o", dest, url],
            capture_output=True, text=True,
        )
        code = proc.returncode
        if code == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            if expected and os.path.getsize(dest) != expected:
                continue
            if subprocess.run(["gzip", "-t", dest], capture_output=True).returncode == 0:
                return dest, True, f"ok(attempt {attempt})"
            try:
                os.remove(dest)
            except OSError:
                pass
        elif code == 33 and os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
    return dest, False, "download/integrity failed"


def extract_ground_truth(sid, archive, staging):
    """Extract the <sid> tree EXCEPT the (large) micrographs into staging, then
    collect every *.star under it. Extracting the whole tree minus micrographs
    (rather than a fixed <sid>/ground_truth) handles datasets with a non-standard
    layout such as 10389, which is split into 10389/10389A/ and 10389/10389B/,
    each with its own ground_truth/."""
    os.makedirs(staging, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", archive, "-C", staging,
         "--exclude", "*/micrographs/*", "--exclude", "*/micrographs", sid],
        capture_output=True, text=True,
    )
    stars = []
    base = os.path.join(staging, sid)
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith(".star"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, os.path.join(staging, sid))  # path under <ID>/
                stars.append((full, rel))
    return stars


def main():
    ap = argparse.ArgumentParser(description="Download CryoPPP ground-truth .star files.")
    ap.add_argument("--data-root", required=True,
                    help="REQUIRED. Data root; .star lands under <root>/cryoppp/, "
                         "run artifacts under <root>/cryoppp_tools/.")
    ap.add_argument("--output-root", default=None,
                    help="Override the output dir (default <data-root>/cryoppp).")
    ap.add_argument("--work-dir", default=None,
                    help="Override the staging/state dir (default <data-root>/cryoppp_tools/star_download).")
    ap.add_argument("--ids", default="", help="Comma-separated IDs; default=all 34.")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    _init_paths(args.data_root)   # sets TOOLS_ROOT / STAR_* / defaults
    out_root = os.path.expanduser(args.output_root) if args.output_root else DEST_ROOT_DEFAULT
    work_dir = os.path.expanduser(args.work_dir) if args.work_dir else WORK_DIR_DEFAULT
    state_dir = os.path.join(work_dir, "state")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids.strip() else list(TARGET_IDS)

    todo = [s for s in ids if not os.path.exists(os.path.join(state_dir, f"{s}.done"))]
    done_already = [s for s in ids if s not in todo]
    print(f"[{ts()}] target IDs: {len(ids)}  already-done: {len(done_already)}  to-process: {len(todo)}")

    # size estimate
    print(f"\n  {'ID':8} {'lite archive':>14} {'state':>10}")
    grand = 0
    for sid in ids:
        st = "DONE" if sid in done_already else "pending"
        cl = curl_content_length(f"{LITE_BASE}/{sid}.tar.gz") if sid in todo else 0
        grand += cl or 0
        szs = human(cl) if sid in todo else "-"
        print(f"  {sid:8} {szs:>14} {st:>10}")
    usage = shutil.disk_usage(out_root if os.path.exists(out_root) else os.path.dirname(out_root))
    print(f"\n[{ts()}] lite download needed (pending IDs): {human(grand)}")
    print(f"[{ts()}] free on data root: {human(usage.free)}")

    if args.dry_run:
        print(f"\n[{ts()}] dry-run only — no downloads performed.")
        return

    if usage.free < grand * 1.10:
        print(f"[{ts()}] INSUFFICIENT SPACE (+10%). Aborting.", file=sys.stderr)
        sys.exit(3)

    star_total = 0
    fail_ids = 0
    for sid in todo:
        print(f"\n[{ts()}] === {sid}: fetching ground-truth .star ===")
        archive, ok, msg = download_archive(sid, work_dir, args.max_retries)
        if not ok:
            print(f"[{ts()}] {sid}: archive download failed: {msg}")
            append_tsv(STAR_FAIL_LOG, [ts(), sid, f"{LITE_BASE}/{sid}.tar.gz", "archive-download-failed"],
                       header=["timestamp", "empiar_id", "url", "reason"])
            fail_ids += 1
            continue

        staging = os.path.join(work_dir, f"{sid}_staging")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        stars = extract_ground_truth(sid, archive, staging)
        if not stars:
            print(f"[{ts()}] {sid}: no .star found in archive!")
            append_tsv(STAR_FAIL_LOG, [ts(), sid, f"{LITE_BASE}/{sid}.tar.gz", "no-star-in-archive"],
                       header=["timestamp", "empiar_id", "url", "reason"])
            fail_ids += 1
            shutil.rmtree(staging, ignore_errors=True)
            if not args.keep_archive:
                try:
                    os.remove(archive)
                except OSError:
                    pass
            continue

        n = 0
        for src, rel in sorted(stars, key=lambda x: x[1]):
            final = os.path.join(out_root, sid, rel)   # <root>/<ID>/ground_truth/...
            os.makedirs(os.path.dirname(final), exist_ok=True)
            if os.path.exists(final) and os.path.getsize(final) > 0:
                continue  # keep existing, don't overwrite
            if os.path.exists(final):
                append_tsv(STAR_COLLISION_LOG, [ts(), sid, rel, "existing size-0 file, not overwritten"],
                           header=["timestamp", "empiar_id", "rel_path", "note"])
                continue
            shutil.move(src, final)
            append_tsv(STAR_LOG, [ts(), sid, rel, f"{os.path.getsize(final)}"],
                       header=["timestamp", "empiar_id", "rel_path", "bytes"])
            n += 1
        star_total += n
        print(f"[{ts()}] {sid}: placed {n} .star file(s) under {sid}/ground_truth/")

        # mark done + cleanup
        open(os.path.join(state_dir, f"{sid}.done"), "w").write(ts() + "\n")
        shutil.rmtree(staging, ignore_errors=True)
        if not args.keep_archive:
            try:
                os.remove(archive)
            except OSError:
                pass

    print(f"\n[{ts()}] STAR DONE. placed={star_total} star file(s); failed IDs={fail_ids}")
    print(f"  manifest  : {STAR_LOG}")
    print(f"  failures  : {STAR_FAIL_LOG}")
    print(f"  collisions: {STAR_COLLISION_LOG}")


if __name__ == "__main__":
    main()
