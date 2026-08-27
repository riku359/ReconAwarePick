#!/usr/bin/env python3
"""
Fallback recovery for CryoPPP `.mrc` micrographs that failed to download
individually from EMPIAR.

For every target EMPIAR ID that still has `.mrc` files MISSING on disk
(catalogue list vs. <output-root>/<ID>/micrographs/*.mrc), this tool:

  1. downloads the full per-ID archive from the CryoPPP server
       https://calla.rnet.missouri.edu/cryoppp/<ID>.tar.gz
     into a work-dir (curl --continue-at - so interrupted downloads resume),
  2. verifies the gzip/tar integrity,
  3. extracts ONLY `<ID>/micrographs/*.mrc` into a staging dir,
  4. moves ONLY the still-missing `.mrc` into <output-root>/<ID>/micrographs/
     (never overwrites an existing size>0 file; basename collisions are logged
      and skipped),
  5. deletes the archive + staging, keeping only the .mrc
     we actually needed), leaving the data dir with micrographs/*.mrc only.

All archives, staging, and logs live under cryoppp_tools — never under the
data dir. Existing data / .mrc files are never deleted.

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
from download_cryoppp_micrographs_only import (  # noqa: E402
    TARGET_IDS, load_catalog, human, ts,
)

CALLA_BASE = "https://calla.rnet.missouri.edu/cryoppp"

# Set from --data-root at the start of main() via _init_paths().
TOOLS_ROOT = DEST_ROOT_DEFAULT = WORK_DIR_DEFAULT = DEFAULT_CATALOG = None
RECOVERY_LOG = RECOVERY_FAIL_LOG = RECOVERY_COLLISION_LOG = None


def _init_paths(data_root):
    global TOOLS_ROOT, DEST_ROOT_DEFAULT, WORK_DIR_DEFAULT, DEFAULT_CATALOG
    global RECOVERY_LOG, RECOVERY_FAIL_LOG, RECOVERY_COLLISION_LOG
    data_root = os.path.abspath(os.path.expanduser(str(data_root)))
    TOOLS_ROOT = os.path.join(data_root, "cryoppp_tools")
    DEST_ROOT_DEFAULT = os.path.join(data_root, "cryoppp")
    WORK_DIR_DEFAULT = os.path.join(TOOLS_ROOT, "targz_recovery")
    DEFAULT_CATALOG = os.path.join(
        TOOLS_ROOT, "cryoppp",
        "download_micrographs_motion_correction_files",
        "micrographs_download_catalogue.xlsx",
    )
    RECOVERY_LOG = os.path.join(TOOLS_ROOT, "cryoppp_mrc_recovery.tsv")
    RECOVERY_FAIL_LOG = os.path.join(TOOLS_ROOT, "cryoppp_mrc_recovery_failures.tsv")
    RECOVERY_COLLISION_LOG = os.path.join(TOOLS_ROOT, "cryoppp_mrc_recovery_collisions.tsv")
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


def missing_for_id(sid, mrc_needed, out_root):
    """Return list of catalogue .mrc entries whose target file is missing/empty."""
    out_dir = os.path.join(out_root, sid, "micrographs")
    missing = []
    for entry in mrc_needed:
        bn = os.path.basename(str(entry))
        final = os.path.join(out_dir, bn)
        if not (os.path.exists(final) and os.path.getsize(final) > 0):
            missing.append(bn)
    return missing


def download_archive(sid, work_dir, max_retries):
    """curl --continue-at - the per-ID tar.gz. Returns (path, ok, msg)."""
    url = f"{CALLA_BASE}/{sid}.tar.gz"
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
                # size mismatch -> resume next attempt
                continue
            # integrity check
            test = subprocess.run(["gzip", "-t", dest], capture_output=True, text=True)
            if test.returncode == 0:
                return dest, True, f"ok(attempt {attempt})"
            # corrupt: remove and restart
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


def extract_micrographs(sid, archive, staging):
    """Extract only <sid>/micrographs/*.mrc into staging. Returns list of paths."""
    os.makedirs(staging, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", archive, "-C", staging, "--wildcards",
         f"{sid}/micrographs/*.mrc"],
        capture_output=True, text=True,
    )
    found = []
    for root, _dirs, files in os.walk(staging):
        for fn in files:
            if fn.lower().endswith(".mrc"):
                found.append(os.path.join(root, fn))
    return found


def main():
    ap = argparse.ArgumentParser(description="Recover failed CryoPPP .mrc via per-ID tar.gz.")
    ap.add_argument("--data-root", required=True,
                    help="REQUIRED. Data root; recovered .mrc land under <root>/cryoppp/.")
    ap.add_argument("--output-root", default=None,
                    help="Override the output dir (default <data-root>/cryoppp).")
    ap.add_argument("--work-dir", default=None,
                    help="Override the staging dir (default <data-root>/cryoppp_tools/targz_recovery).")
    ap.add_argument("--catalog", default=None,
                    help="Override the xlsx catalogue path (default under <data-root>/cryoppp_tools).")
    ap.add_argument("--ids", default="", help="Comma-separated IDs to force; default=auto-detect missing.")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-archive", action="store_true", help="Do not delete tar.gz after extraction.")
    args = ap.parse_args()

    _init_paths(args.data_root)   # sets TOOLS_ROOT / RECOVERY_* / defaults
    args.catalog = args.catalog or DEFAULT_CATALOG
    out_root = os.path.expanduser(args.output_root) if args.output_root else DEST_ROOT_DEFAULT
    work_dir = os.path.expanduser(args.work_dir) if args.work_dir else WORK_DIR_DEFAULT
    os.makedirs(work_dir, exist_ok=True)

    data_links, mrc_by_id, skip_reasons, collisions = load_catalog(args.catalog)

    if args.ids.strip():
        wanted = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        wanted = TARGET_IDS

    # Determine missing per ID (only IDs that have .mrc in the catalogue)
    plan = {}
    for sid in wanted:
        needed = mrc_by_id.get(sid, [])
        if not needed:
            continue  # 0-.mrc IDs (tif-only) have nothing to recover
        miss = missing_for_id(sid, needed, out_root)
        if miss:
            plan[sid] = miss

    print(f"[{ts()}] recovery scope: {len(plan)} ID(s) with missing .mrc")
    total_missing = sum(len(v) for v in plan.values())
    print(f"[{ts()}] total missing .mrc to recover: {total_missing}")
    if not plan:
        print(f"[{ts()}] nothing to do — all target .mrc present.")
        return

    # size estimate via HEAD on archives
    print(f"\n  {'ID':8} {'missing':>8} {'archive size':>14}")
    grand = 0
    sizes = {}
    for sid in sorted(plan):
        cl = curl_content_length(f"{CALLA_BASE}/{sid}.tar.gz")
        sizes[sid] = cl or 0
        grand += sizes[sid]
        print(f"  {sid:8} {len(plan[sid]):8} {human(cl):>14}")
    usage = shutil.disk_usage(out_root if os.path.exists(out_root) else os.path.dirname(out_root))
    print(f"\n[{ts()}] total archive download (peak, sequential): {human(grand)}")
    print(f"[{ts()}] free on data root: {human(usage.free)}")

    if args.dry_run:
        print(f"\n[{ts()}] dry-run only — no downloads performed.")
        return

    if usage.free < grand * 1.10:
        print(f"[{ts()}] INSUFFICIENT SPACE for archives (+10%). Aborting.", file=sys.stderr)
        sys.exit(3)

    recovered_total = 0
    failed_total = 0
    for sid in sorted(plan):
        miss_set = set(plan[sid])
        print(f"\n[{ts()}] === {sid}: recovering {len(miss_set)} missing .mrc ===")
        archive, ok, msg = download_archive(sid, work_dir, args.max_retries)
        if not ok:
            print(f"[{ts()}] {sid}: archive download failed: {msg}")
            for bn in miss_set:
                append_tsv(RECOVERY_FAIL_LOG,
                           [ts(), sid, bn, f"{CALLA_BASE}/{sid}.tar.gz", "archive-download-failed"],
                           header=["timestamp", "empiar_id", "filename", "url", "reason"])
            failed_total += len(miss_set)
            continue

        staging = os.path.join(work_dir, f"{sid}_staging")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        found = extract_micrographs(sid, archive, staging)
        found_by_bn = {os.path.basename(p): p for p in found}

        out_dir = os.path.join(out_root, sid, "micrographs")
        os.makedirs(out_dir, exist_ok=True)

        rec = 0
        for bn in sorted(miss_set):
            final = os.path.join(out_dir, bn)
            if os.path.exists(final) and os.path.getsize(final) > 0:
                continue  # appeared meanwhile (e.g. EBI finished it)
            src = found_by_bn.get(bn)
            if not src or os.path.getsize(src) == 0:
                append_tsv(RECOVERY_FAIL_LOG,
                           [ts(), sid, bn, f"{CALLA_BASE}/{sid}.tar.gz", "not-in-archive"],
                           header=["timestamp", "empiar_id", "filename", "url", "reason"])
                failed_total += 1
                continue
            # collision guard: never silently overwrite
            if os.path.exists(final):
                append_tsv(RECOVERY_COLLISION_LOG,
                           [ts(), sid, bn, "existing file present (size 0) — not overwriting"],
                           header=["timestamp", "empiar_id", "filename", "note"])
                failed_total += 1
                continue
            shutil.move(src, final)
            append_tsv(RECOVERY_LOG, [ts(), sid, bn, "recovered-from-targz"],
                       header=["timestamp", "empiar_id", "filename", "status"])
            rec += 1
        recovered_total += rec
        print(f"[{ts()}] {sid}: recovered {rec}/{len(miss_set)}")

        # cleanup staging + archive (keep only needed .mrc, now in data dir)
        shutil.rmtree(staging, ignore_errors=True)
        if not args.keep_archive:
            try:
                os.remove(archive)
            except OSError:
                pass

    print(f"\n[{ts()}] RECOVERY DONE. recovered={recovered_total} failed={failed_total}")
    print(f"  recovery log : {RECOVERY_LOG}")
    print(f"  failures     : {RECOVERY_FAIL_LOG}")
    print(f"  collisions   : {RECOVERY_COLLISION_LOG}")


if __name__ == "__main__":
    main()
