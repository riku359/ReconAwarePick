#!/usr/bin/env python3
"""
Download the FULL set of `.mrc` micrographs (not just the CryoPPP 300-subset)
for six EMPIAR datasets, straight from the EBI EMPIAR world_availability mirror.

CryoPPP ships only ~300 hand-picked micrographs per dataset. The original
depositions on EMPIAR contain many more (≈1k–1.6k each). This tool enumerates
the *actual* EMPIAR FTP/HTTP directory listing for each dataset and downloads
every `.mrc` micrograph it finds (with per-dataset include/exclude filters so we
grab the aligned micrographs and nothing else).

Layout written:
  <output-root>/<EMPIAR_ID>/micrographs/*.mrc

Mirrors the official CryoPPP downloader conventions:
  * curl --continue-at -   (interrupted transfers resume)
  * skip files already fully downloaded
  * logs / manifest / failures kept under cryoppp_tools/
URL base: https://ftp.ebi.ac.uk/empiar/world_availability/<source-dir>/<file>
"""

import argparse
import csv
import html
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability"

# Byte markers of an S3/HTTP error body that arrived in place of micrograph data.
ERROR_BODY_MARKERS = (
    b"<?xml",
    b"<Error>",
    b"ConnectionClosed",
    b"<!DOCTYPE html",
    b"<html",
)

# A marker alone is not evidence: float32 pixel data spells short markers by
# chance (see verify_mrc_integrity.py). Only ASCII text continuing past it is.
ERROR_TEXT_WINDOW_BYTES = 64
MIN_PRINTABLE_FRACTION = 0.9

# Read the whole file, not just its head. A resume that appends an error body
# onto a partial transfer buries it megabytes in, where a head probe never looks.
SCAN_CHUNK_BYTES = 8 << 20
MARKER_OVERLAP_BYTES = ERROR_TEXT_WINDOW_BYTES + max(len(m) for m in ERROR_BODY_MARKERS)

# Reported in the failure log where curl's own exit code would go. curl exits 0
# on these, and its real codes stay under 100, so this cannot collide.
ERROR_BODY_EXIT = 200

# Set from --data-root at the start of main() via _init_paths().
TOOLS_ROOT = DEST_ROOT_DEFAULT = MANIFEST = FAIL_LOG = None


def _init_paths(data_root):
    global TOOLS_ROOT, DEST_ROOT_DEFAULT, MANIFEST, FAIL_LOG
    data_root = os.path.abspath(os.path.expanduser(str(data_root)))
    TOOLS_ROOT = os.path.join(data_root, "cryoppp_tools")
    DEST_ROOT_DEFAULT = os.path.join(data_root, "cryoppp_fullset")
    MANIFEST = os.path.join(TOOLS_ROOT, "fullset_mrc_manifest.tsv")
    FAIL_LOG = os.path.join(TOOLS_ROOT, "fullset_mrc_failures.tsv")
    os.makedirs(TOOLS_ROOT, exist_ok=True)

# ----------------------------------------------------------------------------
# Per-dataset source directories on EMPIAR and the .mrc filter to apply.
# `dirs`    : list of (dir_path, prefix). dir_path is relative to BASE_URL;
#             prefix is prepended to the saved filename so files from different
#             source dirs that reuse the same basename do not collide on disk
#             (e.g. 10028 part1/part2 both number 001.mrc..). prefix "" = none.
# `include` : regex a basename must match to be kept (None = any .mrc).
# `exclude` : regex that, if it matches, drops the file (None = drop nothing).
# These were verified by listing the live EMPIAR directories on 2026-06-24.
# ----------------------------------------------------------------------------
DATASETS = {
    "10017": {
        "dirs": [("10017/data/", "")],
        "include": None,
        "exclude": None,
    },
    "10028": {
        "dirs": [
            ("10028/data/Micrographs/Micrographs_part1/", "part1_"),
            ("10028/data/Micrographs/Micrographs_part2/", "part2_"),
        ],
        "include": None,
        "exclude": None,
    },
    "10081": {
        "dirs": [("10081/data/micrographs/", "")],
        "include": None,
        "exclude": None,
    },
    "10093": {
        "dirs": [("10093/data/NOMPC/", "")],
        "include": None,
        "exclude": None,
    },
    "10345": {
        "dirs": [("10345/data/Micrographs_18jam15a/", "")],
        "include": None,
        "exclude": None,
    },
    "10532": {
        # This dir holds both *_patch_aligned.mrc (the real micrographs, 67 MB)
        # and *_background.mrc (4 MB background estimates). Keep only aligned.
        "dirs": [("10532/data/02_Aligned_Micrographs/motioncorrected/", "")],
        "include": r"_patch_aligned\.mrc$",
        "exclude": None,
    },
}

HREF_RE = re.compile(r'href="([^"?][^"]*)"', re.IGNORECASE)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def url_join(*parts):
    cleaned = [str(p).strip("/") for p in parts if p not in (None, "")]
    return "/".join(cleaned)


def list_mrc_in_dir(dir_path, max_tries=5):
    """Return basenames of *.mrc files (not .mrcs) in an EMPIAR HTTP dir."""
    url = url_join(BASE_URL, dir_path) + "/"
    last = ""
    for attempt in range(1, max_tries + 1):
        proc = subprocess.run(
            ["curl", "-sfL", "--connect-timeout", "30", "--max-time", "180", url],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and proc.stdout:
            names = []
            for m in HREF_RE.finditer(proc.stdout):
                name = html.unescape(m.group(1))
                if "/" in name:          # subdir or parent link
                    continue
                low = name.lower()
                if low.endswith(".mrc") and not low.endswith(".mrcs"):
                    names.append(name)
            return sorted(set(names))
        last = f"curl exit {proc.returncode}"
        time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"failed to list {url}: {last}")


def printable_fraction(raw):
    printable = sum(1 for byte in raw
                    if 0x20 <= byte < 0x7F or byte in (0x09, 0x0A, 0x0D))
    return printable / len(raw) if raw else 0.0


def has_error_body(path):
    """True if an S3/HTTP error document landed inside the file instead of data.

    EBI's S3 layer answers some requests with a `ConnectionClosedException` XML
    body under HTTP 200, so curl exits 0 and the bytes look like a normal
    transfer. Kept in sync with verify_mrc_integrity.py; duplicated rather than
    imported so this script stays numpy-free.
    """
    carried = b""
    try:
        with open(path, "rb") as f:
            while True:
                raw = f.read(SCAN_CHUNK_BYTES)
                if not raw:
                    return False
                chunk = carried + raw
                for marker in ERROR_BODY_MARKERS:
                    at = chunk.find(marker)
                    while -1 != at:
                        window = chunk[at:at + ERROR_TEXT_WINDOW_BYTES]
                        if printable_fraction(window) >= MIN_PRINTABLE_FRACTION:
                            return True
                        at = chunk.find(marker, at + 1)
                carried = raw[-MARKER_OVERLAP_BYTES:]
    except OSError:
        return False


def download_one(url, final, max_retries):
    """curl --continue-at - to <final>.part then atomically rename. -> (ok, msg, code).

    Every exit path out of the retry loop leaves no `.part` behind that a later
    `--continue-at -` could append to: resuming onto a poisoned prefix is how a
    single bad response turns into a permanently oversized file.
    """
    if os.path.exists(final) and os.path.getsize(final) > 0:
        return (True, "skip-exists", 0)
    part = final + ".part"
    code = -1

    # A leftover from an earlier run may already hold an error body.
    if os.path.exists(part) and has_error_body(part):
        remove_quietly(part)

    for attempt in range(1, max_retries + 1):
        proc = subprocess.run(
            ["curl", "-fSL", "--continue-at", "-", "--retry", "0",
             "--connect-timeout", "30", "-o", part, url],
            capture_output=True, text=True,
        )
        code = proc.returncode
        if code == 0 and os.path.exists(part) and os.path.getsize(part) > 0:
            if not has_error_body(part):
                os.replace(part, final)
                return (True, f"ok(attempt {attempt})", 0)
            remove_quietly(part)
            code = ERROR_BODY_EXIT
        elif code == 33 and os.path.exists(part):   # server can't resume -> restart
            remove_quietly(part)
        time.sleep(min(5 * attempt, 30))

    if ERROR_BODY_EXIT == code:
        return (False, f"error body from server after {max_retries} retries", code)
    return (False, f"curl exit {code} after {max_retries} retries", code)


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def append_tsv(path, row, header=None):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new and header:
            w.writerow(header)
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Download full EMPIAR .mrc micrograph sets.")
    ap.add_argument("--data-root", required=True,
                    help="REQUIRED. Data root; full-set .mrc land under <root>/cryoppp_fullset/.")
    ap.add_argument("--ids", nargs="+", default=sorted(DATASETS.keys()),
                    help="EMPIAR IDs to fetch (default: all six configured).")
    ap.add_argument("--output-root", default=None,
                    help="Override the output dir (default <data-root>/cryoppp_fullset).")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent downloads.")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap the number of files per dataset (debug).")
    ap.add_argument("--list-only", action="store_true",
                    help="Just enumerate and print per-dataset counts, no download.")
    args = ap.parse_args()

    _init_paths(args.data_root)   # sets TOOLS_ROOT / MANIFEST / FAIL_LOG / default
    out_root = os.path.expanduser(args.output_root) if args.output_root else DEST_ROOT_DEFAULT
    print(f"[{ts()}] output-root: {out_root}")
    print(f"[{ts()}] datasets: {args.ids}  workers={args.workers}")

    # ---- enumerate every dataset first ----
    tasks = []           # (sid, url, final_path)
    per_id_counts = {}
    for sid in args.ids:
        cfg = DATASETS[sid]
        inc = re.compile(cfg["include"], re.I) if cfg["include"] else None
        exc = re.compile(cfg["exclude"], re.I) if cfg["exclude"] else None
        out_dir = os.path.join(out_root, sid, "micrographs")
        os.makedirs(out_dir, exist_ok=True)
        names = []   # (url, save_name)
        for d, prefix in cfg["dirs"]:
            found = list_mrc_in_dir(d)
            for bn in found:
                if inc and not inc.search(bn):
                    continue
                if exc and exc.search(bn):
                    continue
                names.append((url_join(BASE_URL, d, bn), prefix + bn))
        # de-dup by saved filename (keep first seen)
        seen = set()
        uniq = []
        for url, save in names:
            if save in seen:
                continue
            seen.add(save)
            uniq.append((url, save))
        if args.limit > 0:
            uniq = uniq[:args.limit]
        per_id_counts[sid] = len(uniq)
        print(f"[{ts()}] {sid}: {len(uniq)} .mrc micrographs")
        for url, save in uniq:
            tasks.append((sid, url, os.path.join(out_dir, save)))

    print(f"[{ts()}] TOTAL queued: {len(tasks)} files "
          f"({', '.join(f'{k}={v}' for k, v in per_id_counts.items())})")

    if args.list_only:
        return

    ok = skip = fail = 0

    def run(t):
        sid, url, final = t
        o, msg, code = download_one(url, final, args.max_retries)
        return sid, url, final, o, msg, code

    def handle(res):
        nonlocal ok, skip, fail
        sid, url, final, o, msg, code = res
        bn = os.path.basename(final)
        if o and msg == "skip-exists":
            skip += 1
        elif o:
            ok += 1
            append_tsv(MANIFEST, [ts(), sid, bn, url],
                       header=["timestamp", "empiar_id", "filename", "url"])
        else:
            fail += 1
            append_tsv(FAIL_LOG, [ts(), sid, bn, url, code, msg],
                       header=["timestamp", "empiar_id", "filename", "url", "exit_code", "note"])

    n = len(tasks)
    if args.workers <= 1:
        for i, t in enumerate(tasks, 1):
            handle(run(t))
            if i % 25 == 0 or i == n:
                print(f"[{ts()}] progress {i}/{n} ok={ok} skip={skip} fail={fail}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                handle(fut.result())
                if done % 25 == 0 or done == n:
                    print(f"[{ts()}] progress {done}/{n} ok={ok} skip={skip} fail={fail}")

    print(f"\n[{ts()}] DONE. ok={ok} skip-exists={skip} fail={fail}")
    print(f"  manifest : {MANIFEST}")
    print(f"  failures : {FAIL_LOG}")


if __name__ == "__main__":
    main()
