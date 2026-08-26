#!/usr/bin/env python3
"""
Verify downloaded CryoPPP / EMPIAR `.mrc` micrographs before feeding them to CryoSPARC.

An existence check is not enough: downloads fail in two ways that both leave a
file sitting on disk looking plausible.

  1. Parallel downloaders appending to the same `.part` — the byte stream is
     restarted and re-appended instead of seeked, so the file ends up LARGER
     than its header declares (or truncated, if the writer died first).
  2. EBI S3 returning a `ConnectionClosedException` XML error body that lands
     inside the `.mrc`. Size can look right; Patch CTF is where it blows up.

Checks per file (in increasing cost):

  header  — parse nx/ny/nz/mode/nsymbt, derive the declared size, compare to the
            actual size, and compare the shape against the modal shape of the set
  scan    — (--full) stream the data block: search the raw bytes for XML/HTML
            error-body markers, and check the pixels are finite and non-constant

The `MAP ` stamp at offset 208 is deliberately NOT required: EMPIAR old-format
micrographs omit it, so a strict signature check false-FAILs valid files.

Usage:
    python3 src/rapick/data/verify_mrc_integrity.py --dir <mrc dir> [--full]
    python3 src/rapick/data/verify_mrc_integrity.py \
        --data-root "$RAPICK_DATA" --dataset fullset --ids 10093 --full

Exit code is 1 if any file fails, so it can gate a pipeline run. `--delete-bad`
turns it into the repair half of a download/verify loop: files damaged in
transit are removed, and the next downloader pass re-fetches exactly those.
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

# MRC mode -> numpy dtype. Modes we never expect are left out on purpose so an
# unknown mode is reported rather than silently mis-sized.
MODE_DTYPES = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    6: np.uint16,
    12: np.float16,
}

HEADER_BYTES = 1024

# Statuses that mean "the transfer broke", so deleting and re-downloading is the
# repair. The content-level statuses (NONFINITE, CONSTANT, SHAPE_OUTLIER) are
# deliberately absent: re-fetching returns the same bytes, so --delete-bad would
# loop forever on a micrograph that is simply odd rather than damaged.
REDOWNLOADABLE_STATUSES = frozenset(
    {"BAD_HEADER", "TRUNCATED", "OVERSIZE", "ERROR_BODY"}
)

# Byte markers of an S3/HTTP error body that got written into the .mrc.
ERROR_BODY_MARKERS = (
    b"<?xml",
    b"<Error>",
    b"ConnectionClosed",
    b"<!DOCTYPE html",
    b"<html",
)

# A marker alone is not evidence. Pixel data is float32, and a 57 MB micrograph
# has enough byte sequences to spell a short marker by chance: 10081's
# HCN1apo_0667 carries a literal "<?xml" at offset 29805846 that the EBI server
# returns byte-for-byte, so it is data, not damage. What separates the two is
# what follows the marker — an error body continues as ASCII text, pixel data
# does not. Measured: 0.41 printable after the false positive, 1.00 after a real
# marker. The window looks FORWARD only; an error body injected mid-transfer is
# preceded by ordinary binary data, which would dilute a centred window.
ERROR_TEXT_WINDOW_BYTES = 64
MIN_PRINTABLE_FRACTION = 0.9

SCAN_CHUNK_BYTES = 32 << 20

# Carried between chunks so that a marker landing near a chunk boundary still
# has its whole forward window available to judge.
MARKER_OVERLAP_BYTES = ERROR_TEXT_WINDOW_BYTES + max(len(m) for m in ERROR_BODY_MARKERS)


def printable_fraction(raw):
    printable = sum(1 for byte in raw
                    if 0x20 <= byte < 0x7F or byte in (0x09, 0x0A, 0x0D))
    return printable / len(raw) if raw else 0.0


def find_error_body(raw):
    """-> the marker that continues as ASCII text, or None."""
    for marker in ERROR_BODY_MARKERS:
        at = raw.find(marker)
        while -1 != at:
            window = raw[at:at + ERROR_TEXT_WINDOW_BYTES]
            if printable_fraction(window) >= MIN_PRINTABLE_FRACTION:
                return marker
            at = raw.find(marker, at + 1)
    return None


class Report:
    """One row of the output TSV: what we found for a single .mrc."""

    def __init__(self, path):
        self.path = path
        self.size_bytes = None
        self.declared_bytes = None
        self.shape = None
        self.mode = None
        self.status = "OK"
        self.detail = ""

    def fail(self, status, detail):
        # First failure wins — later checks are meaningless once the header is wrong.
        if "OK" == self.status:
            self.status = status
            self.detail = detail

    @property
    def is_ok(self):
        return "OK" == self.status

    def row(self):
        return [
            os.path.basename(self.path), self.status, self.detail,
            self.size_bytes if self.size_bytes is not None else "",
            self.declared_bytes if self.declared_bytes is not None else "",
            "x".join(str(n) for n in self.shape) if self.shape else "",
            self.mode if self.mode is not None else "",
        ]


def read_header(path):
    """Return (nx, ny, nz, mode, nsymbt) from the 1024-byte MRC header."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"file shorter than the {HEADER_BYTES}-byte MRC header")

    words = np.frombuffer(raw, dtype="<i4")
    nx, ny, nz, mode = (int(w) for w in words[0:4])
    nsymbt = int(words[23])
    return nx, ny, nz, mode, nsymbt


def check_header(report):
    """Fill in shape/size fields and flag TRUNCATED / OVERSIZE / bad header."""
    try:
        nx, ny, nz, mode, nsymbt = read_header(report.path)
    except Exception as exc:
        report.fail("BAD_HEADER", str(exc))
        return

    report.shape = (nx, ny, nz)
    report.mode = mode

    if mode not in MODE_DTYPES:
        report.fail("BAD_HEADER", f"unknown MRC mode {mode}")
        return
    if min(nx, ny, nz) <= 0:
        report.fail("BAD_HEADER", f"non-positive dimension {nx}x{ny}x{nz}")
        return
    if nsymbt < 0:
        report.fail("BAD_HEADER", f"negative extended-header size {nsymbt}")
        return

    itemsize = np.dtype(MODE_DTYPES[mode]).itemsize
    report.declared_bytes = HEADER_BYTES + nsymbt + nx * ny * nz * itemsize
    report.size_bytes = os.path.getsize(report.path)

    excess = report.size_bytes - report.declared_bytes
    if excess < 0:
        report.fail("TRUNCATED", f"{-excess} bytes short of the declared size")
    elif excess > 0:
        # A duplicated-append leaves the declared extent intact, so CryoSPARC
        # still reads it — but the file is evidence of a broken download.
        report.fail("OVERSIZE", f"{excess} bytes past the declared size")


def scan_data(report):
    """Stream the data block looking for an error body or unusable pixels."""
    if report.declared_bytes is None:
        return

    dtype = np.dtype(MODE_DTYPES[report.mode])
    data_offset = report.declared_bytes - np.prod(report.shape) * dtype.itemsize
    remaining = report.declared_bytes - data_offset

    nonfinite_count = 0
    value_min, value_max = np.inf, -np.inf
    carried = b""      # tail of the previous chunk, so markers can straddle

    with open(report.path, "rb") as f:
        f.seek(data_offset)
        while remaining > 0:
            raw = f.read(min(SCAN_CHUNK_BYTES, remaining))
            if not raw:
                report.fail("TRUNCATED", "data block ended early")
                return
            remaining -= len(raw)

            marker = find_error_body(carried + raw)
            if marker:
                report.fail("ERROR_BODY", f"contains {marker.decode()!r}")
                return
            carried = raw[-MARKER_OVERLAP_BYTES:]

            usable = len(raw) - (len(raw) % dtype.itemsize)
            values = np.frombuffer(raw[:usable], dtype=dtype)
            if np.issubdtype(dtype, np.floating):
                finite = np.isfinite(values)
                nonfinite_count += int((~finite).sum())
                values = values[finite]
            if values.size:
                value_min = min(value_min, float(values.min()))
                value_max = max(value_max, float(values.max()))

    if nonfinite_count:
        report.fail("NONFINITE", f"{nonfinite_count} NaN/Inf pixels")
    elif value_min == value_max:
        report.fail("CONSTANT", f"every pixel is {value_min}")


def check_shape_consistency(reports):
    """Flag micrographs whose shape differs from the modal shape of the set."""
    shapes = Counter(r.shape for r in reports if r.shape)
    if len(shapes) < 2:
        return
    modal_shape, _ = shapes.most_common(1)[0]
    for report in reports:
        if report.shape and report.shape != modal_shape:
            report.fail(
                "SHAPE_OUTLIER",
                f"{report.shape} differs from the set's {modal_shape}",
            )


def collect_mrc_dirs(args):
    if args.dir:
        return [os.path.abspath(os.path.expanduser(d)) for d in args.dir]

    root = os.path.abspath(os.path.expanduser(args.data_root))
    subdir = "cryoppp_fullset" if "fullset" == args.dataset else "cryoppp"
    if not args.ids:
        parent = os.path.join(root, subdir)
        ids = sorted(d for d in os.listdir(parent)
                     if os.path.isdir(os.path.join(parent, d)))
    else:
        ids = args.ids
    return [os.path.join(root, subdir, i, "micrographs") for i in ids]


def verify_dir(mrc_dir, do_full_scan):
    paths = sorted(os.path.join(mrc_dir, n) for n in os.listdir(mrc_dir)
                   if n.endswith(".mrc"))
    reports = []
    for index, path in enumerate(paths, start=1):
        report = Report(path)
        check_header(report)
        if do_full_scan and report.is_ok:
            scan_data(report)
        reports.append(report)

        if 0 == index % 100 or index == len(paths):
            failed = sum(1 for r in reports if not r.is_ok)
            print(f"  [{index}/{len(paths)}] {failed} failing", flush=True)

    check_shape_consistency(reports)
    return reports


def delete_reports(reports):
    """Remove the reported files. -> how many are gone from disk."""
    deleted = 0
    for report in reports:
        try:
            os.remove(report.path)
            deleted += 1
        except OSError as exc:
            print(f"!! could not delete {report.path}: {exc}", file=sys.stderr)
    return deleted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", action="append",
                        help="explicit micrograph directory (repeatable)")
    parser.add_argument("--data-root", default=os.environ.get("RAPICK_DATA"),
                        help="input root; defaults to $RAPICK_DATA")
    parser.add_argument("--dataset", choices=["subset", "fullset"], default="fullset")
    parser.add_argument("--ids", nargs="*", help="EMPIAR ids under --data-root")
    parser.add_argument("--full", action="store_true",
                        help="also stream the data block (slow: reads every byte)")
    parser.add_argument("--report", help="write the per-file TSV here")
    parser.add_argument("--delete-bad", action="store_true",
                        help="delete files damaged in transit so the (idempotent) "
                             "downloader re-fetches them; it skips any file that "
                             "merely exists, so corruption is permanent otherwise")
    args = parser.parse_args()

    if not args.dir and not args.data_root:
        parser.error("give either --dir or --data-root")

    all_reports = []
    for mrc_dir in collect_mrc_dirs(args):
        if not os.path.isdir(mrc_dir):
            print(f"!! missing directory: {mrc_dir}", file=sys.stderr)
            continue
        print(f"== {mrc_dir}", flush=True)
        all_reports.extend(verify_dir(mrc_dir, args.full))

    failures = [r for r in all_reports if not r.is_ok]

    if args.report:
        with open(args.report, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["name", "status", "detail", "size_bytes",
                             "declared_bytes", "shape", "mode"])
            for report in all_reports:
                writer.writerow(report.row())
        print(f"\nreport -> {args.report}")

    print(f"\nchecked {len(all_reports)} .mrc, {len(failures)} failing")
    for report in failures:
        print(f"  {report.status:14s} {os.path.basename(report.path)}  {report.detail}")

    if args.delete_bad:
        deleted = delete_reports(r for r in failures
                                 if r.status in REDOWNLOADABLE_STATUSES)
        kept = len(failures) - deleted
        print(f"\ndeleted {deleted} damaged .mrc for re-download"
              f"{f', kept {kept} not fixable by re-downloading' if kept else ''}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
