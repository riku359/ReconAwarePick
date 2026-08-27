#!/usr/bin/env python3
"""Fetch the figures CryoSPARC renders for its own jobs.

Several supplementary panels are not drawn by anything here: they are the plots and
class-average tiles CryoSPARC renders inside its own jobs (FSC curves, viewing-direction
distributions, `class2D_<i>.png` tiles, extraction renders). Those assets live in
CryoSPARC's GridFS rather than on disk, so `cryosparc-tools` is the only way to read
them.

NEEDS A LIVE CRYOSPARC INSTANCE. Run this wherever the instance is reachable; the
credentials come from the repository-root `.env` like the rest of the repository
(docs/CONFIGURATION.md). Nothing here prints a credential.

Each asset is written as `<JOB>__<filename>` in the output directory, which is the name
every downstream builder in `results/figures/` looks for.

    python cs_fetch_assets.py --project P1 --out /tmp/tiles \\
        --spec 'J115=class2D_0.png,J115=class2D_1.png'

`--spec` is a comma-separated list of `JOB=filename`. The filename may be a substring;
the first asset of that job whose name contains it is taken. Fifty tiles at a time is
the usual case, so expand the list in the shell:

    SPEC=$(python3 -c "print(','.join('J115=class2D_%d.png' % i for i in range(50)))")
    python cs_fetch_assets.py --project P1 --out /tmp/tiles --spec "$SPEC"

The job uids are the authors' instance. A fresh run of the pipeline produces the same
job chain with different uids, so read them out of your own project rather than reusing
the ones the figure READMEs quote.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_paths                                    # noqa: E402


def parse_spec(spec):
    """`'J115=class2D_0.png,J116=fsc'` -> [('J115', 'class2D_0.png'), ('J116', 'fsc')]."""
    items = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        uid, _, want = item.partition("=")
        if not uid.strip() or not want.strip():
            raise SystemExit(f"--spec expects JOB=filename entries, got: {item}")
        items.append((uid.strip(), want.strip()))
    return items


def fetch(cs, project, items, out_dir):
    """Write each requested asset as `<JOB>__<filename>`; report the ones that miss."""
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = {}
    written, missed = 0, []
    for uid, want in items:
        if uid not in listing:
            listing[uid] = cs.find_job(project, uid).list_assets()
        hit = next((a for a in listing[uid]
                    if a["filename"] == want or want in a["filename"]), None)
        if hit is None:
            missed.append(f"{uid} {want}")
            continue
        buffer = io.BytesIO()
        cs.download_asset(hit["_id"], buffer)
        data = buffer.getvalue()
        (out_dir / f"{uid}__{hit['filename']}").write_bytes(data)
        print(f"saved {uid} {hit['filename']} {len(data)}")
        written += 1
    for miss in missed:
        sys.stderr.write(f"MISS {miss}\n")
    print(f"{written} assets in {out_dir}"
          + (f", {len(missed)} not found" if missed else ""))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project",
                        help="CryoSPARC project uid (default: CRYOSPARC_PROJECT in .env)")
    parser.add_argument("--spec", required=True,
                        help="comma separated JOB=filename (filename may be a substring)")
    parser.add_argument("--out", required=True, type=Path,
                        help="directory the assets are written into")
    parser.add_argument("--env", default=None,
                        help="credentials file (default: the repository-root .env)")
    args = parser.parse_args()

    project = figure_paths.cryosparc_project(args.project, args.env)
    cs = figure_paths.connect_cryosparc(args.env)
    fetch(cs, project, parse_spec(args.spec), args.out)


if __name__ == "__main__":
    main()
