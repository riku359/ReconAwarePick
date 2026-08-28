#!/usr/bin/env bash
# Download everything the pipeline reads, into $RAPICK_DATA and $RAPICK_WORK.
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Download everything the pipeline reads, into $RAPICK_DATA and $RAPICK_WORK.

  bash scripts/download.sh

It takes no arguments. Each source has its own script under scripts/download/, and
this runs them in name order; that order is the only thing this file decides. To
re-fetch one source, run its script directly:

  bash scripts/download/04_full_depositions.sh

All four entries at full-set scale come to about 1.6 TB. Every step reads
$RAPICK_ENTRIES, so `export RAPICK_ENTRIES=10081` fetches one entry instead.

Nothing is written into this repository. Resumable and idempotent: already-placed
files are skipped and partial transfers resume, so re-running after a failure costs
only what did not finish.
HELP
}

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# Taking no arguments is not the same as ignoring them: a run started by a typo is a
# 1.6 TB one, so anything on the command line stops it here rather than at the first
# byte transferred.
case "${1:-}" in
  "")        ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: scripts/download.sh takes no arguments (got '$1'). Run one source" >&2
     echo "       directly to re-fetch just it: bash scripts/download/<NN>_*.sh" >&2
     exit 2 ;;
esac

# This script lives in scripts/, so the repository root is the directory above it.
REPO="$(cd "$(dirname "$0")/.." && pwd)"

for step in "$REPO"/scripts/download/[0-9]*.sh; do
  echo
  echo "=== ${step#"$REPO"/} ==="
  bash "$step"
done

echo
echo "Done."
