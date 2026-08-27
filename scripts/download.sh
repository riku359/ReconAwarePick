#!/usr/bin/env bash
# Download everything the pipeline reads, into $RAPICK_DATA and $RAPICK_WORK.
#
# It takes no arguments. Each source has its own script under scripts/download/, and
# this runs them in name order; that order is the only thing this file decides. To
# re-fetch one source, run its script directly:
#
#   bash scripts/download/04_full_depositions.sh
#
# Nothing is written into this repository. Resumable and idempotent: already-placed
# files are skipped and partial transfers resume, so re-running after a failure costs
# only what did not finish.

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# This script lives in scripts/, so the repository root is the directory above it.
REPO="$(cd "$(dirname "$0")/.." && pwd)"

for step in "$REPO"/scripts/download/[0-9]*.sh; do
  echo
  echo "=== ${step#"$REPO"/} ==="
  bash "$step"
done

echo
echo "Done."
