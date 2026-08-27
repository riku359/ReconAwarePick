#!/usr/bin/env bash
# The whole EMPIAR deposition of each entry: 997 / 1,873 / 1,644 / 1,556 micrographs,
# about 1.5 TB in all. Every reconstruction-level result uses these; 300 micrographs
# are not enough for a stable reconstruction.
#
# This is the long one. Set RAPICK_ENTRIES to a single entry to fetch less.

. "$(dirname "$0")/_common.sh"

banner "Full depositions (~1.5 TB for all four entries)"
run_dl "$DL/download_empiar_fullset.py" \
    --data-root "$DATA" --ids "${ENTRIES[@]}" --workers "$WORKERS" --max-retries 5
echo "  -> $DATA/cryoppp_fullset/<id>/micrographs/"
