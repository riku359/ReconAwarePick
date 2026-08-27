#!/usr/bin/env bash
# The annotated subset: 300 micrographs per entry, about 75 GB in all. They are the
# ground truth for the 2D detection metrics and the working set of the feedback loop.

. "$(dirname "$0")/_common.sh"

banner "Annotated micrographs (300 per entry, ~75 GB)"
run_dl "$DL/download_cryoppp_micrographs_only.py" \
    --data-root "$DATA" --ids "${ENTRIES[@]}" --workers "$WORKERS" --max-retries 5
echo "  -> $DATA/cryoppp/<id>/micrographs/"
