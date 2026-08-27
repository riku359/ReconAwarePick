#!/usr/bin/env bash
# CryoPPP's expert particle annotations for those 300 micrographs, unpacked from the
# cryoppp_lite archives keeping only the .star files.

. "$(dirname "$0")/_common.sh"

banner "Expert annotations"
# This downloader wants one comma-separated list where the others take repeated ids,
# so join them: the entries with spaces between, then spaces turned into commas.
ENTRIES_CSV="$(echo "${ENTRIES[*]}" | tr ' ' ',')"
run_dl "$DL/download_cryoppp_star.py" \
    --data-root "$DATA" --ids "$ENTRIES_CSV" --max-retries 5
echo "  -> $DATA/cryoppp/<id>/ground_truth/empiar-<id>_particles_selected.star"
