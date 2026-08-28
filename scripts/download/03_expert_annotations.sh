#!/usr/bin/env bash
# CryoPPP's expert particle annotations for those 300 micrographs, unpacked from the
# cryoppp_lite archives keeping only the .star files.

. "$(dirname "$0")/_common.sh"

banner "Expert annotations"
# This downloader wants the comma-separated spelling ($ENTRIES_CSV), not the array.
run_dl "$DL/download_cryoppp_star.py" \
    --data-root "$DATA" --ids "$ENTRIES_CSV" --max-retries 5
echo "  -> $DATA/cryoppp/<id>/ground_truth/empiar-<id>_particles_selected.star"
