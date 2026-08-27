#!/usr/bin/env bash
# MicrographCleaner's released contamination network, about 127 MiB from Zenodo.
# The work is in src/rapick/cleaner/download_model.sh, because the package's own
# `cleanMics --download` leaves behind a file Keras cannot read; that script explains
# why and does it correctly.

. "$(dirname "$0")/_common.sh"

banner "MicrographCleaner's released weights (~127 MiB)"
bash "$REPO/src/rapick/cleaner/download_model.sh"
