#!/usr/bin/env bash
# Put MicrographCleaner's (tf2) pretrained model in the right place.
#
# Warning: the package's own `cleanMics --download` writes Zenodo's
# deepMicrographCleaner.tgz to defaultModel.h5 after **only gunzipping** it, but the
# contents are in fact a tar (POSIX tar), so what it leaves behind is a broken file
# Keras cannot read. This script gunzips *and* untars, and extracts the real
# defaultModel.h5.
set -euo pipefail

URL='https://zenodo.org/records/17093439/files/deepMicrographCleaner.tgz'

: "${RAPICK_DATA:?environment variable RAPICK_DATA is not set; see docs/CONFIGURATION.md}"
DEST="$RAPICK_DATA/checkpoints"
MODEL="$DEST/micrograph_cleaner_defaultModel.h5"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"
if [ -f "$MODEL" ] && [ "$(wc -c <"$MODEL")" -gt 100000000 ]; then
  echo "model already present: $MODEL"; exit 0
fi

echo "downloading $URL (~127 MiB) ..."
curl -sSL -o "$TMP/model.tgz" "$URL"
tar -xzf "$TMP/model.tgz" -C "$TMP"
cp "$TMP/deepMicrographCleaner/defaultModel.h5" "$MODEL"
echo "installed: $MODEL ($(wc -c <"$MODEL") bytes)"
