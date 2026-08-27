#!/usr/bin/env bash
# Put MicrographCleaner's (tf2) pretrained model in the right place.
#
# Warning: the package's own `cleanMics --download` writes Zenodo's
# deepMicrographCleaner.tgz to defaultModel.h5 after **only gunzipping** it, but the
# contents are in fact a tar (POSIX tar), so what it leaves behind is a broken file
# Keras cannot read. This script gunzips *and* untars, and extracts the real
# defaultModel.h5.

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

URL='https://zenodo.org/records/17093439/files/deepMicrographCleaner.tgz'

if [ -z "${RAPICK_DATA:-}" ]; then
  echo "error: environment variable RAPICK_DATA is not set; see docs/CONFIGURATION.md" >&2
  exit 1
fi
DEST="$RAPICK_DATA/checkpoints"
MODEL="$DEST/micrograph_cleaner_defaultModel.h5"

# A scratch directory that goes away however this script ends.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"
# The real model is about 127 MiB, so anything much smaller is a failed download
# rather than something worth keeping.
MIN_BYTES=100000000
if [ -f "$MODEL" ] && [ "$(wc -c <"$MODEL")" -gt "$MIN_BYTES" ]; then
  echo "model already present: $MODEL"
  exit 0
fi

echo "downloading $URL (~127 MiB) ..."
curl -sSL -o "$TMP/model.tgz" "$URL"
tar -xzf "$TMP/model.tgz" -C "$TMP"
cp "$TMP/deepMicrographCleaner/defaultModel.h5" "$MODEL"
echo "installed: $MODEL ($(wc -c <"$MODEL") bytes)"
