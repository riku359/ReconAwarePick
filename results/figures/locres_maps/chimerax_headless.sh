#!/usr/bin/env bash
# Headless UCSF ChimeraX for these root-less, display-less GPU boxes.
#
# ChimeraX has no plain Linux tarball (only .deb/.rpm) and this account has no root,
# so we extract the .deb locally and render with `--offscreen`, which needs libOSMesa
# (absent from the system, present in llvmpipe's libosmesa6 .deb). Both .debs are
# cached ONCE on shared NFS; each node extracts them to LOCAL /tmp (NFS is slow to
# launch a 1 GB app from and the memory notes never to run apps off NFS). Extraction
# is idempotent -- a marker file skips it on later calls.
#
# Usage: chimerax_headless.sh [chimerax args...]
#   e.g. chimerax_headless.sh --silent --exit --script render.cxc
# It prepends `--offscreen --nogui` and wires LD_LIBRARY_PATH to the bundled OSMesa.
# Every argument is forwarded to ChimeraX, so this script takes no flags of its own.
#
# Override paths via env: CHIMERAX_CACHE (where the .deb packages are kept, so a
# cluster can share one copy), CHIMERAX_LOCAL (where they are extracted, per node).

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# Where the two .deb packages are kept. CHIMERAX_CACHE wins; otherwise they sit
# under the work root.
if [ -n "${CHIMERAX_CACHE:-}" ]; then
  CACHE="$CHIMERAX_CACHE"
elif [ -n "${RAPICK_WORK:-}" ]; then
  CACHE="$RAPICK_WORK/tools/chimerax"
else
  echo "error: set RAPICK_WORK, or CHIMERAX_CACHE, to say where the ChimeraX packages are kept" >&2
  exit 1
fi
LOCAL="${CHIMERAX_LOCAL:-/tmp/chimerax-1.12}"
CX_DEB="$CACHE/debs/ucsf-chimerax_1.12_ubuntu22.04_amd64.deb"
OSMESA_DEB="$CACHE/debs/libosmesa6_23.2.1_22.04.4_amd64.deb"

CX_BIN="$LOCAL/usr/bin/chimerax"
OSMESA_DIR="$LOCAL/osmesa/usr/lib/x86_64-linux-gnu"
MARKER="$LOCAL/.extracted_ok"

if [ ! -f "$MARKER" ]; then
  for deb in "$CX_DEB" "$OSMESA_DEB"; do
    if [ ! -f "$deb" ]; then
      echo "missing cached deb: $deb" >&2
      echo "  (populate $CACHE/debs from the UCSF ChimeraX download + Ubuntu libosmesa6)" >&2
      exit 1
    fi
  done
  echo "extracting ChimeraX + OSMesa to $LOCAL (first run on this node)..." >&2
  rm -rf "$LOCAL"
  mkdir -p "$LOCAL/osmesa"
  dpkg-deb -x "$CX_DEB" "$LOCAL"
  dpkg-deb -x "$OSMESA_DEB" "$LOCAL/osmesa"
  touch "$MARKER"
fi

export LD_LIBRARY_PATH="$OSMESA_DIR:${LD_LIBRARY_PATH:-}"
exec "$CX_BIN" --offscreen --nogui "$@"
