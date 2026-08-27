# Sourced by every script in this directory. Not executable on its own.
#
# There is one script per source rather than one script with a flag per source: what
# gets fetched is the file name, so nothing has to be remembered and nothing can be
# half-fetched. scripts/download.sh runs them all, in order; run one on its own to
# re-fetch just that source.

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# This file lives in scripts/download/, so the repository root is two directories up.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DL="$REPO/src/rapick/data"

# The paper's four entries. All four at full-set scale come to about 1.6 TB, so set
# RAPICK_ENTRIES to fetch a subset:
#   export RAPICK_ENTRIES=10081
# shellcheck disable=SC2206  # deliberate word splitting: a space-separated list
ENTRIES=(${RAPICK_ENTRIES:-10081 10093 10345 10532})

# Parallel transfers per downloader. Four is what EBI tolerates without throttling.
WORKERS=4

if [ -z "${RAPICK_DATA:-}" ]; then
  echo "error: RAPICK_DATA is not set." >&2
  echo "       Point it at the directory the inputs live in (docs/CONFIGURATION.md)." >&2
  exit 1
fi
if [ -z "${RAPICK_WORK:-}" ]; then
  echo "error: RAPICK_WORK is not set." >&2
  echo "       Point it at the directory the pipeline writes to (docs/CONFIGURATION.md)." >&2
  exit 1
fi
DATA="$RAPICK_DATA"
WORK="$RAPICK_WORK"
mkdir -p "$DATA" "$WORK"

# The downloaders read the CryoPPP catalogue spreadsheet, so they need openpyxl, and
# the Hugging Face ones need huggingface_hub. Run them through uv rather than
# requiring either in the ambient interpreter.
run_dl() { uv run --quiet --with openpyxl python3 "$@"; }
run_hf() { uv run --quiet --with huggingface_hub python3 "$DL/hf_assets.py" "$@"; }

banner() { echo; echo "==> $*"; }
