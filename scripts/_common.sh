# Sourced by every numbered driver. Not executable on its own.
#
# The drivers are thin: each one runs the commands its stage's README documents,
# with the entry and the paths filled in. Anything subtle lives in the stage
# README, not here, so that reading either explains the other.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
THIRD_PARTY="${RAPICK_THIRD_PARTY:-$REPO/third_party}"
ENTRIES_ALL=(10081 10093 10345 10532)

require_roots() {
  : "${RAPICK_DATA:?set RAPICK_DATA to the directory the inputs live in (docs/CONFIGURATION.md)}"
  : "${RAPICK_WORK:?set RAPICK_WORK to the directory the pipeline writes to (docs/CONFIGURATION.md)}"
  DATA="$RAPICK_DATA"
  WORK="$RAPICK_WORK"
}

# The environments are built per stage by scripts/00_setup.sh. Resolve one, or say
# which command would have built it rather than falling back to whatever python is
# on PATH and failing later with an import error.
venv_python() {  # venv_python <env name>
  local py="$REPO/envs/$1/.venv/bin/python"
  if [ ! -x "$py" ]; then
    echo "error: the '$1' environment is not built." >&2
    echo "       Run: bash scripts/00_setup.sh" >&2
    exit 1
  fi
  printf '%s' "$py"
}

require_upstream() {  # require_upstream <third_party subdir> <what it is>
  if [ ! -d "$THIRD_PARTY/$1" ]; then
    echo "error: $2 is not checked out at $THIRD_PARTY/$1." >&2
    echo "       Run: bash scripts/00_setup.sh" >&2
    exit 1
  fi
}

valid_entry() {  # valid_entry <id>
  local e
  for e in "${ENTRIES_ALL[@]}"; do [ "$e" = "$1" ] && return 0; done
  echo "error: unknown entry '$1'. The paper's entries are ${ENTRIES_ALL[*]}." >&2
  exit 2
}

banner() { echo; echo "==> $*"; }
