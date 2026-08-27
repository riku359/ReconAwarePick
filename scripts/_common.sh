# Sourced by every numbered driver. Not executable on its own.
#
# The drivers are thin: each one runs the commands its stage's README documents,
# with the entry and the paths filled in. Anything subtle lives in the stage
# README, not here, so that reading either explains the other.

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# This file lives in scripts/, so the repository root is the directory above it.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${RAPICK_THIRD_PARTY:-$REPO/third_party}"
ENTRIES_ALL=(10081 10093 10345 10532)

# The two roots every stage needs, read here rather than at the top of a driver so
# that --help still works on a machine where nothing is configured yet.
# Sets DATA and WORK for the caller.
require_roots() {
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
}

# The environments are built per stage by scripts/00_setup.sh. Resolve one, or say
# which command would have built it rather than falling back to whatever python is
# on PATH and failing later with an import error.
venv_python() {  # venv_python <env name>
  local name="$1"
  local python="$REPO/envs/$name/.venv/bin/python"
  if [ ! -x "$python" ]; then
    echo "error: the '$name' environment is not built." >&2
    echo "       Run: bash scripts/00_setup.sh" >&2
    exit 1
  fi
  echo "$python"
}

require_upstream() {  # require_upstream <third_party subdir> <what it is>
  local subdir="$1"
  local what="$2"
  if [ ! -d "$THIRD_PARTY/$subdir" ]; then
    echo "error: $what is not checked out at $THIRD_PARTY/$subdir." >&2
    echo "       Run: bash scripts/00_setup.sh" >&2
    exit 1
  fi
}

valid_entry() {  # valid_entry <id>
  local known
  for known in "${ENTRIES_ALL[@]}"; do
    if [ "$known" = "$1" ]; then
      return 0
    fi
  done
  echo "error: unknown entry '$1'. The paper's entries are ${ENTRIES_ALL[*]}." >&2
  exit 2
}

require_setting() {  # require_setting <annot|full>
  if [ "$1" != "annot" ] && [ "$1" != "full" ]; then
    echo "error: --setting is annot or full" >&2
    exit 2
  fi
}

# Where the micrographs of one entry live at each of the two scales.
micrograph_root() {  # micrograph_root <annot|full>
  if [ "$1" = "annot" ]; then
    echo "$DATA/cryoppp"
  else
    echo "$DATA/cryoppp_fullset"
  fi
}

# --- reading the job uids the run recorded --------------------------------------
# Two stages have to find a CryoSPARC job that an earlier stage created. The uids
# live in JSON, which the shell cannot read, so a few lines of python read them.

# The class_2D job of a condition's run, or an empty string if the manifest has none.
manifest_class2d() {  # manifest_class2d <manifest.json>
  uv run --quiet python3 -c '
import json, sys

manifest = json.load(open(sys.argv[1]))
job = (manifest.get("jobs") or {}).get("class2d")
print(job.get("uid", "") if isinstance(job, dict) else (job or ""))
' "$1"
}

# The final Select 2D Classes job, at the 3.5 cutoff, or an empty string if the
# selection did not get that far.
select2d_at_cutoff() {  # select2d_at_cutoff <state.json>
  uv run --quiet python3 -c '
import json, sys

state = json.load(open(sys.argv[1]))
final = state.get("final_selects", {}).get("3.5") or {}
print(final.get("uid", ""))
' "$1"
}

# Where scripts/05_select2d.sh keeps the state of one selection run. Needs the
# roots, so call it after require_roots.
select2d_state_file() {  # select2d_state_file <class_2D uid>
  local project="${CRYOSPARC_PROJECT:-}"
  echo "$WORK/select2d/${project}_$1_iter/state.json"
}

banner() { echo; echo "==> $*"; }
