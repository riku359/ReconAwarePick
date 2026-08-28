# Sourced by every driver in this directory. Not executable on its own.
#
# The drivers are thin: each one runs the commands its stage's README documents,
# with the entry and the paths filled in. Anything subtle lives in the stage
# README, not here, so that reading either explains the other.
#
# Each driver is one transform with its inputs and its output named on the command
# line, so the order they run in is the order the arguments imply:
#
#   pick -> contamination_removal -> 2d_classification -> select2d -> reconstruct
#                                                                  -> finetune -> pick
#
# There are no stage numbers in the file names for that reason: what has to run
# first is whatever produces the file the next command is handed.

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
  reject_legacy_layout
}

# The STAR names changed, and one of them changed meaning rather than just spelling:
# `fb.star` used to hold the fb picks AFTER contamination removal and now holds them
# before it. A work directory written by the old layout would therefore be read as
# though the mask had been applied when it had not, and nothing downstream could tell.
# So refuse to run against one rather than guess which convention a file follows.
reject_legacy_layout() {
  # A newline-separated string rather than an array: bash 3.2, which macOS still
  # ships, refuses to expand an empty array while `set -u` is in force.
  local entry name round legacy=""
  for entry in "${ENTRIES_ALL[@]}"; do
    for name in fb_raw.star mask.star baseline.star; do
      if [ -f "$WORK/picks/$entry/$name" ]; then
        legacy="$legacy         picks/$entry/$name
"
      fi
    done
    # A round directory names its files the same way, so an old one would be read
    # under the new names too. `picks.star` there is the round's raw picks.
    for round in "$WORK"/loop/"$entry"/round*/picks.star; do
      if [ -f "$round" ]; then
        legacy="$legacy         ${round#"$WORK"/}
"
      fi
    done
  done
  if [ -z "$legacy" ]; then
    return 0
  fi
  echo "error: \$RAPICK_WORK holds picks under the old names:" >&2
  printf '%s' "$legacy" >&2
  cat >&2 <<'LEGACY'

       The names now say which stages a STAR has been through, and `fb.star` has
       changed meaning: it used to be the masked fb picks and is now the raw ones.
       Reading the old file under the new name would silently treat unmasked picks
       as masked, so rename them before running anything:

         cd "$RAPICK_WORK/picks/<entry>"
         mv fb.star       fb_mask.star          # do this one FIRST
         mv fb_raw.star   fb.star
         mv mask.star     cryotransformer_mask.star
         mv baseline.star cryotransformer.star

       A round directory renames the same way:

         cd "$RAPICK_WORK/loop/<entry>/round<n>"
         mv picks.star                     cryotransformer.star
         mv cryotransformer_clean_tri.star cryotransformer_mask.star
         mv survivors.star                 cryotransformer_mask_select.star

       Or point RAPICK_WORK somewhere new and re-fetch:
         RAPICK_ENTRIES=<entry> bash scripts/download.sh
LEGACY
  exit 1
}

# The environments are built per stage by scripts/setup.sh. Resolve one, or say which
# command would have built it rather than falling back to whatever python is on PATH
# and failing later with an import error.
venv_python() {  # venv_python <env name>
  local name="$1"
  local python="$REPO/envs/$name/.venv/bin/python"
  if [ ! -x "$python" ]; then
    echo "error: the '$name' environment is not built." >&2
    echo "       Run: bash scripts/setup.sh" >&2
    exit 1
  fi
  echo "$python"
}

require_upstream() {  # require_upstream <third_party subdir> <what it is>
  local subdir="$1"
  local what="$2"
  if [ ! -d "$THIRD_PARTY/$subdir" ]; then
    echo "error: $what is not checked out at $THIRD_PARTY/$subdir." >&2
    echo "       Run: bash scripts/setup.sh" >&2
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

require_entry() {  # require_entry <value>
  if [ -z "$1" ]; then
    echo "error: --entry is required" >&2
    exit 2
  fi
  valid_entry "$1"
}

# Where the micrographs of one entry live at each of the two scales.
micrograph_root() {  # micrograph_root <annot|full>
  if [ "$1" = "annot" ]; then
    echo "$DATA/cryoppp"
  else
    echo "$DATA/cryoppp_fullset"
  fi
}

# --- the three places a stage reads and writes -----------------------------------
# Needs the roots, so call these after require_roots.

# $RAPICK_WORK/picks/<id>: every STAR of one entry. The file name records which
# stages the picks have been through: cryotransformer.star -> cryotransformer_mask.star,
# fb.star -> fb_mask.star.
picks_dir() { echo "$WORK/picks/$1"; }

# $RAPICK_WORK/masks/<id>: one .npz per micrograph, written once per entry. The masks
# depend on the micrograph and not on the picks, so every checkpoint reads the same
# files. Matches rapick.loop.paths.mask_dir().
masks_dir() { echo "$WORK/masks/$1"; }

# $RAPICK_WORK/empiar_<id>/<setting>/<name>: one arm's CryoSPARC jobs and metrics.
arm_dir() { echo "$WORK/empiar_$1/$2/$3"; }

# --- reading the job uids the run recorded --------------------------------------
# Two stages have to find a CryoSPARC job that an earlier stage created. The uids
# live in JSON, which the shell cannot read, so a few lines of python read them.

# One job uid out of an arm's manifest, or an empty string if it has none.
manifest_job() {  # manifest_job <manifest.json> <step>
  uv run --quiet python3 -c '
import json, sys

manifest = json.load(open(sys.argv[1]))
job = (manifest.get("jobs") or {}).get(sys.argv[2])
print(job.get("uid", "") if isinstance(job, dict) else (job or ""))
' "$1" "$2"
}

manifest_class2d() {  # manifest_class2d <manifest.json>
  manifest_job "$1" class2d
}

# The STAR an arm's manifest says was imported. scripts/2d_classification.sh recorded
# it, so scripts/reconstruct.sh does not have to be handed the same path twice.
manifest_input_star() {  # manifest_input_star <manifest.json>
  uv run --quiet python3 -c '
import json, sys

manifest = json.load(open(sys.argv[1]))
print((manifest.get("input_star") or {}).get("path", ""))
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

# One value out of the .env every CryoSPARC stage reads, with the process environment
# winning over the file -- the same precedence rapick.recon.config.load_env applies. The
# shell needs this because the directory a stage writes into can be named after a
# credential: read only the environment and the name comes out with an empty field where
# the value should be, pointing at a directory nothing ever wrote.
env_value() {  # env_value <KEY>
  local key="$1" val=""
  eval "val=\${$key:-}"
  if [ -n "$val" ]; then
    echo "$val"
    return 0
  fi
  [ -f "$REPO/.env" ] || return 0
  sed -n "s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//p" "$REPO/.env" \
    | tail -1 | tr -d "\"'"
}

# Where scripts/select2d.sh keeps the state of one selection run. The name is
# <project>_<class_2D>_iter, and rapick.select2d.iterate_class2d builds it from the
# project in .env -- so this has to read the same place rather than only the environment,
# or the two disagree and every run reads back as "the selection did not get that far".
select2d_state_file() {  # select2d_state_file <class_2D uid>
  local project
  project="$(env_value CRYOSPARC_PROJECT)"
  echo "$WORK/select2d/${project}_$1_iter/state.json"
}

banner() { echo; echo "==> $*"; }
