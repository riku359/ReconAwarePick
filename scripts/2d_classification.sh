#!/usr/bin/env bash
# Import, extract and 2D classify one STAR, stopping before any reconstruction.
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Import, extract and 2D classify one STAR, stopping before any reconstruction.

  import_micrographs -> patch_ctf -> import_particles -> extract -> class_2D

The first two are created once per (entry, scale) and reused by every arm of that
entry, so all arms are compared over identical CTF estimates. K = 50 and 20 full
iterations, from configs/recon.yaml, the same for every arm and for the CryoSift
cycles, so the stacks stay comparable.

  bash scripts/2d_classification.sh --entry 10081 \
      --star $RAPICK_WORK/picks/10081/cryotransformer_mask.star

  --star     the picks to classify (required)
  --name     what to record the arm as (default: the STAR's name without .star).
             Names the manifest directory, $RAPICK_WORK/empiar_<entry>/<setting>/<name>/
  --setting  annot or full (default full).
  --gpus     which GPUs the CryoSPARC jobs queue on, comma-separated
             (default: $RAPICK_GPU, else 0). --gpu is accepted as an alias.
  --dry-run  run the preflight and print what a real run would create, creating
             nothing.

Prints the class_2D job uid, which is what scripts/select2d.sh selects on and what
scripts/reconstruct.sh starts its ab-initio from. This stage runs no reconstruction:
an arm that goes on to a 2D class selection would throw one away, and one that does
not gets it from scripts/reconstruct.sh.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
SETTING="full"
STAR=""
NAME=""
DRY_RUN=""
GPUS="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)   ENTRY="$2"; shift 2 ;;
    --setting) SETTING="$2"; shift 2 ;;
    --star)    STAR="$2"; shift 2 ;;
    --name)    NAME="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    --gpu)     GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
require_entry "$ENTRY"
require_setting "$SETTING"
if [ -z "$STAR" ]; then
  echo "error: --star is required: this stage classifies one set of picks." >&2
  echo "       bash scripts/pick.sh and scripts/contamination_removal.sh write them." >&2
  exit 2
fi
# The arm's name defaults to the STAR's, so the manifest directory says which stages
# the particles in it have been through without anyone having to name it twice.
[ -n "$NAME" ] || NAME="$(basename "$STAR" .star)"

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

if [ ! -f "$STAR" ]; then
  echo "error: no picks at $STAR." >&2
  exit 1
fi

PY="$(venv_python recon)"
DATACFG="$REPO/configs/datasets/empiar_$ENTRY.yaml"

banner "Classifying $(basename "$STAR") as '$NAME' ($ENTRY, $SETTING)"
# --star declares the picks whether or not the dataset config names them, so a
# re-pick does not need a config edit to be classified.
# $DRY_RUN is deliberately left unquoted: it is either empty, and then adds no
# argument at all, or the single word --dry-run.
# The run lasts hours, so its output is shown as it happens and read afterwards
# rather than swallowed into a variable.
# Spelled out rather than `mktemp -t rapick_class2d`: GNU coreutils' -t wants a
# template ending in at least three X's and refuses that one outright, so the driver
# died on its own log file on every Linux host.
LOG="$(mktemp "${TMPDIR:-/tmp}/rapick_class2d.XXXXXX")"
trap 'rm -f "$LOG"' EXIT
PYTHONPATH="$REPO/src" "$PY" -m rapick.loop.run_to_class2d \
    --env "$REPO/.env" --profile "$REPO/configs/cryosparc_v47.yaml" \
    --condition "$REPO/configs/recon.yaml" --dataset "$DATACFG" \
    --setting "$SETTING" --source "$NAME" --star "$STAR" \
    --gpus "$GPUS" $DRY_RUN 2>&1 | tee "$LOG"

if [ -n "$DRY_RUN" ]; then
  exit 0
fi

# The driver prints CLASS2D=<uid> as its last line; everything downstream is
# addressed by that uid rather than by re-deriving it from the manifest.
CLASS2D="$(sed -n 's/^CLASS2D=//p' "$LOG" | tail -1)"
if [ -z "$CLASS2D" ]; then
  echo "error: the classification reported no class_2D job." >&2
  exit 1
fi

echo
echo "class_2D: $CLASS2D"
echo "Manifest: $(arm_dir "$ENTRY" "$SETTING" "$NAME")/manifest.json"
echo
echo "Next, either select 2D classes on it:"
echo "  bash scripts/select2d.sh --entry $ENTRY --class2d $CLASS2D"
echo "or reconstruct the whole stack as it stands:"
echo "  bash scripts/reconstruct.sh --entry $ENTRY --name $NAME"
