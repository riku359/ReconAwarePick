#!/usr/bin/env bash
# Select 2D classes with CryoSift's iterative workflow (Sec. 3.4, Sec. S4).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Select 2D classes with CryoSift's iterative workflow (Sec. 3.4, Sec. S4).

This stage decides at the level of a class rather than of a single candidate:
CryoSift scores each class average of an existing class_2D job, and three cycles
of re-classification narrow them down. The thresholds and the exact composition:
src/rapick/select2d/README.md.

  bash scripts/select2d.sh --entry 10081 --class2d J15
  bash scripts/select2d.sh --entry 10081 --name cryotransformer_mask
  bash scripts/select2d.sh --entry 10081 --class2d J15 --dry-run

  --class2d  the job to select on. scripts/2d_classification.sh prints it.
  --name     an arm whose manifest holds that job, if you would rather not
             copy the uid: $RAPICK_WORK/empiar_<entry>/<setting>/<name>/

Prints the final Select 2D Classes uid, which is what scripts/reconstruct.sh
reconstructs and what scripts/finetune.sh takes its teacher labels from.

RESUMABLE. Each cycle's re-classification runs for hours; the job uids are
recorded in state.json before they are queued, so an interrupted run continues
rather than restarting.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
SETTING="full"
CLASS2D=""
NAME=""
DRY_RUN=""
GPU="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)   ENTRY="$2"; shift 2 ;;
    --setting) SETTING="$2"; shift 2 ;;
    --class2d) CLASS2D="$2"; shift 2 ;;
    --name)    NAME="$2"; shift 2 ;;
    --gpu)     GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
require_entry "$ENTRY"
require_setting "$SETTING"

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

# Unless the job was named on the command line, take it from the arm's manifest.
if [ -z "$CLASS2D" ] && [ -n "$NAME" ]; then
  MANIFEST="$(arm_dir "$ENTRY" "$SETTING" "$NAME")/manifest.json"
  if [ -f "$MANIFEST" ]; then
    CLASS2D="$(manifest_class2d "$MANIFEST")"
  fi
fi
if [ -z "$CLASS2D" ]; then
  echo "error: no class_2D job to select on." >&2
  echo "       Classify a set of picks first:" >&2
  echo "         bash scripts/2d_classification.sh --entry $ENTRY --star <picks>.star" >&2
  echo "       then pass the uid it prints as --class2d J<n>, or name the arm it" >&2
  echo "       recorded with --name." >&2
  exit 1
fi

PY="$(venv_python cryosift)"
require_upstream magellon "CryoSift (Magellon)"

banner "Iterative 2D class selection on $CLASS2D ($ENTRY)"
# $DRY_RUN is deliberately left unquoted: it is either empty, and then adds no
# argument at all, or the single word --dry-run.
PYTHONPATH="$REPO/src" "$PY" -m rapick.select2d.iterate_class2d \
    --class2d "$CLASS2D" --gpu "$GPU" $DRY_RUN

if [ -n "$DRY_RUN" ]; then
  exit 0
fi

STATE="$(select2d_state_file "$CLASS2D")"
SELECT2D=""
if [ -f "$STATE" ]; then
  SELECT2D="$(select2d_at_cutoff "$STATE")"
fi

echo
echo "State: $STATE"
if [ -z "$SELECT2D" ]; then
  echo "The selection did not reach the 3.5 cutoff; state.json says how far it got."
  exit 1
fi
echo "select_2D: $SELECT2D"
echo
echo "Next, reconstruct what it kept:"
echo "  bash scripts/reconstruct.sh --entry $ENTRY --from-select2d $SELECT2D --name <arm>"
echo "or fine-tune on it, which is one round of the loop:"
echo "  bash scripts/finetune.sh --entry $ENTRY --select2d $SELECT2D --out <model>.pth"
