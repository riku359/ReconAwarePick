#!/usr/bin/env bash
# Select 2D classes with CryoSift's iterative workflow (Sec. 3.4, Sec. S4).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Select 2D classes with CryoSift's iterative workflow (Sec. 3.4, Sec. S4).

This stage decides at the level of a class rather than of a single candidate:
CryoSPARC classifies the surviving picks at K=50, CryoSift scores each class
average, and three cycles of re-classification narrow them down. The thresholds
and the exact composition: src/rapick/select2d/README.md.

  bash scripts/05_select2d.sh --entry 10081 --condition both
  bash scripts/05_select2d.sh --entry 10081 --condition both --class2d J15
  bash scripts/05_select2d.sh --entry 10081 --condition both --dry-run

The stage hangs off an existing class_2D job. It is found in the parent
condition's manifest, or named with --class2d. The parent of `select` is
`baseline`, of `both` is `mask`, of `cryosegnet_both` is `cryosegnet`, and `fb`
classifies its own stack.

RESUMABLE. Each cycle's re-classification runs for hours; the job uids are
recorded in state.json before they are queued, so an interrupted run continues
rather than restarting.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
CONDITION="both"
SETTING="full"
CLASS2D=""
DRY_RUN=""
GPU="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)     ENTRY="$2"; shift 2 ;;
    --condition) CONDITION="$2"; shift 2 ;;
    --setting)   SETTING="$2"; shift 2 ;;
    --class2d)   CLASS2D="$2"; shift 2 ;;
    --gpu)       GPU="$2"; shift 2 ;;
    --dry-run)   DRY_RUN="--dry-run"; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$ENTRY" ]; then
  echo "error: --entry is required" >&2
  exit 2
fi
valid_entry "$ENTRY"

# Which condition's run made the class_2D job this selection sits on.
case "$CONDITION" in
  select)          PARENT="baseline" ;;
  both)            PARENT="mask" ;;
  cryosegnet_both) PARENT="cryosegnet" ;;
  fb)              PARENT="fb" ;;
  *)
    echo "error: --condition must be one of select, both, cryosegnet_both, fb." >&2
    echo "       Those are the four conditions that apply 2D class selection." >&2
    exit 2
    ;;
esac

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

# Unless the job was named on the command line, take it from the parent
# condition's manifest.
if [ -z "$CLASS2D" ]; then
  MANIFEST="$WORK/empiar_$ENTRY/$SETTING/$PARENT/manifest.json"
  if [ -f "$MANIFEST" ]; then
    CLASS2D="$(manifest_class2d "$MANIFEST")"
  fi
fi
if [ -z "$CLASS2D" ]; then
  echo "error: no class_2D job to select on." >&2
  echo "       Run the parent condition first:" >&2
  echo "         bash scripts/07_reconstruct.sh --entry $ENTRY --condition $PARENT --setting $SETTING" >&2
  echo "       or name the job directly with --class2d J<n>." >&2
  exit 1
fi

PY="$(venv_python cryosift)"
require_upstream magellon "CryoSift (Magellon)"

banner "Iterative 2D class selection on $CLASS2D ($ENTRY, $CONDITION)"
# $DRY_RUN is deliberately left unquoted: it is either empty, and then adds no
# argument at all, or the single word --dry-run.
PYTHONPATH="$REPO/src" "$PY" -m rapick.select2d.iterate_class2d \
    --class2d "$CLASS2D" --gpu "$GPU" $DRY_RUN

if [ -n "$DRY_RUN" ]; then
  exit 0
fi

echo
echo "State: $(select2d_state_file "$CLASS2D")"
echo "Read the final Select 2D Classes uid out of it, then:"
echo "  bash scripts/07_reconstruct.sh --entry $ENTRY --condition $CONDITION --select2d J<n>"
