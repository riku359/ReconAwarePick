#!/usr/bin/env bash
# Reconstruct one condition and collect its metrics (Sec. S1).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Reconstruct one condition and collect its metrics (Sec. S1).

The chain is the same for every condition: import micrographs, patch CTF, import
particles, extract, 2D classify, three ab-initio reconstructions, three
homogeneous refinements, best of the three by GSFSC 0.143, and a local
resolution estimate on the winner.

  bash scripts/07_reconstruct.sh --entry 10081 --condition baseline
  bash scripts/07_reconstruct.sh --entry 10081 --condition both
  bash scripts/07_reconstruct.sh --entry 10081 --condition fb --setting full
  bash scripts/07_reconstruct.sh --entry 10081 --condition baseline --dry-run

Conditions fall into two groups. Most start from a STAR of their own and run
straight through. Five of them (select, both, cryosegnet_both, fb and fb_gt)
take their particles from a 2D class selection instead, so their chain starts one
step lower, at an existing Select 2D Classes job; this script finds that job in
the state.json that scripts/05_select2d.sh wrote and hands it to the driver that
can start there.

THREE SEEDS, NOT ONE. The protocol (Sec. 4.2) reports the best of three by GSFSC
0.143, so a single-seed run reproduces something the paper does not report. If a
seed's ab-initio dies, advance the seed number rather than reporting a
best-of-two as a best-of-three.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
CONDITION=""
SETTING="full"
SEEDS="0,1,2"
SELECT2D=""
DRY_RUN=""
GPUS="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)     ENTRY="$2"; shift 2 ;;
    --condition) CONDITION="$2"; shift 2 ;;
    --setting)   SETTING="$2"; shift 2 ;;
    --seeds)     SEEDS="$2"; shift 2 ;;
    --select2d)  SELECT2D="$2"; shift 2 ;;
    --gpus)      GPUS="$2"; shift 2 ;;
    --dry-run)   DRY_RUN="--dry-run"; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$ENTRY" ]; then
  echo "error: --entry is required" >&2
  exit 2
fi
if [ -z "$CONDITION" ]; then
  echo "error: --condition is required" >&2
  exit 2
fi
valid_entry "$ENTRY"
require_setting "$SETTING"

CONDCFG="$REPO/configs/conditions/$CONDITION.yaml"
DATACFG="$REPO/configs/datasets/empiar_$ENTRY.yaml"
if [ ! -f "$CONDCFG" ]; then
  echo "error: no such condition: $CONDITION" >&2
  echo "       Available: $(ls "$REPO/configs/conditions" | sed 's/\.yaml//' | tr '\n' ' ')" >&2
  exit 2
fi

# --help must work with nothing configured, so the roots are demanded only once
# the arguments are known to be valid.
require_roots

PY="$(venv_python recon)"

# Every call into the reconstruction CLI goes through here.
recon() {  # recon <subcommand> [flags...]
  PYTHONPATH="$REPO/src" "$PY" -m rapick.recon.cli "$@"
}

# Which conditions take their particles from a 2D class selection, and whose
# class_2D that selection sits on.
case "$CONDITION" in
  select)          FROM_SELECTION=1; PARENT="baseline"   ; OWN_STACK=0 ;;
  both)            FROM_SELECTION=1; PARENT="mask"       ; OWN_STACK=0 ;;
  cryosegnet_both) FROM_SELECTION=1; PARENT="cryosegnet" ; OWN_STACK=0 ;;
  fb)              FROM_SELECTION=1; PARENT="fb"         ; OWN_STACK=1 ;;
  fb_gt)           FROM_SELECTION=1; PARENT="fb_gt"      ; OWN_STACK=1 ;;
  *)               FROM_SELECTION=0; PARENT=""           ; OWN_STACK=0 ;;
esac

banner "Preflight"
recon check-setup --condition "$CONDCFG" --dataset "$DATACFG" --setting "$SETTING"

# `rapick-recon run` has no dry run: it either creates the chain or it does not.
# For a condition that goes through it, --dry-run therefore means the preflight
# above and a description of what would follow. The from-selection driver does
# have one, and it is passed through below.
if [ -n "$DRY_RUN" ] && [ "$FROM_SELECTION" -eq 0 ]; then
  echo
  echo "Dry run. The preflight passed; nothing was created. A real run would:"
  echo "  import the micrographs, estimate CTF, import $CONDITION's picks, extract,"
  echo "  2D classify at K=50, then ab-initio and refine on seeds $SEEDS, keep the"
  echo "  best by GSFSC 0.143, and estimate local resolution on it."
  echo "  Output: $WORK/empiar_$ENTRY/$SETTING/$CONDITION/"
  exit 0
fi

if [ "$FROM_SELECTION" -eq 0 ]; then
  banner "Reconstructing $CONDITION on $ENTRY ($SETTING), seeds $SEEDS"
  recon run --condition "$CONDCFG" --dataset "$DATACFG" --setting "$SETTING" \
      --seeds "$SEEDS" --gpus "$GPUS"
else
  # fb and fb_gt classify a stack of their own before anything can be selected on
  # it; the other three select on a class_2D their parent condition already made.
  MANIFEST="$WORK/empiar_$ENTRY/$SETTING/$PARENT/manifest.json"
  if [ "$OWN_STACK" -eq 1 ] && [ ! -f "$MANIFEST" ]; then
    banner "Building $CONDITION's own stack up to 2D classification"
    recon run --condition "$CONDCFG" --dataset "$DATACFG" --setting "$SETTING" \
        --seeds "$SEEDS" --gpus "$GPUS"
  fi
  if [ ! -f "$MANIFEST" ]; then
    echo "error: no manifest for the parent condition '$PARENT' at" >&2
    echo "         $MANIFEST" >&2
    echo "       Run it first:" >&2
    echo "         bash scripts/07_reconstruct.sh --entry $ENTRY --condition $PARENT --setting $SETTING" >&2
    exit 1
  fi

  # The selection's final Select 2D Classes job, at the 3.5 cutoff, is recorded in
  # the state.json that scripts/05_select2d.sh left behind.
  if [ -z "$SELECT2D" ]; then
    CLASS2D="$(manifest_class2d "$MANIFEST")"
    STATE="$(select2d_state_file "$CLASS2D")"
    if [ -f "$STATE" ]; then
      SELECT2D="$(select2d_at_cutoff "$STATE")"
    fi
  fi
  if [ -z "$SELECT2D" ]; then
    echo "error: no 2D class selection to reconstruct from." >&2
    echo "       Build it first:" >&2
    echo "         bash scripts/05_select2d.sh --entry $ENTRY --condition $CONDITION" >&2
    echo "       or name the job with --select2d J<n>." >&2
    exit 1
  fi

  banner "Reconstructing $CONDITION on $ENTRY from $SELECT2D, seeds $SEEDS"
  # This driver takes a single GPU, so hand it the first of the list.
  FIRST_GPU="$(echo "$GPUS" | cut -d, -f1)"
  # $DRY_RUN is deliberately left unquoted: it is either empty, and then adds no
  # argument at all, or the single word --dry-run.
  recon reconstruct-from-selection \
      --entry "$ENTRY" --select2d "$SELECT2D" \
      --condition "$CONDITION" --parent "$PARENT" --setting "$SETTING" \
      --seeds "$SEEDS" --gpu "$FIRST_GPU" $DRY_RUN
fi

if [ -n "$DRY_RUN" ]; then
  exit 0
fi

# The from-selection driver deliberately does not collect, so this is not optional:
# without it the arm ends up with a manifest and no metrics.json.
banner "Collecting metrics"
recon collect --condition "$CONDCFG" --dataset "$DATACFG" --setting "$SETTING"

OUT="$WORK/empiar_$ENTRY/$SETTING/$CONDITION"
echo
echo "Metrics: $OUT/metrics.json"
if [ -f "$OUT/metrics.json" ]; then
  # A one-line summary of the file just written. Reading it is a convenience, so a
  # failure here must not fail the run.
  uv run --quiet python3 -c '
import json, sys

metrics = json.load(open(sys.argv[1]))
best = metrics.get("best") or {}
resolution = metrics.get("res_gsfsc_0143") or best.get("res_gsfsc_0143")
if resolution:
    print(f"  GSFSC 0.143: {resolution} A  (best of the seeds run)")
' "$OUT/metrics.json" || true
fi
echo "Compare against results/tables/ — see docs/PAPER_TO_CODE.md for which table this row is."
