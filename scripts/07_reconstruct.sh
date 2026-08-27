#!/usr/bin/env bash
# Reconstruct one condition and collect its metrics (Sec. S1).
#
# The chain is the same for every condition: import micrographs, patch CTF, import
# particles, extract, 2D classify, three ab-initio reconstructions, three
# homogeneous refinements, best of the three by GSFSC 0.143, and a local
# resolution estimate on the winner.
#
#   bash scripts/07_reconstruct.sh --entry 10081 --condition baseline
#   bash scripts/07_reconstruct.sh --entry 10081 --condition both
#   bash scripts/07_reconstruct.sh --entry 10081 --condition fb --setting full
#   bash scripts/07_reconstruct.sh --entry 10081 --condition baseline --dry-run
#
# Conditions fall into two groups. Most start from a STAR of their own and run
# straight through. Four of them (select, both, cryosegnet_both, fb, and fb_gt)
# take their particles from a 2D class selection instead, so their chain starts one
# step lower, at an existing Select 2D Classes job; this script finds that job in
# the state.json that scripts/05_select2d.sh wrote and hands it to the driver that
# can start there.
#
# WHY THREE SEEDS. A single-seed resolution is not trustworthy. The seed-to-seed
# spread measured in this project ranges from 0.011 to 2.073 angstrom, so on some
# entries it exceeds the effect being compared. If a seed's ab-initio dies, advance
# the seed number rather than reporting a best-of-two as a best-of-three.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

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
    -h|--help)   sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ENTRY" ]     || { echo "error: --entry is required" >&2; exit 2; }
[ -n "$CONDITION" ] || { echo "error: --condition is required" >&2; exit 2; }
valid_entry "$ENTRY"
case "$SETTING" in (annot|full) ;; (*) echo "error: --setting is annot or full" >&2; exit 2 ;; esac

CONDCFG="$REPO/configs/conditions/$CONDITION.yaml"
DATACFG="$REPO/configs/datasets/empiar_$ENTRY.yaml"
[ -f "$CONDCFG" ] || { echo "error: no such condition: $CONDITION" >&2
                       echo "       Available: $(ls "$REPO/configs/conditions" | sed 's/\.yaml//' | tr '\n' ' ')" >&2
                       exit 2; }

# --help must work with nothing configured, so the roots are demanded only once
# the arguments are known to be valid.
require_roots

PY="$(venv_python recon)"
RECON=( env PYTHONPATH="$REPO/src" "$PY" -m rapick.recon.cli )
COMMON=( --condition "$CONDCFG" --dataset "$DATACFG" --setting "$SETTING" )

# Which conditions take their particles from a 2D class selection, and whose
# class_2D that selection sits on.
case "$CONDITION" in
  (select)          FROM_SELECTION=1; PARENT="baseline"   ; OWN_STACK=0 ;;
  (both)            FROM_SELECTION=1; PARENT="mask"       ; OWN_STACK=0 ;;
  (cryosegnet_both) FROM_SELECTION=1; PARENT="cryosegnet" ; OWN_STACK=0 ;;
  (fb)              FROM_SELECTION=1; PARENT="fb"         ; OWN_STACK=1 ;;
  (fb_gt)           FROM_SELECTION=1; PARENT="fb_gt"      ; OWN_STACK=1 ;;
  (*)               FROM_SELECTION=0; PARENT=""           ; OWN_STACK=0 ;;
esac

banner "Preflight"
"${RECON[@]}" check-setup "${COMMON[@]}"

if [ "$FROM_SELECTION" -eq 0 ]; then
  banner "Reconstructing $CONDITION on $ENTRY ($SETTING), seeds $SEEDS"
  "${RECON[@]}" run "${COMMON[@]}" --seeds "$SEEDS" --gpus "$GPUS" $DRY_RUN
else
  # fb and fb_gt classify a stack of their own before anything can be selected on
  # it; the other three select on a class_2D their parent condition already made.
  MANIFEST="$WORK/empiar_$ENTRY/$SETTING/$PARENT/manifest.json"
  if [ "$OWN_STACK" -eq 1 ] && [ ! -f "$MANIFEST" ]; then
    banner "Building $CONDITION's own stack up to 2D classification"
    "${RECON[@]}" run "${COMMON[@]}" --seeds "$SEEDS" --gpus "$GPUS" $DRY_RUN
  fi
  [ -f "$MANIFEST" ] || { echo "error: no manifest for the parent condition '$PARENT' at" >&2
                          echo "         $MANIFEST" >&2
                          echo "       Run it first:" >&2
                          echo "         bash scripts/07_reconstruct.sh --entry $ENTRY --condition $PARENT --setting $SETTING" >&2
                          exit 1; }

  # The selection's final Select 2D Classes job, at the 3.5 cutoff, is recorded in
  # the state.json that scripts/05_select2d.sh left behind.
  if [ -z "$SELECT2D" ]; then
    CLASS2D="$(uv run --quiet python3 -c '
import json, sys
j = (json.load(open(sys.argv[1])).get("jobs") or {}).get("class2d")
print(j.get("uid", "") if isinstance(j, dict) else (j or ""))
' "$MANIFEST")"
    STATE="$WORK/select2d/${CRYOSPARC_PROJECT:-}_${CLASS2D}_iter/state.json"
    if [ -f "$STATE" ]; then
      SELECT2D="$(uv run --quiet python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
print((s.get("final_selects", {}).get("3.5") or {}).get("uid", ""))
' "$STATE")"
    fi
  fi
  [ -n "$SELECT2D" ] || { echo "error: no 2D class selection to reconstruct from." >&2
                          echo "       Build it first:" >&2
                          echo "         bash scripts/05_select2d.sh --entry $ENTRY --condition $CONDITION" >&2
                          echo "       or name the job with --select2d J<n>." >&2
                          exit 1; }

  banner "Reconstructing $CONDITION on $ENTRY from $SELECT2D, seeds $SEEDS"
  "${RECON[@]}" reconstruct-from-selection \
      --entry "$ENTRY" --select2d "$SELECT2D" \
      --condition "$CONDITION" --parent "$PARENT" --setting "$SETTING" \
      --seeds "$SEEDS" --gpu "${GPUS%%,*}" $DRY_RUN
fi

[ -n "$DRY_RUN" ] && exit 0

# The from-selection driver deliberately does not collect, so this is not optional:
# without it the arm ends up with a manifest and no metrics.json.
banner "Collecting metrics"
"${RECON[@]}" collect "${COMMON[@]}"

OUT="$WORK/empiar_$ENTRY/$SETTING/$CONDITION"
echo
echo "Metrics: $OUT/metrics.json"
if [ -f "$OUT/metrics.json" ]; then
  uv run --quiet python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
res = m.get("res_gsfsc_0143") or (m.get("best") or {}).get("res_gsfsc_0143")
if res: print(f"  GSFSC 0.143: {res} A  (best of the seeds run)")
' "$OUT/metrics.json" || true
fi
echo "Compare against results/tables/ — see docs/PAPER_TO_CODE.md for which table this row is."
