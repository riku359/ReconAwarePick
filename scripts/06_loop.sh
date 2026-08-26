#!/usr/bin/env bash
# Run the reconstruction-aware feedback loop (Sec. 3.5).
#
# One round: pick with the current checkpoint, discard the picks that fall on
# contamination, select 2D classes on what survives, take the surviving particles
# on 50 micrographs as pseudo-labels, and fine-tune. Every round restarts from
# theta_0 rather than from the previous round's weights, following TranSPHIRE:
#
#     theta_{n+1} = FineTune(theta_0; S_n),   S_n = sigma(f_theta_n(M))
#
# The teacher set is 50 micrographs, sampled with a fixed per-round seed from the
# micrographs that contain surviving particles and split 40/10 into training and
# validation; the 10 validation micrographs monitor the loss only. The fine-tune
# trains every weight of the picker except the first residual stage of the
# backbone, which stays frozen. It is not a LoRA adapter: the research repository
# had LoRA arms, they are not the paper's method, and they are not here.
#
# The loop runs no reconstruction. At the 300-micrograph scale a reconstruction
# does not resolve one round from the next, so rounds are followed by the 2D
# metrics and the pick counts instead.
#
#   bash scripts/06_loop.sh --entry 10081                 rounds 0 to 2
#   bash scripts/06_loop.sh --entry 10081 --rounds 0-3
#   bash scripts/06_loop.sh --entry 10081 --metrics-only  just rebuild Table 6
#
# The paper runs three rounds and reports round 1: on every entry except
# EMPIAR-10345 both the particle counts and the macro F1 settle there, and rounds
# 2 and 3 stay within 0.013 macro F1 of it.
#
# Any flag this script does not recognise is passed through to the loop CLI; see
# src/rapick/loop/README.md for the full list.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ENTRY=""
ROUNDS="0-2"
METRICS_ONLY=0
GPU="${RAPICK_GPU:-0}"
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)        ENTRY="$2"; shift 2 ;;
    --rounds)       ROUNDS="$2"; shift 2 ;;
    --gpu)          GPU="$2"; shift 2 ;;
    --metrics-only) METRICS_ONLY=1; shift ;;
    -h|--help)      sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
[ -n "$ENTRY" ] || { echo "error: --entry is required" >&2; exit 2; }
valid_entry "$ENTRY"

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

PY="$(venv_python cryotransformer)"

if [ "$METRICS_ONLY" -eq 0 ]; then
  require_upstream cryotransformer "CryoTransformer"
  CKPT="$DATA/checkpoints/CryoTransformer_head_repaired.pth"
  [ -f "$CKPT" ] || { echo "error: theta_0 is missing at $CKPT." >&2
                      echo "       Run scripts/01_download_data.sh --intermediates, or scripts/02_repair_head.sh." >&2
                      exit 1; }

  banner "Feedback loop on $ENTRY, rounds $ROUNDS"
  echo "    Each round fine-tunes for 50 epochs. Budget about two hours per round,"
  echo "    plus the 2D classification and the CryoSift cycles it waits on."
  PYTHONPATH="$REPO/src" "$PY" -m rapick.loop.run_loop \
      --id "$ENTRY" --rounds "$ROUNDS" --gpu "$GPU" "${PASSTHROUGH[@]}"
fi

banner "Per-round diagnostics (Table 6)"
PYTHONPATH="$REPO/src" "$PY" -m rapick.loop.round_metrics \
    --id "$ENTRY" --csv "$WORK/loop/$ENTRY/rounds.csv"

echo
echo "Rounds:      $WORK/loop/$ENTRY/"
echo "Table 6 row: $WORK/loop/$ENTRY/rounds.csv"
echo "Compare against results/tables/loop_rounds.json."
echo
echo "The round-1 checkpoint is what the fb condition picks with:"
echo "  bash scripts/03_pick.sh --entry $ENTRY --checkpoint $WORK/loop/$ENTRY/round1/model.pth"
