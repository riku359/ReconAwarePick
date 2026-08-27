#!/usr/bin/env bash
# Run the reconstruction-aware feedback loop (Sec. 3.5).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Run the reconstruction-aware feedback loop (Sec. 3.5).

One round is the other drivers in this directory, in order:

    pick -> contamination_removal -> 2d_classification -> select2d -> finetune

and the checkpoint that comes out is what the next round picks with. Every round
restarts from theta_0 rather than from the previous round's weights, following
TranSPHIRE:

    theta_{n+1} = FineTune(theta_0; S_n),   S_n = sigma(f_theta_n(M))

What this adds over running those five by hand is the bookkeeping: a per-round
directory, a resumable step record, a lock so two rounds of one entry cannot run
at once, and the per-round diagnostics. How the teacher set is sampled and which
weights the fine-tune trains: src/rapick/loop/README.md.

The loop runs no reconstruction. At the 300-micrograph scale a reconstruction
does not resolve one round from the next, so rounds are followed by the 2D
metrics and the pick counts instead.

  bash scripts/loop.sh --entry 10081                 rounds 0 to 2
  bash scripts/loop.sh --entry 10081 --rounds 0-3
  bash scripts/loop.sh --entry 10081 --metrics-only  just rebuild Table 6

The paper runs three rounds and reports round 1: on every entry except
EMPIAR-10345 both the particle counts and the macro F1 settle there, and rounds
2 and 3 stay within 0.013 macro F1 of it.

Any flag this script does not recognise is passed through to the loop CLI; see
src/rapick/loop/README.md for the full list.
HELP
}

source "$(dirname "$0")/_common.sh"

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
    -h|--help)      usage; exit 0 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
if [ -z "$ENTRY" ]; then
  echo "error: --entry is required" >&2
  exit 2
fi
valid_entry "$ENTRY"

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

PY="$(venv_python cryotransformer)"

run_loop() {  # run_loop [extra flags for the loop CLI]
  PYTHONPATH="$REPO/src" "$PY" -m rapick.loop.run_loop \
      --id "$ENTRY" --rounds "$ROUNDS" --gpu "$GPU" "$@"
}

if [ "$METRICS_ONLY" -eq 0 ]; then
  require_upstream cryotransformer "CryoTransformer"
  CKPT="$DATA/checkpoints/CryoTransformer_head_repaired.pth"
  if [ ! -f "$CKPT" ]; then
    echo "error: theta_0 is missing at $CKPT." >&2
    echo "       Run scripts/download.sh, or scripts/repair_head.sh." >&2
    exit 1
  fi

  banner "Feedback loop on $ENTRY, rounds $ROUNDS"
  echo "    Each round fine-tunes for 50 epochs. Budget about two hours per round,"
  echo "    plus the 2D classification and the CryoSift cycles it waits on."
  # The empty case is spelled out because bash 3.2, which macOS still ships,
  # refuses to expand an empty array while `set -u` is in force.
  if [ ${#PASSTHROUGH[@]} -gt 0 ]; then
    run_loop "${PASSTHROUGH[@]}"
  else
    run_loop
  fi
fi

banner "Per-round diagnostics (Table 6)"
PYTHONPATH="$REPO/src" "$PY" -m rapick.loop.round_metrics \
    --id "$ENTRY" --csv "$WORK/loop/$ENTRY/rounds.csv"

echo
echo "Rounds:      $WORK/loop/$ENTRY/"
echo "Table 6 row: $WORK/loop/$ENTRY/rounds.csv"
echo
echo "The round-1 checkpoint is what the fb arm picks with:"
echo "  bash scripts/pick.sh --entry $ENTRY \\"
echo "      --checkpoint $WORK/loop/$ENTRY/models/model_1.pth \\"
echo "      --out $(picks_dir "$ENTRY")/fb.star"
