#!/usr/bin/env bash
# Turn a 2D class selection into the next checkpoint (Sec. 3.5, Eq. 1).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Turn a 2D class selection into the next checkpoint (Sec. 3.5, Eq. 1).

Two steps. The particles a selection kept, on 50 sampled micrographs, become
pseudo-labels; theta_0 is then fine-tuned on them:

    theta_{n+1} = FineTune(theta_0; S_n),   S_n = sigma(f_theta_n(M))

theta_0 every time, never the checkpoint that just picked. Resuming from the
picking model instead would let its own bias accumulate: it would be trained on
the particles it chose, having chosen them because it was trained on them. How the
teacher set is sampled and which weights the fine-tune trains:
src/rapick/loop/README.md.

  bash scripts/finetune.sh --entry 10081 --select2d J212 \
      --parent fb_r0 --round-dir $RAPICK_WORK/loop/10081/round0 \
      --out $RAPICK_WORK/loop/10081/models/model_1.pth

  --select2d   the selection to learn from. scripts/select2d.sh prints it.
  --parent     the arm whose class_2D that selection sits on. Its manifest names the
               extraction and the STAR, which together prove the coordinate inverse
               the labels are written back through.
  --out        where the checkpoint lands (required)
  --round-dir  where the labels and the training run land (default: --out's directory).
               Pass the round this fine-tune belongs to, so its inputs stay with the
               round that produced them and only the checkpoint moves.
  --num-mics   how many micrographs the teacher set draws (default 50)

The labels are on the 300 annotated micrographs, so this stage reads the `annot`
scale whatever the reconstruction is run at.

This is one step of scripts/loop.sh, runnable on its own. Running it repeatedly --
pick, contamination_removal, 2d_classification, select2d, finetune, pick again --
is exactly what the loop does; scripts/loop.sh adds the per-round bookkeeping, the
locking and the diagnostics.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
SELECT2D=""
PARENT=""
SOURCE_STAR=""
EXTRACT=""
OUT=""
ROUND_DIR=""
SETTING="annot"
NUM_MICS="50"
SEED="1"
GPU="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)     ENTRY="$2"; shift 2 ;;
    --select2d)  SELECT2D="$2"; shift 2 ;;
    --parent)    PARENT="$2"; shift 2 ;;
    --star)      SOURCE_STAR="$2"; shift 2 ;;
    --extract)   EXTRACT="$2"; shift 2 ;;
    --out)       OUT="$2"; shift 2 ;;
    --round-dir) ROUND_DIR="$2"; shift 2 ;;
    --setting)   SETTING="$2"; shift 2 ;;
    --num-mics)  NUM_MICS="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    --gpu)       GPU="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
require_entry "$ENTRY"
require_setting "$SETTING"
if [ -z "$SELECT2D" ]; then
  echo "error: --select2d is required: the labels come from what a selection kept." >&2
  echo "       bash scripts/select2d.sh --entry $ENTRY --class2d J<n>  prints it." >&2
  exit 2
fi
if [ -z "$OUT" ]; then
  echo "error: --out is required: name the checkpoint this writes." >&2
  exit 2
fi
if [ -z "$PARENT" ] && { [ -z "$SOURCE_STAR" ] || [ -z "$EXTRACT" ]; }; then
  echo "error: --parent is required (or --star and --extract together)." >&2
  echo "       It names the arm scripts/2d_classification.sh recorded, whose manifest" >&2
  echo "       holds both." >&2
  exit 2
fi

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

if [ -n "$PARENT" ]; then
  MANIFEST="$(arm_dir "$ENTRY" "$SETTING" "$PARENT")/manifest.json"
  if [ ! -f "$MANIFEST" ]; then
    echo "error: no manifest for the parent arm '$PARENT' at" >&2
    echo "         $MANIFEST" >&2
    exit 1
  fi
  [ -n "$SOURCE_STAR" ] || SOURCE_STAR="$(manifest_input_star "$MANIFEST")"
  [ -n "$EXTRACT" ]     || EXTRACT="$(manifest_job "$MANIFEST" extract)"
fi
if [ ! -f "$SOURCE_STAR" ]; then
  echo "error: no picks at '$SOURCE_STAR'." >&2
  echo "       The teacher labels are written back in that STAR's coordinate" >&2
  echo "       convention, so the run refuses to guess it." >&2
  exit 1
fi
if [ -z "$EXTRACT" ]; then
  echo "error: no extraction job to check the coordinate convention against." >&2
  echo "       A wrong inverse produces a plausible-looking STAR of wrong labels, so" >&2
  echo "       the check is not optional here." >&2
  exit 1
fi

[ -n "$ROUND_DIR" ] || ROUND_DIR="$(dirname "$OUT")"
mkdir -p "$ROUND_DIR" "$(dirname "$OUT")"
RECON_PY="$(venv_python recon)"
CT_PY="$(venv_python cryotransformer)"
require_upstream cryotransformer "CryoTransformer"

banner "Teacher labels from $SELECT2D ($NUM_MICS micrographs)"
# The sampling lives here rather than in the fine-tuner: the fine-tuner's own
# micrograph subsampling happens after the train/validation split, so 50 sampled
# there would not become 40 + 10. Sampling up front also leaves the chosen
# micrographs on disk as a reproduction input.
PYTHONPATH="$REPO/src" "$RECON_PY" -m rapick.loop.export_teacher_star \
    --select2d "$SELECT2D" --extract "$EXTRACT" --empiar "$ENTRY" \
    --input-star "$SOURCE_STAR" \
    --seed "$SEED" --num-mics "$NUM_MICS" --out-dir "$ROUND_DIR"

TEACHER="$ROUND_DIR/teacher.star"
if [ ! -f "$TEACHER" ]; then
  echo "error: no teacher labels at $TEACHER" >&2
  exit 1
fi

banner "Fine-tuning theta_0 on $(basename "$TEACHER")"
PYTHONPATH="$REPO/src" "$CT_PY" -m rapick.loop.finetune \
    --id "$ENTRY" --star "$TEACHER" --out-dir "$ROUND_DIR/finetune" --gpu "$GPU"

CKPT="$ROUND_DIR/finetune/checkpoint.pth"
if [ ! -f "$CKPT" ]; then
  echo "error: fine-tuning produced no checkpoint at $CKPT" >&2
  exit 1
fi
cp "$CKPT" "$OUT"

echo
echo "Teacher labels: $TEACHER"
echo "Checkpoint:     $OUT"
echo
echo "Next, pick with it, which starts the next round:"
echo "  bash scripts/pick.sh --entry $ENTRY --checkpoint $OUT \\"
echo "      --out $(picks_dir "$ENTRY")/fb.star"
