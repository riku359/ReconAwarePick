#!/usr/bin/env bash
# Pick candidates with CryoTransformer (Sec. 3.2).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Pick candidates with CryoTransformer (Sec. 3.2).

The picker over-picks on purpose: the two purification stages downstream remove
the background it accepts, and neither can recover a particle it never proposed.
The operating point is the original implementation's, unchanged at every round
and for every entry; the settings are in src/rapick/picker/README.md.

  bash scripts/03_pick.sh --entry 10081                  full deposition
  bash scripts/03_pick.sh --entry 10081 --setting annot  the 300 annotated
  bash scripts/03_pick.sh --entry 10081 --checkpoint PATH --out-name fb

Writes a GT-aligned STAR to $RAPICK_WORK/picks/<entry>/<out-name>.star, which
defaults to baseline.star. Re-picking with a fine-tuned checkpoint needs a
different --out-name, or it overwrites the base checkpoint's candidates.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
SETTING="full"
CKPT=""
OUT_NAME="baseline"
GPU="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)      ENTRY="$2"; shift 2 ;;
    --setting)    SETTING="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --out-name)   OUT_NAME="$2"; shift 2 ;;
    --gpu)        GPU="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$ENTRY" ]; then
  echo "error: --entry is required" >&2
  exit 2
fi
valid_entry "$ENTRY"
require_setting "$SETTING"

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

if [ -z "$CKPT" ]; then
  CKPT="$DATA/checkpoints/CryoTransformer_head_repaired.pth"
fi
if [ ! -f "$CKPT" ]; then
  echo "error: no checkpoint at $CKPT." >&2
  echo "       Run scripts/01_download_data.sh --intermediates, or scripts/02_repair_head.sh." >&2
  exit 1
fi

MICS="$(micrograph_root "$SETTING")/$ENTRY/micrographs"
if [ ! -d "$MICS" ]; then
  echo "error: no micrographs at $MICS. Run scripts/01_download_data.sh." >&2
  exit 1
fi

require_upstream cryotransformer "CryoTransformer"
PY="$(venv_python cryotransformer)"
CT="$THIRD_PARTY/cryotransformer"

# Upstream resolves its input as <root>/<entry>/images, and we kept that contract
# rather than patching the loader. Link this setting's micrographs into place.
TEST_DATA="${RAPICK_TEST_DATA:-$WORK/test_data}"
mkdir -p "$TEST_DATA/$ENTRY"
ln -sfn "$MICS" "$TEST_DATA/$ENTRY/images"

OUT_DIR="$WORK/picks/$ENTRY"
mkdir -p "$OUT_DIR"
REMARKS="rapick_${SETTING}_${OUT_NAME}"

banner "Picking $ENTRY ($SETTING) with $(basename "$CKPT")"
# Upstream writes into output/predictions/..._timestamp_<ts>/ relative to its own
# working directory, so the run happens inside the clone and the STAR is copied out
# afterwards. --remarks is what makes the directory findable among earlier runs.
( cd "$CT" && CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" predict_fullset.py --empiar "$ENTRY" --resume "$CKPT" --gt-format \
      --data_root "$TEST_DATA" --remarks "$REMARKS" )

banner "Collecting the STAR"
# ls -dt lists newest first, so the first line is the run that just finished.
# The trailing `|| true` keeps an empty or short-circuited listing from ending
# the script here, so that the check below can report it properly.
PRED_DIR="$(ls -dt "$CT"/output/predictions/predictions_EMPIAR_"$ENTRY"_remarks_"$REMARKS"_timestamp_* 2>/dev/null | head -1 || true)"
if [ -z "$PRED_DIR" ]; then
  echo "error: the picker wrote no output directory under $CT/output/predictions/." >&2
  exit 1
fi
STAR="$(ls "$PRED_DIR"/*star_file.star 2>/dev/null | head -1 || true)"
if [ -z "$STAR" ]; then
  echo "error: no combined STAR in $PRED_DIR." >&2
  exit 1
fi
cp "$STAR" "$OUT_DIR/$OUT_NAME.star"

echo
# grep -c . counts the lines that are not empty, which is what a STAR reader sees.
echo "Picks:  $OUT_DIR/$OUT_NAME.star  ($(grep -c . "$OUT_DIR/$OUT_NAME.star") lines)"
echo "Source: $PRED_DIR"
# The masking stage names its output after the condition that consumes it, which
# is `mask` for the base checkpoint's picks and `fb` for the round-1 one's.
if [ "$OUT_NAME" = "baseline" ]; then
  echo "Next:   bash scripts/04_mask.sh --entry $ENTRY"
else
  # Its convention is <name>_raw in, <name> out, so drop a trailing _raw.
  NEXT_NAME="$(echo "$OUT_NAME" | sed 's/_raw$//')"
  echo "Next:   bash scripts/04_mask.sh --entry $ENTRY \\"
  echo "          --star $OUT_DIR/$OUT_NAME.star --out-name $NEXT_NAME"
fi
