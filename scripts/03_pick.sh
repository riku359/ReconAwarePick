#!/usr/bin/env bash
# Pick candidates with CryoTransformer (Sec. 3.2).
#
# The picker over-picks on purpose: it emits 600 scored candidate queries per
# micrograph, keeps the top 75% by score, and removes duplicates by non-maximum
# suppression at an overlap threshold of 0.7. Accepting background candidates in
# exchange for missing few true particles is what the two purification stages
# downstream are for, and neither of them can recover a particle the picker never
# proposed.
#
# The operating point is the original implementation's, unchanged at every round
# and for every entry.
#
#   bash scripts/03_pick.sh --entry 10081                  full deposition
#   bash scripts/03_pick.sh --entry 10081 --setting annot  the 300 annotated
#   bash scripts/03_pick.sh --entry 10081 --checkpoint PATH --out-name fb
#
# Writes a GT-aligned STAR to $RAPICK_WORK/picks/<entry>/<out-name>.star, which
# defaults to baseline.star. Re-picking with a fine-tuned checkpoint needs a
# different --out-name, or it overwrites the base checkpoint's candidates.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

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
    -h|--help)    sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ENTRY" ] || { echo "error: --entry is required" >&2; exit 2; }
valid_entry "$ENTRY"
case "$SETTING" in annot|full) ;; *) echo "error: --setting is annot or full" >&2; exit 2 ;; esac

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

[ -n "$CKPT" ] || CKPT="$DATA/checkpoints/CryoTransformer_head_repaired.pth"
[ -f "$CKPT" ] || { echo "error: no checkpoint at $CKPT." >&2
                    echo "       Run scripts/01_download_data.sh --intermediates, or scripts/02_repair_head.sh." >&2
                    exit 1; }

case "$SETTING" in
  annot) MICS="$DATA/cryoppp/$ENTRY/micrographs" ;;
  full)  MICS="$DATA/cryoppp_fullset/$ENTRY/micrographs" ;;
esac
[ -d "$MICS" ] || { echo "error: no micrographs at $MICS. Run scripts/01_download_data.sh." >&2; exit 1; }

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
PRED_DIR="$(ls -dt "$CT"/output/predictions/predictions_EMPIAR_"$ENTRY"_remarks_"$REMARKS"_timestamp_* 2>/dev/null | head -1)"
[ -n "$PRED_DIR" ] || { echo "error: the picker wrote no output directory under $CT/output/predictions/." >&2; exit 1; }
STAR="$(ls "$PRED_DIR"/*star_file.star 2>/dev/null | head -1)"
[ -n "$STAR" ] || { echo "error: no combined STAR in $PRED_DIR." >&2; exit 1; }
cp "$STAR" "$OUT_DIR/$OUT_NAME.star"

echo
echo "Picks:  $OUT_DIR/$OUT_NAME.star  ($(grep -c . "$OUT_DIR/$OUT_NAME.star") lines)"
echo "Source: $PRED_DIR"
# The masking stage names its output after the condition that consumes it, which
# is `mask` for the base checkpoint's picks and `fb` for the round-1 one's.
if [ "$OUT_NAME" = "baseline" ]; then
  echo "Next:   bash scripts/04_mask.sh --entry $ENTRY"
else
  echo "Next:   bash scripts/04_mask.sh --entry $ENTRY \\"
  echo "          --star $OUT_DIR/$OUT_NAME.star --out-name ${OUT_NAME%_raw}"
fi
