#!/usr/bin/env bash
# Discard the picks that fall on contamination (Sec. 3.3, Sec. S3).
#
# Contamination such as carbon film, ice and aggregates carries more contrast than
# the particles, so candidates taken from it would otherwise reach 2D
# classification and survive there as coherent false-positive classes. A candidate
# is discarded when MicrographCleaner's predicted mask, resized to full micrograph
# resolution, reaches probability 0.5 at the candidate's centre.
#
# The mask is assembled from the network's per-window predictions with triangular
# blending rather than the released uniform averaging: windows overlap by half
# their size and the weights fall linearly from one at the window centre to zero at
# its border, so adjacent windows hand over smoothly and no seam forms. The
# released assembly repairs its seams afterwards with fixJumpInBorders, which
# cannot tell a seam from a steep intensity change in the micrograph itself and
# floods a rectangular region of the mask when it misreads one.
#
#   bash scripts/04_mask.sh --entry 10081                    full deposition
#   bash scripts/04_mask.sh --entry 10081 --setting annot
#   bash scripts/04_mask.sh --entry 10081 --masks-only       just precompute masks
#
# Reads  $RAPICK_WORK/picks/<entry>/baseline.star
# Writes $RAPICK_WORK/masks/<entry>/*.npz and $RAPICK_WORK/picks/<entry>/mask.star
#
# If you downloaded the published masks (scripts/01_download_data.sh
# --intermediates) the inference pass is skipped and only the filter runs.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ENTRY=""
SETTING="full"
SOURCE_STAR=""
OUT_NAME="mask"
MASKS_ONLY=0
GPU="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)      ENTRY="$2"; shift 2 ;;
    --setting)    SETTING="$2"; shift 2 ;;
    --star)       SOURCE_STAR="$2"; shift 2 ;;
    --out-name)   OUT_NAME="$2"; shift 2 ;;
    --masks-only) MASKS_ONLY=1; shift ;;
    --gpu)        GPU="$2"; shift 2 ;;
    -h|--help)    sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ENTRY" ] || { echo "error: --entry is required" >&2; exit 2; }
valid_entry "$ENTRY"
case "$SETTING" in annot|full) ;; *) echo "error: --setting is annot or full" >&2; exit 2 ;; esac

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

PY="$(venv_python micrograph_cleaner)"
CL="$REPO/src/rapick/cleaner"
MASK_DIR="$WORK/masks/$ENTRY"
PICKS_DIR="$WORK/picks/$ENTRY"

case "$SETTING" in
  annot) MIC_ROOT="$DATA/cryoppp" ;;
  full)  MIC_ROOT="$DATA/cryoppp_fullset" ;;
esac

# The annotated 300 are a strict subset of the full deposition and share their
# filenames, so one mask store serves both scales.
have=$(ls "$MASK_DIR"/*.npz 2>/dev/null | wc -l | tr -d ' ')
if [ "$have" -gt 0 ]; then
  banner "Reusing $have cached masks in $MASK_DIR"
else
  require_upstream micrograph_cleaner_em "MicrographCleaner"
  banner "Predicting contamination masks for $ENTRY ($SETTING)"
  "$PY" "$CL/save_fullset_triangular_masks.py" \
      --ids "$ENTRY" --gpu "$GPU" --mic-root "$MIC_ROOT" --out-root "$WORK/masks"
fi

if [ "$MASKS_ONLY" -eq 1 ]; then
  echo; echo "Masks under $MASK_DIR"; exit 0
fi

[ -n "$SOURCE_STAR" ] || SOURCE_STAR="$PICKS_DIR/baseline.star"
[ -f "$SOURCE_STAR" ] || { echo "error: no picks at $SOURCE_STAR." >&2
                           echo "       Run: bash scripts/03_pick.sh --entry $ENTRY --setting $SETTING" >&2
                           exit 1; }

banner "Filtering $(basename "$SOURCE_STAR") against the masks"
# Applying a cached mask needs only numpy and opencv, so this half runs even
# without MicrographCleaner's TensorFlow environment built.
"$PY" "$CL/filter_star_from_masks.py" \
    --star "$SOURCE_STAR" --empiar-id "$ENTRY" \
    --mask-dir "$MASK_DIR" --out-dir "$PICKS_DIR"

# The filter names its output by its own convention; the reconstruction configs
# key picks by condition name, so publish it under that.
PRODUCED="$PICKS_DIR/cryotransformer_clean_tri.star"
[ -f "$PRODUCED" ] || { echo "error: the filter wrote no $PRODUCED" >&2; exit 1; }
cp "$PRODUCED" "$PICKS_DIR/$OUT_NAME.star"

echo
echo "Masked picks: $PICKS_DIR/$OUT_NAME.star"
echo "Removed:      $PICKS_DIR/cryotransformer_removed_tri.star"
echo "Per-micrograph counts: $PICKS_DIR/filter_stats_tri.csv"
echo "Next: bash scripts/05_select2d.sh --entry $ENTRY --condition both"
