#!/usr/bin/env bash
# Discard the picks that fall on contamination (Sec. 3.3, Sec. S3).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Discard the picks that fall on contamination (Sec. 3.3, Sec. S3).

A candidate is discarded when MicrographCleaner's predicted mask, resized to full
micrograph resolution, reaches probability 0.5 at the candidate's centre. The
mask is assembled with triangular blending rather than the released uniform
averaging; what that fixes: src/rapick/cleaner/README.md.

  bash scripts/contamination_removal.sh --entry 10081
  bash scripts/contamination_removal.sh --entry 10081 --star $RAPICK_WORK/picks/10081/fb.star
  bash scripts/contamination_removal.sh --entry 10081 --masks-only

  --star   the picks to filter   (default: <picks dir>/cryotransformer.star)
  --masks  the .npz mask store   (default: $RAPICK_WORK/masks/<entry>)
  --out    where to write        (default: the input with _mask before .star)

The masks are per-micrograph and do not depend on the picks, so one store serves
every checkpoint and both scales. They are predicted into --masks the first time
and reused afterwards; scripts/download.sh puts the published ones there, and then
no GPU inference runs at all.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
SETTING="full"
SOURCE_STAR=""
MASK_DIR=""
OUT=""
MASKS_ONLY=0
GPU="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)      ENTRY="$2"; shift 2 ;;
    --setting)    SETTING="$2"; shift 2 ;;
    --star)       SOURCE_STAR="$2"; shift 2 ;;
    --masks)      MASK_DIR="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --masks-only) MASKS_ONLY=1; shift ;;
    --gpu)        GPU="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
require_entry "$ENTRY"
require_setting "$SETTING"

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

PICKS_DIR="$(picks_dir "$ENTRY")"
[ -n "$MASK_DIR" ]    || MASK_DIR="$(masks_dir "$ENTRY")"
[ -n "$SOURCE_STAR" ] || SOURCE_STAR="$PICKS_DIR/cryotransformer.star"
# The name of a STAR records which stages it has been through, so the default output
# is the input with one more stage appended: cryotransformer.star -> _mask.star,
# fb.star -> fb_mask.star.
[ -n "$OUT" ] || OUT="$(dirname "$SOURCE_STAR")/$(basename "$SOURCE_STAR" .star)_mask.star"

PY="$(venv_python micrograph_cleaner)"
CL="$REPO/src/rapick/cleaner"
MIC_ROOT="$(micrograph_root "$SETTING")"

# The annotated 300 are a strict subset of the full deposition and share their
# filenames, so one mask store serves both scales.
CACHED=0
if [ -d "$MASK_DIR" ]; then
  CACHED=$(find "$MASK_DIR" -name '*.npz' | wc -l | tr -d ' ')
fi
if [ "$CACHED" -gt 0 ]; then
  banner "Reusing $CACHED cached masks in $MASK_DIR"
else
  require_upstream micrograph_cleaner_em "MicrographCleaner"
  banner "Predicting contamination masks for $ENTRY ($SETTING) into $MASK_DIR"
  # The predictor lays out <out-root>/<entry>/, so it is handed the parent of the
  # store the filter below reads. Passing --masks a directory not named after the
  # entry would put the two halves of this stage in different places.
  if [ "$(basename "$MASK_DIR")" != "$ENTRY" ]; then
    echo "error: --masks must end in the entry id, so that the predictor and the" >&2
    echo "       filter agree on where the .npz go. Got $MASK_DIR" >&2
    exit 2
  fi
  "$PY" "$CL/save_fullset_triangular_masks.py" \
      --ids "$ENTRY" --gpu "$GPU" --mic-root "$MIC_ROOT" \
      --out-root "$(dirname "$MASK_DIR")"
fi

if [ "$MASKS_ONLY" -eq 1 ]; then
  echo
  echo "Masks under $MASK_DIR"
  exit 0
fi

if [ ! -f "$SOURCE_STAR" ]; then
  echo "error: no picks at $SOURCE_STAR." >&2
  echo "       Run: bash scripts/pick.sh --entry $ENTRY --setting $SETTING" >&2
  exit 1
fi
# Masking only ever removes picks, so writing the result over its own input would
# quietly destroy the unfiltered set and make the run unrepeatable.
if [ "$SOURCE_STAR" = "$OUT" ]; then
  echo "error: --star and --out name the same file, $SOURCE_STAR." >&2
  echo "       The filtered picks would overwrite the picks they were filtered from." >&2
  exit 2
fi

banner "Filtering $(basename "$SOURCE_STAR") against the masks"
# Applying a cached mask needs only numpy and opencv, so this half runs even
# without MicrographCleaner's TensorFlow environment built.
# Everything the filter writes lands beside --out, so a round of the loop filtered into
# its own directory does not overwrite the full-set run's counts.
OUT_DIR="$(dirname "$OUT")"
mkdir -p "$OUT_DIR"
"$PY" "$CL/filter_star_from_masks.py" \
    --star "$SOURCE_STAR" --empiar-id "$ENTRY" \
    --mask-dir "$MASK_DIR" --out-dir "$OUT_DIR"

# The filter names its output after itself; publish it under the name that says which
# stages the picks have been through. Renamed rather than copied, so there is never a
# second copy of the same STAR under two names for someone to read the stale one of.
PRODUCED="$OUT_DIR/cryotransformer_clean_tri.star"
if [ ! -f "$PRODUCED" ]; then
  echo "error: the filter wrote no $PRODUCED" >&2
  exit 1
fi
mv "$PRODUCED" "$OUT"

echo
echo "Masked picks: $OUT"
echo "Removed:      $OUT_DIR/cryotransformer_removed_tri.star"
echo "Per-micrograph counts: $OUT_DIR/filter_stats_tri.csv"
echo "Next: bash scripts/2d_classification.sh --entry $ENTRY --star $OUT"
