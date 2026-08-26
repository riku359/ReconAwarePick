#!/usr/bin/env bash
# Download everything the pipeline reads, into $RAPICK_DATA.
#
# Nothing is written into this repository. All four entries at full-set scale come
# to about 1.6 TB, so run --dry-run first: it enumerates what would be fetched and
# checks that the disk has the space.
#
# Usage:
#   bash scripts/01_download_data.sh --dry-run
#   bash scripts/01_download_data.sh                      all four entries
#   bash scripts/01_download_data.sh --entry 10081        one entry
#   bash scripts/01_download_data.sh --annot-only         skip the full depositions
#   bash scripts/01_download_data.sh --intermediates      add the published artifacts
#   bash scripts/01_download_data.sh --intermediates --picks
#
# --intermediates fetches theta_0 and the contamination masks from Hugging Face,
# so the head repair and the masking stage can be skipped. --picks additionally
# fetches the four pickers' picks, so Table 2 and Table S2 can be reproduced
# without installing crYOLO, Topaz or CryoSegNet.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${RAPICK_DATA:?set RAPICK_DATA to the directory the inputs live in (see docs/CONFIGURATION.md)}"
WORK="${RAPICK_WORK:?set RAPICK_WORK to the directory the pipeline writes to (see docs/CONFIGURATION.md)}"

ENTRIES=(10081 10093 10345 10532)
DRY_RUN=""
ANNOT_ONLY=0
INTERMEDIATES=0
PICKS=0
WORKERS=4

while [ $# -gt 0 ]; do
  case "$1" in
    --entry)          ENTRIES=("$2"); shift 2 ;;
    --dry-run)        DRY_RUN="--dry-run"; shift ;;
    --annot-only)     ANNOT_ONLY=1; shift ;;
    --intermediates)  INTERMEDIATES=1; shift ;;
    --picks)          PICKS=1; shift ;;
    --workers)        WORKERS="$2"; shift 2 ;;
    -h|--help)        sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

PY="$REPO/envs/figures/.venv/bin/python3"
[ -x "$PY" ] || PY=python3
DL="$REPO/src/rapick/data"

# The downloaders read the CryoPPP catalogue spreadsheet, so they need openpyxl.
# Run them through uv rather than requiring it in the ambient interpreter.
run_dl() { uv run --quiet --with openpyxl python3 "$@"; }

echo "==> Data root:  $DATA"
echo "==> Work root:  $WORK"
echo "==> Entries:    ${ENTRIES[*]}"
mkdir -p "$DATA" "$WORK"

# --- the CryoPPP catalogue ---------------------------------------------------
CATALOG="$DATA/cryoppp_tools/cryoppp"
if [ ! -d "$CATALOG/.git" ]; then
  echo "==> Cloning the CryoPPP catalogue"
  mkdir -p "$(dirname "$CATALOG")"
  git clone --quiet --depth 1 https://github.com/BioinfoMachineLearning/cryoppp.git "$CATALOG"
else
  echo "==> CryoPPP catalogue already present"
fi

# --- annotated micrographs and the expert annotations ------------------------
echo "==> Annotated micrographs (300 per entry)"
run_dl "$DL/download_cryoppp_micrographs_only.py" \
    --data-root "$DATA" --ids "${ENTRIES[@]}" --workers "$WORKERS" --max-retries 5 $DRY_RUN

echo "==> Expert annotations"
run_dl "$DL/download_cryoppp_star.py" \
    --data-root "$DATA" --ids "$(IFS=,; echo "${ENTRIES[*]}")" --max-retries 5 $DRY_RUN

# --- the full depositions ----------------------------------------------------
if [ "$ANNOT_ONLY" -eq 0 ]; then
  echo "==> Full depositions (997 / 1,873 / 1,644 / 1,556 micrographs)"
  if [ -n "$DRY_RUN" ]; then
    run_dl "$DL/download_empiar_fullset.py" --data-root "$DATA" --ids "${ENTRIES[@]}" --list-only
  else
    run_dl "$DL/download_empiar_fullset.py" \
        --data-root "$DATA" --ids "${ENTRIES[@]}" --workers "$WORKERS" --max-retries 5
  fi
else
  echo "==> Skipping the full depositions (--annot-only)"
fi

# --- recovery and verification ----------------------------------------------
# A download can fail in ways an existence check does not catch: two workers
# appending to one .part file, or EBI's S3 endpoint returning an XML error body
# that lands inside the .mrc with a plausible size and only fails hours later at
# Patch CTF. Both are worth catching before a run, not during one.
if [ -z "$DRY_RUN" ]; then
  echo "==> Recovering anything that failed"
  run_dl "$DL/recover_failed_mrc_from_targz.py" --data-root "$DATA" --max-retries 5 || \
    echo "  (nothing to recover, or recovery reported failures; see the log under cryoppp_tools/)"

  echo "==> Verifying micrograph integrity"
  for id in "${ENTRIES[@]}"; do
    run_dl "$DL/verify_mrc_integrity.py" --data-root "$DATA" --dataset cryoppp --ids "$id" || true
    [ "$ANNOT_ONLY" -eq 0 ] && \
      run_dl "$DL/verify_mrc_integrity.py" --data-root "$DATA" --dataset fullset --ids "$id" || true
  done
fi

# --- published artifacts -----------------------------------------------------
if [ "$INTERMEDIATES" -eq 1 ]; then
  echo "==> Published artifacts from Hugging Face"
  HF_ARGS=(--repo-weights rikrikrik/recon-aware-pick-weights
           --repo-data    rikrikrik/recon-aware-pick-data
           --data-root    "$DATA"
           --experiments-root "$WORK"
           --ids "${ENTRIES[@]}"
           --with-masks)
  uv run --quiet --with huggingface_hub python3 "$DL/hf_assets.py" download "${HF_ARGS[@]}"
  echo "  theta_0        -> $DATA/checkpoints/CryoTransformer_head_repaired.pth"
  echo "  masks          -> $WORK/masks/<id>/"
  echo "  masked picks   -> $WORK/picks/<id>/"
fi

if [ "$PICKS" -eq 1 ]; then
  echo "==> Picks of the four pickers"
  echo "  NOT YET PUBLISHED. This is on the repository's TODO list: the four"
  echo "  pickers' full-set picks have to be uploaded to"
  echo "  rikrikrik/recon-aware-pick-data before --picks does anything."
  echo "  Until then, reproduce Table 2 and Table S2 by installing the pickers"
  echo "  (docs/BASELINES.md) or read the numbers from results/tables/."
fi

# --- what to check before starting a run -------------------------------------
# import_particles in CryoSPARC dies on the first missing micrograph, and a *.mrc
# glob happily imports a partial download, so the count is worth confirming here
# rather than three hours into an extraction.
if [ -z "$DRY_RUN" ]; then
  echo
  echo "==> Micrograph counts"
  declare -A EXPECTED=([10081]=997 [10093]=1873 [10345]=1644 [10532]=1556)
  for id in "${ENTRIES[@]}"; do
    a=$(find "$DATA/cryoppp/$id/micrographs" -name '*.mrc' 2>/dev/null | wc -l | tr -d ' ')
    printf "  %s  annotated %4s / 300" "$id" "$a"
    if [ "$ANNOT_ONLY" -eq 0 ]; then
      f=$(find "$DATA/cryoppp_fullset/$id/micrographs" -name '*.mrc' 2>/dev/null | wc -l | tr -d ' ')
      printf "   full %5s / %s" "$f" "${EXPECTED[$id]}"
    fi
    echo
  done
fi

echo
echo "Done."
