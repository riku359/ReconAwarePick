#!/usr/bin/env bash
# Download everything the pipeline reads, into $RAPICK_DATA.
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Download everything the pipeline reads, into $RAPICK_DATA.

Nothing is written into this repository. All four entries at full-set scale come
to about 1.6 TB, so run --dry-run first: it enumerates what would be fetched and
checks that the disk has the space.

Usage:
  bash scripts/01_download_data.sh --dry-run
  bash scripts/01_download_data.sh                      all four entries
  bash scripts/01_download_data.sh --entry 10081        one entry
  bash scripts/01_download_data.sh --annot-only         skip the full depositions
  bash scripts/01_download_data.sh --intermediates      add the published artifacts
  bash scripts/01_download_data.sh --intermediates --picks
  bash scripts/01_download_data.sh --fb-weights          the Ours checkpoints

--intermediates fetches theta_0 and the contamination masks from Hugging Face,
so the head repair and the masking stage can be skipped. --picks additionally
fetches the four pickers' picks, so Table 2 and Table S2 can be reproduced
without installing crYOLO, Topaz or CryoSegNet.
HELP
}

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# This script lives in scripts/, so the repository root is the directory above it.
REPO="$(cd "$(dirname "$0")/.." && pwd)"

ENTRIES=(10081 10093 10345 10532)
DRY_RUN=""
ANNOT_ONLY=0
INTERMEDIATES=0
PICKS=0
FB_WEIGHTS=0
WORKERS=4

while [ $# -gt 0 ]; do
  case "$1" in
    --entry)          ENTRIES=("$2"); shift 2 ;;
    --dry-run)        DRY_RUN="--dry-run"; shift ;;
    --annot-only)     ANNOT_ONLY=1; shift ;;
    --intermediates)  INTERMEDIATES=1; shift ;;
    --picks)          PICKS=1; shift ;;
    --fb-weights)     FB_WEIGHTS=1; shift ;;
    --workers)        WORKERS="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# --help must work with nothing configured, so the roots are demanded only once the
# arguments are known to be valid.
if [ -z "${RAPICK_DATA:-}" ]; then
  echo "error: RAPICK_DATA is not set." >&2
  echo "       Point it at the directory the inputs live in (docs/CONFIGURATION.md)." >&2
  exit 1
fi
if [ -z "${RAPICK_WORK:-}" ]; then
  echo "error: RAPICK_WORK is not set." >&2
  echo "       Point it at the directory the pipeline writes to (docs/CONFIGURATION.md)." >&2
  exit 1
fi
DATA="$RAPICK_DATA"
WORK="$RAPICK_WORK"

PY="$REPO/envs/figures/.venv/bin/python3"
if [ ! -x "$PY" ]; then
  PY=python3
fi
DL="$REPO/src/rapick/data"

# The downloaders read the CryoPPP catalogue spreadsheet, so they need openpyxl.
# Run them through uv rather than requiring it in the ambient interpreter.
run_dl() { uv run --quiet --with openpyxl python3 "$@"; }

# $DRY_RUN below is deliberately left unquoted: it is either empty, and then adds
# no argument at all, or the single word --dry-run.

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
# This downloader wants one comma-separated list where the others take repeated
# ids, so join them: the entries with spaces between, then spaces turned into commas.
ENTRIES_CSV="$(echo "${ENTRIES[*]}" | tr ' ' ',')"
run_dl "$DL/download_cryoppp_star.py" \
    --data-root "$DATA" --ids "$ENTRIES_CSV" --max-retries 5 $DRY_RUN

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
  # A failure here is not fatal: there may simply be nothing to recover.
  if ! run_dl "$DL/recover_failed_mrc_from_targz.py" --data-root "$DATA" --max-retries 5; then
    echo "  (nothing to recover, or recovery reported failures; see the log under cryoppp_tools/)"
  fi

  echo "==> Verifying micrograph integrity"
  # The verifier exits non-zero when it finds a bad file. It has already said which
  # one, so keep going and let the operator decide.
  for id in "${ENTRIES[@]}"; do
    run_dl "$DL/verify_mrc_integrity.py" --data-root "$DATA" --dataset cryoppp --ids "$id" || true
    if [ "$ANNOT_ONLY" -eq 0 ]; then
      run_dl "$DL/verify_mrc_integrity.py" --data-root "$DATA" --dataset fullset --ids "$id" || true
    fi
  done
fi

# --- the picker's released weights -------------------------------------------
# Needed to redo the head repair, which starts from them, and as the fallback if
# you would rather not download theta_0. The archive also carries the COCO DETR
# checkpoint the picker was initialised from; both are kept.
CKPT_DIR="$DATA/checkpoints"
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_DIR/CryoTransformer_pretrained_model.pth" ]; then
  echo "==> CryoTransformer's released weights already present"
elif [ -n "$DRY_RUN" ]; then
  echo "==> Would fetch CryoTransformer's released weights (about 3 GB)"
else
  echo "==> CryoTransformer's released weights"
  TARBALL="$CKPT_DIR/pretrained_model.tar.gz"
  curl -fSL --retry 5 --retry-delay 10 -C - \
      https://calla.rnet.missouri.edu/CryoTransformer/pretrained_model.tar.gz \
      -o "$TARBALL"
  # gzip -t first: a truncated transfer otherwise unpacks into a plausible-looking
  # tree and only fails when torch tries to load the checkpoint.
  gzip -t "$TARBALL"
  tar -xzf "$TARBALL" -C "$CKPT_DIR" --strip-components=1
  rm -f "$TARBALL"
  echo "  -> $CKPT_DIR/CryoTransformer_pretrained_model.pth"
fi

# --- published artifacts -----------------------------------------------------
if [ "$INTERMEDIATES" -eq 1 ]; then
  echo "==> Published artifacts from Hugging Face"
  uv run --quiet --with huggingface_hub python3 "$DL/hf_assets.py" download \
      --repo-weights rikrikrik/recon-aware-pick-weights \
      --repo-data rikrikrik/recon-aware-pick-data \
      --data-root "$DATA" \
      --experiments-root "$WORK" \
      --ids "${ENTRIES[@]}" \
      --with-masks
  echo "  theta_0        -> $DATA/checkpoints/CryoTransformer_head_repaired.pth"
  echo "  masks          -> $WORK/masks/<id>/"
  echo "  masked picks   -> $WORK/picks/<id>/"
fi

if [ "$PICKS" -eq 1 ]; then
  echo "==> Candidates of the four pickers"
  uv run --quiet --with huggingface_hub python3 "$DL/hf_assets.py" download \
      --repo-data rikrikrik/recon-aware-pick-data \
      --experiments-root "$WORK" --ids "${ENTRIES[@]}" --with-picks
  echo "  -> $WORK/picks/<id>/{baseline,cryolo,topaz,cryosegnet}.star"
  echo "  These are what Table 2 and Table S2 need, so neither needs crYOLO, Topaz"
  echo "  or CryoSegNet installed (docs/BASELINES.md)."
fi

if [ "$FB_WEIGHTS" -eq 1 ]; then
  echo "==> Round-1 fine-tuned checkpoints (the fb condition's weights)"
  uv run --quiet --with huggingface_hub python3 "$DL/hf_assets.py" download \
      --repo-weights rikrikrik/recon-aware-pick-weights \
      --data-root "$DATA" --ids "${ENTRIES[@]}" --with-loop-checkpoints fb
  echo "  -> $DATA/checkpoints/loop_fb_round1_empiar_<id>.pth  (about 870 MB each)"
  echo "  Point scripts/03_pick.sh at one with --checkpoint to re-pick as Ours does."
fi

# --- what to check before starting a run -------------------------------------
# import_particles in CryoSPARC dies on the first missing micrograph, and a *.mrc
# glob happily imports a partial download, so the count is worth confirming here
# rather than three hours into an extraction.
if [ -z "$DRY_RUN" ]; then
  echo
  echo "==> Micrograph counts"
  # A case rather than an associative array: macOS still ships bash 3.2.
  expected_full() {  # expected_full <id>
    case "$1" in
      10081) echo 997 ;;
      10093) echo 1873 ;;
      10345) echo 1644 ;;
      10532) echo 1556 ;;
      *)     echo "?" ;;
    esac
  }
  for id in "${ENTRIES[@]}"; do
    annotated=0
    if [ -d "$DATA/cryoppp/$id/micrographs" ]; then
      annotated=$(find "$DATA/cryoppp/$id/micrographs" -name '*.mrc' | wc -l | tr -d ' ')
    fi
    printf "  %s  annotated %4s / 300" "$id" "$annotated"
    if [ "$ANNOT_ONLY" -eq 0 ]; then
      full=0
      if [ -d "$DATA/cryoppp_fullset/$id/micrographs" ]; then
        full=$(find "$DATA/cryoppp_fullset/$id/micrographs" -name '*.mrc' | wc -l | tr -d ' ')
      fi
      printf "   full %5s / %s" "$full" "$(expected_full "$id")"
    fi
    echo
  done
fi

echo
echo "Done."
