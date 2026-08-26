#!/usr/bin/env bash
# Repair CryoTransformer's classification head and write theta_0 (Sec. S2).
#
# The released CryoTransformer weights carry a training defect that leaves the
# classification head uninformative: the distributed COCO annotations label every
# particle category_id=1, which collides with num_classes=1's no-object index, so
# the matched query's target lands on no-object and the head learns nothing. The
# repair discards that layer's released weights and refits it, alone, on the
# features the frozen detector produces for the 22 CryoPPP entries of the picker's
# training set.
#
# Every condition in the paper starts from the checkpoint this writes, and every
# round of the feedback loop restarts from it.
#
#   bash scripts/02_repair_head.sh                    do the repair
#   bash scripts/02_repair_head.sh --train-dir DIR    CryoPPP training split
#
# SKIP THIS unless you want to redo it: theta_0 is published, and
# `scripts/01_download_data.sh --intermediates` fetches it. Redoing it needs the
# 22-entry CryoPPP training split, which is far more data than the four test
# entries this repository otherwise uses.
#
# Details, including why the standardization is folded back into the retrained
# weights: src/rapick/picker/README.md.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TRAIN_DIR=""
EPOCHS=15          # Table S1. The CLI's own default is 25; the paper's runs used 15.
while [ $# -gt 0 ]; do
  case "$1" in
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --epochs)    EPOCHS="$2"; shift 2 ;;
    -h|--help)   sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# --help must work with nothing configured, so the roots are demanded only
# once the arguments are known to be valid.
require_roots

OUT="$DATA/checkpoints/CryoTransformer_head_repaired.pth"
if [ -f "$OUT" ]; then
  echo "theta_0 is already at $OUT."
  echo "Delete it first if you really mean to recompute it."
  exit 0
fi

BASE="$DATA/checkpoints/CryoTransformer_pretrained_model.pth"
[ -f "$BASE" ] || { echo "error: the released checkpoint is missing at $BASE." >&2
                    echo "       Run: bash scripts/01_download_data.sh" >&2; exit 1; }
[ -n "$TRAIN_DIR" ] || { echo "error: --train-dir is required: the repair fits on the 22 CryoPPP" >&2
                         echo "       training entries, which this repository does not download." >&2
                         echo "       Point it at <extracted>/train_val_test_data/train." >&2
                         exit 2; }

require_upstream cryotransformer "CryoTransformer"
PY="$(venv_python cryotransformer)"
CT="$THIRD_PARTY/cryotransformer"
HR="$WORK/head_repair"
mkdir -p "$HR"

banner "Mapping micrograph stems to their EMPIAR entry"
( cd "$CT" && "$PY" head_repair/build_train_stem_mapping.py \
    --train-dir "$TRAIN_DIR" --out-csv "$HR/stem_to_id.csv" )

banner "Dumping decoder features for the training entries"
( cd "$CT" && "$PY" predict.py --empiar all --dump_hs "$HR/hs_dumps" --resume "$BASE" )

banner "Grouping the per-micrograph dumps by entry"
( cd "$CT" && "$PY" head_repair/aggregate_hs_by_id.py \
    --dump-hs-dir "$HR/hs_dumps" --mapping-csv "$HR/stem_to_id.csv" --out-dir "$HR/by_id" )

banner "Leave-one-entry-out cross-validation over the head configurations"
( cd "$CT" && "$PY" head_repair/phase_d_train_heads.py --mode eos_sweep --epochs "$EPOCHS" \
    --hs-dumps-dir "$HR/by_id" --out-dir "$HR/cv" )

banner "Fitting the deployed head and writing it back into the checkpoint"
# No split is held out here: the deployed head is the one fit on every micrograph
# of the 22 entries. The weight and bias of the no-object class are set to zero so
# that inference reads the head exactly as before.
( cd "$CT" && "$PY" head_repair/phase_e_writeback.py \
    --arch linear --loss softmax --eos-coef 0.1 --epochs "$EPOCHS" \
    --hs-dumps-dir "$HR/by_id" \
    --checkpoint-in "$BASE" --checkpoint-out "$OUT" )

echo
echo "theta_0 written to $OUT"
