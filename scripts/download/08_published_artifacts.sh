#!/usr/bin/env bash
# The artifacts this project produced, from Hugging Face: theta_0, the contamination
# masks, and the picks after masking. With them, scripts/repair_head.sh and
# scripts/contamination_removal.sh can both be skipped.

. "$(dirname "$0")/_common.sh"

banner "Published artifacts (theta_0, masks, masked picks)"
run_hf download \
    --repo-weights rikrikrik/recon-aware-pick-weights \
    --repo-data rikrikrik/recon-aware-pick-data \
    --data-root "$DATA" \
    --experiments-root "$WORK" \
    --ids "$ENTRIES_CSV" \
    --with-cleaner-data --with-masks
echo "  theta_0        -> $DATA/checkpoints/CryoTransformer_head_repaired.pth"
echo "  masks          -> $WORK/masks/<id>/*_tri.npz"
echo "  masked picks   -> $WORK/picks/<id>/cryotransformer_mask.star"
