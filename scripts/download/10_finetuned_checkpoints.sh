#!/usr/bin/env bash
# The round-1 fine-tuned checkpoints, one per entry, about 870 MB each. These are what
# the fb arm picks with, so with them the paper's headline row needs no loop run at all.

. "$(dirname "$0")/_common.sh"

banner "Round-1 fine-tuned checkpoints (~870 MB each)"
run_hf download \
    --repo-weights rikrikrik/recon-aware-pick-weights \
    --data-root "$DATA" --ids "$ENTRIES_CSV" --with-loop-checkpoints fb
echo "  -> $DATA/checkpoints/loop_fb_round1_empiar_<id>.pth"
echo "  Point scripts/pick.sh at one with --checkpoint to re-pick as Ours does."
