#!/usr/bin/env bash
# The four pickers' own full-set candidates, from Hugging Face. These are what Table 2
# and Table S2 need, so neither needs crYOLO, Topaz or CryoSegNet installed
# (docs/BASELINES.md).

. "$(dirname "$0")/_common.sh"

banner "Candidates of the four pickers"
run_hf download \
    --repo-data rikrikrik/recon-aware-pick-data \
    --experiments-root "$WORK" --ids "$ENTRIES_CSV" --with-picks
echo "  -> $WORK/picks/<id>/{cryotransformer,cryolo,topaz,cryosegnet}.star"
