#!/usr/bin/env bash
# CryoTransformer's released weights, about 3 GB. The head repair starts from them,
# and they are the fallback if you would rather not use the published theta_0. The
# archive also carries the COCO DETR checkpoint the picker was initialised from; both
# are kept.

. "$(dirname "$0")/_common.sh"

CKPT_DIR="$DATA/checkpoints"
MODEL="$CKPT_DIR/CryoTransformer_pretrained_model.pth"

banner "CryoTransformer's released weights (~3 GB)"
if [ -f "$MODEL" ]; then
  echo "  already present: $MODEL"
  exit 0
fi

mkdir -p "$CKPT_DIR"
TARBALL="$CKPT_DIR/pretrained_model.tar.gz"
curl -fSL --retry 5 --retry-delay 10 -C - \
    https://calla.rnet.missouri.edu/CryoTransformer/pretrained_model.tar.gz \
    -o "$TARBALL"
# gzip -t first: a truncated transfer otherwise unpacks into a plausible-looking tree
# and only fails when torch tries to load the checkpoint.
gzip -t "$TARBALL"
tar -xzf "$TARBALL" -C "$CKPT_DIR" --strip-components=1
rm -f "$TARBALL"
echo "  -> $MODEL"
