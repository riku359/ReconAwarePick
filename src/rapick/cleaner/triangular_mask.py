#!/usr/bin/env python3
"""triangular_mask.py -- contamination mask generation by triangular-window blending
(a replacement for the released `predictMask`).

MicrographCleaner's U-Net predicts on 256x256 windows. The released `predictMask`
averages the overlapping windows uniformly and finally evens out the steps at the
window borders with `fixJumpInBorders`. That post-processing smears the edge of a
carbon film into a band and so treats good particles in clean areas as contaminated,
so this module removes the seams with **overlap-add weighted by a triangular
(Bartlett) window** instead.

The `*_clean_tri.star` the reconstruction pipeline consumes is produced by this
implementation (filter_star_triangular.py). `extract_blended` returns the embedding
as well, so embedding analyses can share the same function.
"""
import numpy as np

MODEL_IMG_SIZE = 256            # one U-Net window; matches micrograph_cleaner_em's config.
EMBED_LAYER = "leaky_re_lu_20"  # just before the output conv (conv2d_30) = last-layer embedding (256,256,32)
OUTPUT_LAYER = "conv2d_30"      # the 0..1 contamination mask (256,256,1)


def build_extractor(model):
    """Build an extractor that emits (embedding, mask) together from MaskPredictor's keras model.

    The box size only decides the downsample on the preprocessMic side and has nothing
    to do with the model weights, so one extractor can be reused across EMPIAR ids.
    """
    import keras
    return keras.Model(model.inputs,
                       [model.get_layer(EMBED_LAYER).output,
                        model.get_layer(OUTPUT_LAYER).output])


def _tri_window(n):
    """Triangular (Bartlett) window: 0 at the ends, 1 at the centre. Overlap-added at 50%
    overlap the windows sum to a constant (partition of unity), so the seams at the window
    borders disappear."""
    r = (n - 1) / 2.0
    return 1.0 - np.abs((np.arange(n) - r) / r)


def extract_blended(extractor, mic_pre, stride_factor, pool, batch=16):
    """Extract embedding/mask with the seams corrected.

    Slide 256 windows with step=256/stride_factor so they overlap, weight each window by
    a triangular window and overlap-add them at full resolution, then divide by the sum
    of the weights (predictMask's return_as_oneMic, with triangular-window blending
    substituted). The seams (a cross pattern) that come from the different receptive-field
    context at a window border are removed by the smooth blend over the overlap. The
    outside is reflect-padded by one step as well, so the border is also covered by several
    windows. sf=1 means no overlap = the old behaviour (no correction).

    Returns: emb_grid(h8,w8,C), mask_grid(h8,w8), h8, w8 (h8=ceil(h'/pool)).
    """
    patch = MODEL_IMG_SIZE
    stride = max(1, patch // stride_factor)
    h0, w0 = mic_pre.shape
    margin = stride if stride_factor > 1 else 0
    pad = (np.pad(mic_pre, ((margin, margin), (margin, margin)), mode="reflect")
           if margin else mic_pre)
    extra_h = (-(pad.shape[0] - patch)) % stride         # pad bottom/right so window starts line up
    extra_w = (-(pad.shape[1] - patch)) % stride
    pad = np.pad(pad, ((0, extra_h), (0, extra_w)), mode="reflect")
    H, W = pad.shape

    tri = _tri_window(patch)
    win2d = (np.outer(tri, tri) + 1e-3)[..., None].astype(np.float32)   # (patch,patch,1)
    coords = [(y, x) for y in range(0, H - patch + 1, stride)
              for x in range(0, W - patch + 1, stride)]

    C = int(extractor.outputs[0].shape[-1])
    acc_e = np.zeros((H, W, C), np.float32)
    acc_m = np.zeros((H, W, 1), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    for c0 in range(0, len(coords), batch):              # accumulate in mini-batches to hold RAM down
        chunk = coords[c0:c0 + batch]
        wins = np.stack([pad[y:y + patch, x:x + patch] for (y, x) in chunk])[..., None]
        emb, mask = extractor.predict(wins.astype(np.float32), batch_size=8, verbose=0)
        for j, (y, x) in enumerate(chunk):
            acc_e[y:y + patch, x:x + patch] += emb[j] * win2d
            acc_m[y:y + patch, x:x + patch] += mask[j] * win2d
            wsum[y:y + patch, x:x + patch] += win2d
    emb_full = (acc_e / wsum)[margin:margin + h0, margin:margin + w0]   # drop the reflect margin
    mask_full = (acc_m / wsum)[margin:margin + h0, margin:margin + w0, 0]

    ph, pw = (-h0) % pool, (-w0) % pool                                # pad the edge to a multiple of pool
    emb_full = np.pad(emb_full, ((0, ph), (0, pw), (0, 0)), mode="edge")
    mask_full = np.pad(mask_full, ((0, ph), (0, pw)), mode="edge")
    h8, w8 = emb_full.shape[0] // pool, emb_full.shape[1] // pool
    emb_grid = emb_full.reshape(h8, pool, w8, pool, C).mean((1, 3))
    mask_grid = mask_full.reshape(h8, pool, w8, pool).mean((1, 3))
    return emb_grid, mask_grid, h8, w8
