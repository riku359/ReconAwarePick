#!/usr/bin/env python3
"""overlay_panel.py -- the background denoise the contamination overlays draw on.

The overlay itself -- denoised background + mask in red alpha + threshold contour
(yellow) + kept/removed circles -- is rendered by
`filter_star_by_contamination.render_validation`, which is the only renderer any
script in this release calls. What lives here is the piece the renderer cannot
reimplement without drifting: the background denoise.

It calls the same chain of operations as CryoSegNet's own denoise(), so a
micrograph drawn from CryoSegNet's released denoised JPG and one denoised on the
spot look the same. The chain itself is vendored in denoise_pipeline.py (the
CryoSegNet picker is not part of this release).
"""
import numpy as np

import denoise_pipeline as dp


def denoise_flip_frame(mic):
    """mrc-native array -> denoised uint8 in the flipped frame.

    The chain **calls exactly the same functions** as CryoSegNet's own denoise()
    (only the mrc read is left out):
    `.T`->`rot90` flip -> standard_scaler -> NlMeans -> wiener(K=30) -> CLAHE -> guided.
    It is the fallback for a micrograph with no pre-computed denoised JPG, but the
    result goes through the same processing as the JPG.
    """
    image = np.asarray(mic, dtype=np.float32).T                      # same order as the original denoise()
    image = np.rot90(image)
    normalized = dp.standard_scaler(np.array(image))
    contrast = dp.contrast_enhancement(normalized)
    wiener = dp.wiener_filter(contrast, dp.KERNEL.copy(), K=30)
    clahe_image = dp.clahe(wiener)
    return dp.guided_filter(clahe_image, wiener)
