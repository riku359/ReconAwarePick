#!/usr/bin/env python3
"""overlay_panel.py -- the only overlay renderer for contamination masks.

Every MicrographCleaner overlay -- the released `predictMask` version, the
triangular-window version and the comparison galleries -- is read as the same
single picture:

  denoised background + mask in red alpha + threshold contour (yellow) + particles
  that did not land on contamination (green circle) / that did (red circle)
  + a header bar naming the mask method and the removed/kept counts

The background denoise calls the same chain of operations as CryoSegNet's own
denoise(), so a micrograph drawn from CryoSegNet's released denoised JPG and one
denoised on the spot look the same. The chain itself is vendored in
denoise_pipeline.py (the CryoSegNet picker is not part of this release).
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


def panel(image, mask, picks, flags, deep, box, max_out_dim, denoised, label, n_rm):
    """Return the overlay panel (header bar + image) for one micrograph, in BGR.

    mask is in the same mrc-native frame as image. It is downscaled to the output
    resolution and then flipped, so the caller needs no resize whether the mask is
    full-res (released) or at model scale (triangular). Denoises on the spot when
    denoised is None.
    """
    import cv2
    h, w = image.shape
    s = min(1.0, max_out_dim / float(max(h, w)))
    ow, oh = max(1, int(round(w*s))), max(1, int(round(h*s)))
    bg = (cv2.resize(denoised, (ow, oh), interpolation=cv2.INTER_AREA) if denoised is not None
          else denoise_flip_frame(cv2.resize(image, (ow, oh), interpolation=cv2.INTER_AREA)))
    bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR).astype(np.float32)
    m = np.flipud(cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_LINEAR)).clip(0, 1)
    a = (0.45 * m)[..., None]; red = np.zeros_like(bgr); red[..., 2] = 255.0
    img = (bgr * (1-a) + red * a).astype(np.uint8)
    cont, _ = cv2.findContours((m >= deep).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cont, -1, (0, 255, 255), 2)
    r = max(2, int(round(box*s/2)))
    for (x, y), keep in zip(picks, flags):
        cx, cy = int(round(x*s)), int(round(y*s))
        cv2.circle(img, (cx, cy), r, (0, 220, 0) if keep else (0, 0, 255), 3)
    bar_h = max(48, int(img.shape[1] * 0.030))          # a bar tall enough to read against the width
    fs = bar_h / 42.0                                    # font scale proportional to the bar height
    bar = np.zeros((bar_h, img.shape[1], 3), np.uint8)
    cv2.putText(bar, "%s    removed=%d (red)   kept=%d (green)" % (label, n_rm, sum(flags)),
                (14, int(bar_h*0.70)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])
