"""The panel strip every picker-feedback overlay is built from.

One micrograph, downscaled, with boxes drawn on it and a black header naming the panel and
counting its boxes -- repeated left to right and stacked into one image. Shared so the
per-round stage figure and the across-rounds pick figure are the same object seen from two
angles rather than two figures that happen to look similar.

The header carries a swatch and a count per colour used in the panel, which is what lets
these images drop their captions: the legend is in the panel that needs it.
"""
from __future__ import annotations

import cv2
import numpy as np


def parse_crop(spec):
    """'x,y,w,h' (mrc pixels, top-left origin) -> (x, y, w, h). None passes through."""
    if not spec:
        return None
    try:
        x, y, w, h = (int(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(f"--crop expects 'x,y,w,h' in mrc pixels, got: {spec}")
    return x, y, w, h


def crop_view(gray, points, crop):
    """Restrict image and points to the crop window; coordinates become crop-relative."""
    if crop is None:
        return gray, points
    x, y, w, h = crop
    h_img, w_img = gray.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"--crop {crop} lies outside the {w_img}x{h_img} micrograph")
    inside = [(px - x0, py - y0, c) for px, py, c in points if x0 <= px < x1 and y0 <= py < y1]
    return gray[y0:y1, x0:x1], inside


def header_bar(width, label, points):
    """Black strip naming the panel, with one swatch and count per colour used in it."""
    bar_h = max(34, int(width * 0.062))
    bar = np.zeros((bar_h, width, 3), np.uint8)
    pad = int(bar_h * 0.22)

    per_color = {}
    for _x, _y, color in points:
        per_color[color] = per_color.get(color, 0) + 1

    cv2.putText(bar, f"{label}   n={len(points):,}", (pad, int(bar_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, bar_h / 46.0, (255, 255, 255), 2, cv2.LINE_AA)
    if len(per_color) < 2:
        return bar

    # Right-aligned tallies, each behind its own swatch, so a multi-colour panel reads as
    # "how many of each" without a separate legend. A one-colour panel would repeat n.
    x = width - pad
    for color, count in sorted(per_color.items(), key=lambda kv: -kv[1]):
        text = f"{count:,}"
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, bar_h / 46.0, 2)
        x -= text_w
        cv2.putText(bar, text, (x, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    bar_h / 46.0, color, 2, cv2.LINE_AA)
        swatch = bar_h - 2 * pad
        x -= swatch + pad
        cv2.rectangle(bar, (x, pad), (x + swatch, bar_h - pad), color, -1)
        x -= pad
    return bar


def render_panel(gray, points, box_px, panel_width, label):
    """One micrograph with its boxes, downscaled to panel_width, header on top.

    Boxes are drawn after the resize; drawing at full resolution and then shrinking thins
    the strokes away.
    """
    h, w = gray.shape
    scale = panel_width / float(w)
    img = cv2.resize(gray, (panel_width, max(1, int(round(h * scale)))),
                     interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    r = max(1, int(round(box_px * scale / 2)))
    thickness = max(1, int(round(box_px * scale / 14)))
    for x, y, color in points:
        cx, cy = int(round(x * scale)), int(round(y * scale))
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness)

    return np.vstack([header_bar(panel_width, label, points), img])


def render_strip(jpg_path, panels, box_px, panel_width, crop, out_path, quality=82):
    """panels = [(label, [(x, y, colour), ...]), ...] in the order they are laid out.

    There is no caption strip: each panel's header names it and counts its boxes, and the
    arm, round and id are in the output path (see fb_paths).
    """
    gray = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"    skip (unreadable jpg): {jpg_path}")
        return False

    drawn = []
    for label, points in panels:
        view, inside = crop_view(gray, points, crop)
        drawn.append(render_panel(view, inside, box_px, panel_width, label))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = ([cv2.IMWRITE_PNG_COMPRESSION, 6] if out_path.suffix.lower() == ".png"
              else [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(str(out_path), np.hstack(drawn), params)
    print(f"    -> {out_path.name}")
    return True
