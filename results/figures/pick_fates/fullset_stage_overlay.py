#!/usr/bin/env python3
"""Draw the raw-picks and after-mask panels for the full micrograph set.

`fb_stage_overlays.py` draws that strip for a feedback round, reading the round
directory the loop leaves behind. The full set is not a round and has no such directory,
so this walks the same code with the two STAR files the full-set arm does leave: the
picker's raw output and what survives the contamination mask. The panels come out in the
same format as the round strip, so `lib/prepare_overlay_panels.py` splits and crops both
the same way.

The strip is the supplementary protocol figure's picking and mask panels for the full
set. That figure is drawn in TikZ in the manuscript; the panels are cut from this strip
and from the round strip, so both blocks of it show the same field of view. The picks
differ because the subset is picked with the round's checkpoint and the full set with
the one the round delivers.

No CryoSPARC connection: the two STAR files are pipeline outputs read from disk. It does
need the full-set arm's outputs under `$RAPICK_WORK/loop/<id>/fullset/<tag>/` and the
denoised micrographs under `$RAPICK_WORK/denoised/<id>/`. The original ran on the server
that held them, over ssh with the strip returned base64-encoded; that transport is gone
and the script writes the strip itself.

    python fullset_stage_overlay.py --id 10081 --tag round1 \\
        --mic HCN1apo_0343_2xaligned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import fb_panels                                       # noqa: E402  the shared panel strip
import fb_paths                                        # noqa: E402  loop and figure layout
import fb_stages                                       # noqa: E402  the per-stage colours
import figure_paths                                    # noqa: E402

figure_paths.add_src_to_path()

from rapick.loop.star import load_star_points, normalize_mic_name   # noqa: E402

# The full-set arm writes these two next to each other. The first is everything the
# picker proposed over the whole set, the second what the contamination mask let
# through, so the two panels differ by exactly what the mask removed.
STARS = (("picks", "raw picks", "picks.star"),
         ("mask", "after cleaner mask", "masked.star"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="10081", help="EMPIAR entry")
    ap.add_argument("--tag", default="round1",
                    help="which full-set evaluation, under "
                         "$RAPICK_WORK/loop/<id>/fullset/<tag>/")
    ap.add_argument("--arm", default=fb_paths.default_arm(), choices=fb_paths.arms(),
                    help="which loop arm the checkpoint came from, for the output path")
    ap.add_argument("--mic", required=True, help="micrograph stem")
    ap.add_argument("--images-dir", default=None,
                    help="denoised jpg cache (default $RAPICK_WORK/denoised/<id>)")
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--diam", type=float, default=None,
                    help="box size in px (default: this entry's registered diameter)")
    ap.add_argument("--out-dir", default=None,
                    help="default $RAPICK_FIGURES_OUT/stage_overlays")
    ap.add_argument("--stage-star", action="append", default=[], metavar="FATE=FILENAME",
                    help="override the STAR one stage is read from, e.g. mask=cleaned.star")
    args = ap.parse_args()

    eid = str(args.id)
    diam = args.diam if args.diam is not None else fb_paths.diameter_px(eid)
    overrides = fb_stages.parse_star_overrides(args.stage_star)

    root = fb_paths.fullset_dir(eid, args.tag)
    if not root.is_dir():
        raise SystemExit(f"no full-set arm at {root}")

    images_dir = (Path(args.images_dir).expanduser() if args.images_dir
                  else fb_paths.denoised_dir(eid))
    backgrounds = {normalize_mic_name(str(p)): p for p in sorted(images_dir.glob("*.jpg"))}
    key = normalize_mic_name(args.mic)
    if key not in backgrounds:
        raise SystemExit(f"no denoised jpg for {args.mic} under {images_dir}")

    panels = []
    for stage, label, default_name in STARS:
        star = root / overrides.get(stage, default_name)
        if not star.is_file():
            raise SystemExit(f"missing {star}")
        points = load_star_points(star).get(key, [])
        color = fb_stages.KEPT_COLORS[stage]
        panels.append((label, [(x, y, color) for x, y in points]))
        print(f"{label}: {len(points)} boxes")

    out_dir = fb_paths.fullset_out_dir(args.arm, eid, args.out_dir, kind="stage")
    fb_panels.render_strip(backgrounds[key], panels, diam, args.panel_width, None,
                           out_dir / f"stage_{key}.jpg", args.quality)
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
