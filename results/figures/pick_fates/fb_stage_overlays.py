#!/usr/bin/env python3
"""Draw what survives each stage of a feedback round, on the micrograph itself.

The loop reports what each filter removed as a fraction. Those numbers say how much
went, never what. This renders the funnel on the micrograph the model actually read, so
"the contamination mask takes ice, the 2D selection takes non-particles on clean
background" is something to check rather than to assume.

Every raw pick lands in exactly one of five fates, which nest as set operations on
(micrograph, x, y). `src/rapick/loop/fb_export_stage_stars.py` writes the three STAR
files that only CryoSPARC can produce and proves the nesting:

    mask removed      its centre fell on the triangular-blend contamination mask
    edge dropped      extraction refused it: the box would cross the micrograph edge
    class_2D rejected class_2D itself did not accept it (`particles_rejected`)
    selection removed it was classified, but its class did not survive the iterative
                      2D class selection
    survived          it is in the final select_2D

Panels per micrograph, left to right, all on the same denoised jpg. Each one draws what
is still in play once that stage has had its say, so the strip narrows from the picker's
raw output to the teacher labels and every panel is a set the next one is drawn from:

    raw picks               magenta: everything the picker proposed this round
    after cleaner mask      cyan: what contamination detection let through
    after 2D classification orange: what extraction took and class_2D gave a class to
    after 2D selection      green: what reached the teacher labels
    GT                      green: the CryoPPP annotations, to read the survivors against

Consecutive panel counts differ by exactly what the stage between them removed, so the
selection's own share is read off the last two panels and is not inflated by the
extraction edge and the class_2D rejects, which the panel before it has already taken
out. Those two are not the selection's doing: together they are 6,243 particles against
the selection's 21,192 on EMPIAR-10081 round 0.

The default micrograph choice comes from the annotated particle count alone, so the SAME
micrographs are drawn for every round and every arm and the images can be flipped
through round by round. `--select mask` / `--select selection` instead ranks by what
that stage discarded in this round.

Reads `$RAPICK_WORK/loop/<id>/round<n>/` and `$RAPICK_WORK/denoised/<id>/`; writes under
`$RAPICK_FIGURES_OUT/stage_overlays/`. No CryoSPARC connection: the three STAR files the
export stage leaves behind are read from disk.

    python fb_stage_overlays.py --id 10081 --rounds 1 --mic HCN1apo_0343_2xaligned \\
        --panel-width 640 --quality 92

The image carries no caption, so its path is what identifies it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import fb_panels                                       # noqa: E402  the shared panel strip
import fb_paths                                        # noqa: E402  loop and figure layout
import fb_stages                                       # noqa: E402  per-stage set arithmetic
import figure_paths                                    # noqa: E402

figure_paths.add_src_to_path()

from rapick.loop.star import load_star_points, normalize_mic_name   # noqa: E402

GT_COLOR = fb_stages.COLORS["green"]     # same green as the survivors, since comparing
                                         # those two panels is the point

# The survivor panels after `raw picks`, in the order the funnel narrows. The name on
# the right is what the panel header says; the colours are fb_stages.KEPT_COLORS.
KEPT_PANELS = (("mask", "after cleaner mask"),
               ("class2d", "after 2D classification"),
               ("select", "after 2D selection"))

ROUNDS = range(6)


def select_mics(requested, gt, stages, backgrounds, n, mode):
    """Which micrographs to draw.

    'density' ranks by annotated particle count alone, so it returns the same
    micrographs for every round and arm and the rounds can be compared side by side. The
    other two rank by what this round discarded and therefore move between rounds: use
    them to find the failure, not to compare.
    """
    if requested:
        keys = [normalize_mic_name(m) for m in requested]
        unknown = [m for m, k in zip(requested, keys) if k not in backgrounds]
        if unknown:
            print(f"    warn: no denoised jpg for {unknown}", file=sys.stderr)
        return [k for k in keys if k in backgrounds]

    have = sorted(backgrounds)
    if mode in ("mask", "selection"):
        fate = "mask" if mode == "mask" else "select"
        if stages[fate]:
            removed = fb_stages.by_mic(stages[fate])
            have.sort(key=lambda m: -len(removed.get(m, ())))
            return have[:n]
        print(f"    note: this run has no {mode} stage; falling back to the "
              f"annotation-density choice")

    have.sort(key=lambda m: len(gt.get(m, ())))
    if n >= len(have):
        return have
    picked = {int(round(i * (len(have) - 1) / (n - 1))) for i in range(n)} if n > 1 else {0}
    return [have[i] for i in sorted(picked)]


def draw_run(eid, arm, n, args, gt, backgrounds, diam, out_root, overrides):
    round_dir = fb_paths.round_dir(eid, n, arm)
    if not round_dir.is_dir():
        print(f"  {arm} r{n}: no such dir, skipped ({round_dir})")
        return

    stages, kept, missing = fb_stages.load_stages(round_dir, overrides)
    per_mic = {stage: fb_stages.by_mic(keys) for stage, keys in kept.items()}
    totals = {fate: len(keys) for fate, keys in stages.items()}
    n_picks = sum(totals.values())

    note = f"   (no {', '.join(missing)})" if missing else ""
    print(f"  {arm} r{n}: picks {n_picks:,}  mask -{totals['mask']:,}  "
          f"edge -{totals['edge']:,}  class_2D -{totals['class2d']:,}  "
          f"selection -{totals['select']:,}  survived {totals['survived']:,}{note}")

    mics = select_mics(args.mic, gt, stages, backgrounds, args.n, args.select)
    crop = fb_panels.parse_crop(args.crop)
    suffix = "_crop" if crop else ""
    out_dir = fb_paths.round_out_dir(arm, eid, n, out_root, kind="stage")

    def boxes(stage, mic):
        color = fb_stages.KEPT_COLORS[stage]
        return [(x, y, color) for x, y in per_mic[stage].get(mic, [])]

    # A stage whose STAR never arrived would draw a panel identical to the one before
    # it, which reads as "this stage removed nothing" rather than "this stage has not
    # run".
    drawn_stages = [(stage, label) for stage, label in KEPT_PANELS if stage not in missing]

    for mic in mics:
        panels = [(f"raw picks r{n}", boxes("picks", mic))]
        panels += [(label, boxes(stage, mic)) for stage, label in drawn_stages]
        panels.append(("GT", [(x, y, GT_COLOR) for x, y in gt.get(mic, [])]))

        fb_panels.render_strip(backgrounds[mic], panels, diam, args.panel_width, crop,
                               out_dir / f"stage_{mic}{suffix}{args.ext}", args.quality)


def parse_rounds(spec):
    if "-" in spec:
        first, last = (int(v) for v in spec.split("-", 1))
        return list(range(first, last + 1))
    return [int(v) for v in spec.split(",")]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="10081", help="EMPIAR entry")
    ap.add_argument("--arm", nargs="+", choices=fb_paths.arms(),
                    default=[fb_paths.default_arm()],
                    help="which loop arm; `fb` is the paper's method")
    ap.add_argument("--rounds", default="0-5", help="'0-5' or '0,3,5'")
    ap.add_argument("--images-dir", default=None,
                    help="denoised jpg cache (default $RAPICK_WORK/denoised/<id>)")
    ap.add_argument("--mic", nargs="+", default=None, help="micrographs to draw (stem match)")
    ap.add_argument("--n", type=int, default=4, help="how many micrographs when none named")
    ap.add_argument("--select", choices=("density", "mask", "selection"), default="density",
                    help="annotation-count spread (the same micrographs every round), or "
                         "the ones that stage discarded most of in this round")
    ap.add_argument("--crop", default=None, help="zoom window 'x,y,w,h' in mrc pixels")
    ap.add_argument("--panel-width", type=int, default=640, help="px per panel")
    ap.add_argument("--ext", choices=(".jpg", ".png"), default=".jpg",
                    help="png is lossless and costs about 18%% more disk here (the "
                         "denoised background is smooth, so it compresses well either way)")
    ap.add_argument("--quality", type=int, default=82,
                    help="JPEG quality (ignored for png). Raise it for a figure")
    ap.add_argument("--out-dir", default=None,
                    help="default $RAPICK_FIGURES_OUT/stage_overlays; the tree below it "
                         "is <arm>/round<n>/<id>/stage")
    ap.add_argument("--diam", type=float, default=None,
                    help="box size in px (default: this entry's registered diameter)")
    ap.add_argument("--stage-star", action="append", default=[], metavar="FATE=FILENAME",
                    help="override the STAR one stage is read from, e.g. mask=cleaned.star")
    return ap.parse_args()


def main():
    args = parse_args()
    eid = str(args.id)
    diam = args.diam if args.diam is not None else fb_paths.diameter_px(eid)
    overrides = fb_stages.parse_star_overrides(args.stage_star)

    images_dir = (Path(args.images_dir).expanduser() if args.images_dir
                  else fb_paths.denoised_dir(eid))
    backgrounds = {normalize_mic_name(str(p)): p for p in sorted(images_dir.glob("*.jpg"))}
    if not backgrounds:
        raise SystemExit(f"no background jpg under {images_dir}")

    gt = load_star_points(fb_paths.gt_star(eid))
    out_root = fb_paths.out_root(args.out_dir)

    print(f"EMPIAR-{eid}  boxes {diam:g} px  {len(backgrounds)} micrographs")
    for arm in args.arm:
        for n in parse_rounds(args.rounds):
            if n not in ROUNDS:
                continue
            draw_run(eid, arm, n, args, gt, backgrounds, diam, out_root, overrides)
    print(f"DONE -> {out_root}")


if __name__ == "__main__":
    main()
