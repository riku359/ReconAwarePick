"""What each stage of one feedback round kept and what it threw away.

Every population is a set of (micrograph, x, y), so the stages are plain set
operations. The loop leaves its two picker-side stages on disk (the round's picks and
the contamination filter's survivors); `src/rapick/loop/fb_export_stage_stars.py` adds
the three that live inside CryoSPARC and proves that they nest:

    picks >= mask-kept >= extracted >= class_2D-accepted >= survivors

So each pick has exactly one fate, and the fates are differences of consecutive sets.
Two of them are not a filter's judgement and are kept separate for that reason: `edge`
is extraction refusing a box that would cross the micrograph border, and `class2d` is
class_2D's own `particles_rejected`. Folding either into the selector overstates it, by
6,243 particles against the selector's 21,192 on EMPIAR-10081 round 0.

Both views come out of one read: `kept` is the nesting above, `removed` its consecutive
differences. A caller that drew one from a second read could show a survivor panel and a
removal panel that disagree.

This module is shared so that the stage figure and any other per-round overlay
attribute a pick the same way; two copies of this arithmetic would eventually disagree.
"""
from __future__ import annotations

from pathlib import Path

import figure_paths

figure_paths.add_src_to_path()

from rapick.eval.vis_star_overlay import COLORS      # noqa: E402  repository palette
from rapick.loop.star import star_keys               # noqa: E402  GT-aligned STAR reader

# Drawing order matters: later fates are drawn on top, so the ones worth seeing go last.
FATE_COLORS = {
    "edge":     (150, 150, 150),         # extraction refused it at the micrograph edge
    "class2d":  COLORS["yellow"],        # class_2D itself rejected it
    "mask":     COLORS["red"],           # the contamination mask took it
    "select":   COLORS["orange"],        # the 2D class selection took it
    "survived": COLORS["green"],         # it reached the teacher labels
}
FATES = tuple(FATE_COLORS)

# Colours for the survivor view: what is still in play once that stage has had its say.
# The chain ends on the same green as the annotations, since reading the final survivors
# against them is what the figure is for. The earlier links get their own hue so that a
# panel whose header has been cropped off for print is still told apart from its
# neighbour.
KEPT_COLORS = {
    "picks":   COLORS["magenta"],        # everything the picker proposed this round
    "mask":    COLORS["cyan"],           # survived the triangular-blend contamination mask
    "class2d": COLORS["orange"],         # extraction took it and class_2D gave it a class
    "select":  COLORS["green"],          # in the final select_2D: the teacher labels
}

# The nesting above, outermost first: each entry is (fate name, the STAR that ends it).
# These are the file names the loop and its stage export write into a round directory.
# `--stage-star mask=<name>` on the overlay scripts overrides one, for a round directory
# whose contamination filter wrote its survivors under a different name.
CHAIN = (("mask", "masked.star"),
         ("edge", "extracted.star"),
         ("class2d", "class2d_accepted.star"),
         ("select", "survivors.star"))

PICKS_STAR = "picks.star"


def chain_with_overrides(overrides=None):
    """CHAIN with any `fate=filename` overrides applied, in the same order."""
    overrides = overrides or {}
    return tuple((fate, overrides.get(fate, star)) for fate, star in CHAIN)


def parse_star_overrides(items):
    """`["mask=cleaned.star"]` -> `{"mask": "cleaned.star"}`, checking the fate name."""
    known = {fate for fate, _ in CHAIN} | {"picks"}
    out = {}
    for item in items or ():
        fate, _, name = item.partition("=")
        if not name or fate not in known:
            raise SystemExit(f"--stage-star expects <fate>=<filename> with fate in "
                             f"{sorted(known)}, got: {item}")
        out[fate] = name
    return out


def load_keys(path: Path):
    """{(mic_key, x_int, y_int)} for a GT-aligned STAR, or None if it is not on disk."""
    if not Path(path).is_file():
        return None
    return star_keys(path)


def load_stages(round_dir: Path, star_overrides=None):
    """({fate: removed}, {stage: still in play}, [stages whose STAR was missing]).

    The second dict is keyed by "picks" plus every name in CHAIN, and holds the nesting
    itself: `kept["mask"]` is what the contamination mask let through, `kept["select"]`
    the final survivors.

    Only the picks STAR is required. A stage whose STAR is absent falls through to the
    previous one, so a half-processed round reads as "nothing was lost there" rather
    than "everything was", and the caller can say which panels it is entitled to draw.
    """
    round_dir = Path(round_dir)
    overrides = star_overrides or {}
    picks = load_keys(round_dir / overrides.get("picks", PICKS_STAR))
    if picks is None:
        raise SystemExit(f"no {overrides.get('picks', PICKS_STAR)} under {round_dir}")

    kept = {"picks": picks}
    missing = []
    still_in_play = picks
    for fate, star in chain_with_overrides(overrides):
        found = load_keys(round_dir / star)
        if found is None:
            missing.append(fate)
        else:
            still_in_play = found
        kept[fate] = still_in_play

    outer = picks
    stages = {}
    for fate, _star in CHAIN:
        stages[fate] = outer - kept[fate]
        outer = kept[fate]
    stages["survived"] = outer
    return stages, kept, missing


def by_mic(keys):
    """{(mic, x, y)} -> {mic: [(x, y), ...]}."""
    out = {}
    for mic, x, y in keys:
        out.setdefault(mic, []).append((x, y))
    return out


def colored_points(fates_for_mic):
    """{fate: [(x, y), ...]} -> [(x, y, colour), ...] in FATE_COLORS' drawing order."""
    return [(x, y, FATE_COLORS[fate])
            for fate in FATES for x, y in fates_for_mic.get(fate, ())]
