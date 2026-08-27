"""Where a loop overlay reads from and where it writes to.

Reading comes from `rapick.loop.entries`, which owns the loop's own layout
(`$RAPICK_WORK/loop/<id>/round<n>/`), so the figure code cannot drift from the loop it
draws. Writing is the figures' own tree, keyed the way the experiment is: arm, then
round, then EMPIAR entry, then which script drew it.

    <out>/<arm>/round<n>/<id>/stage/    the stage strip -> stage_<mic>.jpg
    <out>/<arm>/fullset/<id>/stage/     the full-set strip, which has no round number

The images carry no caption, so that path is what identifies one. Keeping the two
kinds of strip in separate directories matters for the same reason: mixed together,
`stage_x.jpg` from a loop round and `stage_x.jpg` from the full set look like a
before/after pair of one figure when they are two different measurements.
"""
from __future__ import annotations

from pathlib import Path

import figure_paths

figure_paths.add_src_to_path()

from rapick.loop import entries                       # noqa: E402
from rapick.loop import paths as paths_module          # noqa: E402


def arms():
    """The loop's arm names. `fb` is the paper's method and the default."""
    return sorted(entries.ARMS)


def default_arm():
    return entries.DEFAULT_ARM


def round_dir(empiar, n, arm=None):
    """The loop's own round directory: `$RAPICK_WORK/loop/<id>/round<n>/`."""
    return entries.round_dir(str(empiar), int(n), arm or entries.DEFAULT_ARM)


def fullset_dir(empiar, tag):
    """One full-set evaluation of one checkpoint: `$RAPICK_WORK/loop/<id>/fullset/<tag>/`."""
    return entries.fullset_dir(str(empiar), tag)


def diameter_px(empiar):
    """The particle diameter this repository scores and draws boxes at."""
    entry = entries.ENTRIES.get(str(empiar))
    if entry is None:
        raise SystemExit(f"EMPIAR-{empiar} is not one of the four entries "
                         f"({', '.join(sorted(entries.ENTRIES))}); pass --diam")
    return entry.diameter_px


def gt_star(empiar):
    """CryoPPP's annotation for one entry, in the GT-aligned convention."""
    return entries.ENTRIES[str(empiar)].gt_star


def denoised_dir(empiar):
    """The denoised jpgs an overlay is drawn on: `$RAPICK_WORK/denoised/<id>/`.

    The overlays are drawn on the denoised background rather than on the raw `.mrc`,
    because that is what the picker reads. One directory serves both settings: the 300
    annotated micrographs are a subset of the deposition, so a strip of the loop and a
    strip of the full set can be drawn on the same field of view.
    """
    return paths_module.denoised_dir(str(empiar))


def out_root(explicit=None):
    """Where the strips go: `--out-dir`, else `$RAPICK_FIGURES_OUT/stage_overlays`."""
    if explicit:
        return Path(explicit).expanduser()
    return figure_paths.figures_out("stage_overlays")


def round_out_dir(arm, empiar, n, root=None, kind="stage"):
    """Figures that belong to one round of one arm on one entry."""
    return out_root(root) / arm / f"round{n}" / str(empiar) / kind


def fullset_out_dir(arm, empiar, root=None, kind="stage"):
    """Figures drawn from a full-set arm, which has no round number."""
    return out_root(root) / arm / "fullset" / str(empiar) / kind
