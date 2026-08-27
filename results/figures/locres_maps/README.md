# Fig. 3: local-resolution maps

One row per EMPIAR entry, five panels across in the column order of Table 2 (crYOLO,
Topaz, CryoSegNet, CryoTransformer, Ours), each density map coloured by its local
resolution. It is the figure CryoTransformer's Supplementary S10 shows, with two changes.
S10 gives every panel its own colour bar, so red on one panel and red on the next mean
different things and its columns cannot be compared by eye; here the range is computed
once per row and one bar is drawn for it. S10 also leaves every map in whatever
orientation its refinement ended in; here the reference map of a row is oriented on the
principal axes of its own density and the others are fitted onto it, so a difference
between two panels is a difference between the maps rather than between two viewpoints.
The paper carries the EMPIAR-10081 and 10532 rows, because four rows at full width
overrun the page.

**Needs ChimeraX**, which is an application rather than a Python dependency. The renderer
takes `--chimerax`, then `$RAPICK_CHIMERAX`, then the macOS bundle under `/Applications`.
On Linux point it at `chimerax_headless.sh` next to the renderer: that wrapper extracts a
cached ChimeraX 1.12 `.deb` and a `libosmesa6` `.deb` into a per-node local directory and
prepends `--offscreen --nogui`. It reads them from `$RAPICK_WORK/tools/chimerax/debs`,
overridable with `CHIMERAX_CACHE` and `CHIMERAX_LOCAL`.

```bash
# 1. which volumes each panel is drawn from, out of the pipeline's own manifests
envs/figures/.venv/bin/python results/figures/locres_maps/build_locres_spec.py \
    --out /tmp/locres/spec.json

# 2. render the twenty panels and tile them, replaying each row's frozen placement
envs/figures/.venv/bin/python results/figures/locres_maps/render_locres_3d.py \
    --spec /tmp/locres/spec.json --out /tmp/locres/locres_maps.pdf \
    --entries 10093 10345 --reference crYOLO \
    --poses results/figures/locres_maps/poses/poses_10093.json \
    --poses results/figures/locres_maps/poses/poses_10345.json \
    --render-dir /tmp/locres/panels --panel-px 1040 --silhouette-width 0.5
```

`--panel-px 1040` and `--silhouette-width 0.5` are what the committed figure used; the
defaults are 520 and 1.0. `build_locres_spec.py` also takes `--experiments` (manifest
roots, default `$RAPICK_WORK`), `--setting` (default `full`) and `--entries`.
`render_locres_3d.py` also takes `--chimerax`, `--stops`, `--turn` and `--panel-turn`.

| Reads | Writes | Needs |
| --- | --- | --- |
| `$RAPICK_WORK/empiar_<id>/full/<condition>/metrics.json`, and through it each job's `*_volume_map_sharp.mrc`, `*_map_locres.mrc` and `*_mask_refine.mrc` | `--out` (the tiled PDF), the panel PNGs under `--render-dir`, and the masked volumes under `<render-dir>/masked/` | **ChimeraX**, matplotlib, mrcfile |

## Frozen placements

`--turn` and `--panel-turn` are how a row used to be brought upright by hand. They are
gone from the committed figure: the placement is now decided in a ChimeraX window,
checked by turning it, and frozen to JSON, and the render replays that instead of
refitting.

```bash
envs/figures/.venv/bin/python results/figures/locres_maps/locres_gui/locres_gui_prep.py \
    --spec /tmp/locres/spec.json --entry 10093 --reference crYOLO \
    --work /tmp/locres/gui/masked_cryolo --out /tmp/locres/params_10093.json
# add "fit": "envelope" to the params for the smoothed, both-hands fit, then
LOCRES_GUI_PARAMS=/tmp/locres/params_10093.json LOCRES_WORK=/tmp/locres \
    <chimerax> --script results/figures/locres_maps/locres_gui/locres_gui_session.py
```

In the window: `flat` then `ghost`, then turn it. Five envelopes that track each other
from every angle are aligned and ones that do not are not; `figview` returns to the
figure's camera. When it looks right, `open results/figures/locres_maps/locres_gui/save_poses.py`
writes `$LOCRES_WORK/poses_<entry>.json`. `freeze_headless.py` runs the session and
freezes it without anyone turning it, which is worth doing when the envelope fit
converges. `locres_hand_check.py` fits every panel twice, as it is and z-mirrored, and
prints both correlations. [`poses/`](poses/README.md) holds the two freezes that survive.

## Traps

- **macOS has no OSMesa.** The bundle cannot render offscreen: `--offscreen` and `--nogui`
  both fail with "OpenGL rendering is not available", so it must run windowed and windows
  flash up and close by themselves for every panel. On Linux, ChimeraX 1.12 is built for
  Ubuntu 22.04 and dies with `GLIBC_2.34 not found` on 20.04 before it draws anything.
- **The silhouette is heavier on Retina.** The width is in screen pixels and the macOS
  build draws on a Retina framebuffer, so leaving it to ChimeraX gives a line two to three
  times heavier than a Linux render and the maps come out ringed in black. `1` reproduces
  the Linux outline and `0.5` is as thin as the setting goes: `0.25` renders identically,
  to the byte. Thinner than that means drawing bigger panels and letting them shrink into
  the same box, which is what `--panel-px 1040` does, since the tiling pads panels rather
  than scaling them. Render a whole figure on one machine, so every panel of a row
  matches.
- **A failed render silently reuses the last panels.** ChimeraX exits 0 on some failures,
  so the renderer deletes each target PNG before it renders and treats a missing file as
  the error. Without that a broken run reassembles the previous run's panels and says
  nothing.
- **Save paths are relative to ChimeraX's own working directory.** The renderer writes the
  masked volume and the save target as absolute paths for that reason; a relative path
  typed into a session lands wherever ChimeraX was started.
- **`view matrix` is punctuated two ways**: the camera takes the numbers after a space, a
  model after its spec and a comma.
- **Align to the best map of the row, not to a fixed method.** `--reference` defaults to
  `CryoTransformer`, and on EMPIAR-10093 and 10345 that map is the worst reconstruction of
  its row (6.78 and 7.11 A), so the default fits four good maps onto a broken one.
  Envelope correlation after freezing, worst panel of the row and mean of the four: on
  10093, 0.774 / 0.793 with CryoTransformer as reference against 0.834 / 0.890 with
  crYOLO; on 10345, 0.878 / 0.903 against 0.918 / 0.975. crYOLO is the reference of the
  committed figure: it is the best of the base pickers on every entry and it is not our
  own map.
- **The renderer's own fit cannot cross a mirror.** It works on the raw masked density and
  settles at 0.72 to 0.87, against 0.94 to 0.996 once the smoothed envelope is fitted with
  `search` and both hands are tried, which is what the GUI session does. EMPIAR-10081's
  CryoSegNet and Ours maps came out of refinement in the opposite hand and are drawn
  mirrored. A mirrored panel is then drawn from a volume ChimeraX derived and wrote out,
  not from the file the spec names, which is why the session records what it drew each
  panel from and the freeze copies that into the pose file.
- **One colour scale per row, never one for the figure.** EMPIAR-10345's declared pixel
  size is half the physical one, so its row sits on a different Angstrom scale from the
  other three and a global range would saturate it red and show nothing. `--stops
  10081=8,10,14` overrides one row.
- What is drawn is the masked map at whatever level ChimeraX picks when it opens, which is
  the GUI procedure this figure follows. The enclosed-volume level the script computes is
  used only to work out the orientation, where noise outside the molecule would otherwise
  dominate the principal axes.
- A frozen row ignores `--turn` and `--panel-turn`, because its turns are already baked
  into the placement it was saved at, and it exits naming any panel the pose file has no
  placement for.
- Under llvmpipe on a headless server a panel takes roughly an order of magnitude longer
  than a windowed render on a machine with a GPU.

## What cannot be re-rendered

The camera placements of the EMPIAR-10081 and 10532 rows were approved by eye and frozen
to JSON, and that JSON was lost. `fitmap search` starts from random placements, so a
second freeze of the same row lands somewhere else: a re-render gives the same maps at a
different orientation. Those two rows survive only inside the figure PDF the manuscript
carries, and [`poses/README.md`](poses/README.md) says how their panels are read back out
of it. The palette stops they were coloured with are EMPIAR-10081 `4.174,9.526,15.32` and
EMPIAR-10532 `3.345,7.703,16.73`; `tile_locres_panels.py` takes them in its manifest,
alongside panels rendered fresh for the other rows.
