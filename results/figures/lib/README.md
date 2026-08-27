# lib/

What more than one figure uses. Nothing here is a figure of its own.

| Module | What it is |
| --- | --- |
| `figure_paths.py` | The one place the figure code reads `RAPICK_DATA`, `RAPICK_WORK`, `RAPICK_THIRD_PARTY`, `RAPICK_FIGURES_OUT`, `RAPICK_CHIMERAX` and the repository-root `.env`. A variable that cannot be resolved raises with the variable named. Also opens the CryoSPARC session, so credentials are read in one place and none of them reaches a command line or a log. |
| `tables.py` | Reads `results/tables/*.json`. Two figures plot values a table also prints; reading the one file is what keeps the figure and the table from drifting apart. |
| `locres_render_lib.py` | Preparing a refinement volume so that ChimeraX draws the molecule: mask by the refinement mask, contour by enclosed volume rather than by a percentile, orient on the principal axes of the density. Used by the renderer and by the interactive session, so both prepare a volume the same way. |
| `cs_fetch_assets.py` | Fetches the figures CryoSPARC renders inside its own jobs. They live in GridFS, not on disk, so `cryosparc-tools` is the only way to read them. **Needs a live CryoSPARC instance.** |
| `fb_panels.py` | The panel strip every stage overlay is built from: one micrograph, downscaled, with boxes drawn on it and a black header naming the panel and counting its boxes, repeated left to right. |
| `fb_stages.py` | What each stage of one loop round kept and what it threw away, as set operations on (micrograph, x, y). Shared so that two figures cannot attribute a pick differently. |
| `fb_paths.py` | Where a loop overlay reads from (`rapick.loop.entries`, so the figure cannot drift from the loop it draws) and where it writes to. |
| `pptx_deck.py` | The deck helpers the two hand-laid overlay figures share. |
| `prepare_overlay_panels.py` | Cuts the rendered strips into the bare panels the decks place: strips the burnt-in title bar, inpaints the mask contour away, splits a strip into its stages, keeps the upper-right quarter. Three figures are cut from the same two kinds of strip, so `--only` selects one. |

## Fetching CryoSPARC assets

The job uids in the figure scripts are the authors' instance. A fresh run produces the
same chain with different uids, so read yours out of your own project.

```bash
SPEC=$(python3 -c "print(','.join('J115=class2D_%d.png' % i for i in range(50)))")
python cs_fetch_assets.py --project P1 --spec "$SPEC" --out /tmp/tiles
```

`--project` defaults to `CRYOSPARC_PROJECT` in the repository-root `.env`, and the
credentials come from there too. Each asset lands as `<JOB>__<filename>`, which is the
name every builder downstream looks for. A filename may be given as a substring; the
first asset of that job whose name contains it is taken.

The original version of this script ran on the CryoSPARC machine over ssh, fed on stdin,
and returned the assets base64-encoded on stdout. That transport is gone: run it wherever
the instance is reachable and it writes the files itself.

## Cutting panels out of a strip

```bash
python prepare_overlay_panels.py --strips <dir> --only pick_fates
```

`--strips` holds the strips the renderers wrote, under these names:

```
stage_10081_round1.jpg   stage_10532_round1.jpg    the five-panel loop strips
stage_10081_fullset.jpg                            the two-panel full-set strip
cleaner_10532_typical.jpg  cleaner_10532_inverted.jpg
```

`--out` defaults to `$RAPICK_FIGURES_OUT/overlay_panels`, which is where both deck
builders look.

The crop keeps the upper-right quarter of each stage panel, the same corner in every
panel of a row. The figure is set at 0.8 of the text width, and the whole micrograph at
that width leaves the picks too small to tell apart. Halving both sides keeps the aspect
ratio, so the deck layout is unaffected by the crop.
