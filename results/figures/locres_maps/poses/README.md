# poses/

Two frozen placements for Fig. 3, one per row: `poses_10093.json` and `poses_10345.json`.
Each carries the camera, the 3x4 placement of every panel, the palette the row was
coloured with (10093 `4.839,blue:8.584,white:15.08,red`, 10345
`3.517,blue:7.033,white:12,red`) and the volume each panel was actually drawn from.
`render_locres_3d.py --poses <file>` replays them instead of refitting the row.

They are committed because they cannot be derived again: `fitmap search` starts from
random placements, so a second freeze of the same row lands somewhere else. Both were
frozen with crYOLO as the reference.

- The paths inside name the `--work` directory of `locres_gui_prep.py`, which is
  `/tmp/locres/gui/masked_cryolo` in these two files. Re-running the prep with the same
  `--work` refills the masked maps they point at; the local-resolution volumes have to sit
  there too.
- A panel named `__masked_flip.mrc` is one the freeze mirrored, and `volume flip #N axis z`
  followed by `save` writes it back. Both rows mirrored their Topaz map.
- Never overwrite a freeze: it is the only record of a placement decided by hand.
  `save_poses.py` keeps an existing file as `.bak` and takes `LOCRES_POSES_OUT` for a
  second freeze of the same entry.

## Reading a row back out of a PDF

EMPIAR-10081 and 10532 have no pose file: theirs was lost, so those two rows cannot be
re-rendered and are reused instead. Their panels survive inside the figure PDF the
manuscript carries, and `pdfimages` takes them back out:

```bash
pdfimages -png -all <the figure PDF> /tmp/locres/extract/p
```

That writes each panel as an image and its alpha as a separate `smask`, in drawing order:
five panels of EMPIAR-10081 in column order, then five of 10532. Recombine each pair into
one RGBA PNG, name every panel and its row's palette stops in a manifest, and lay the rows
out with `tile_locres_panels.py`; [`../README.md`](../README.md) carries that command and
the stops of those two rows. Nothing in this figure is lost while its PDF exists.
