# Fig. 2: what survives each stage

Four panels across, one row per entry, on one micrograph of each. Every raw pick lands in
exactly one of five fates, and the panels are the survivors after each stage, so the strip
narrows from the picker's raw output to the teacher labels and the last panel can be read
against the annotations.

Three steps: render the strips, cut them into bare panels, lay the panels out as a deck.

## 1. Export the stage STAR files

Three of the five populations exist only inside CryoSPARC, and the loop does not leave
them behind, so export them first from a session that reaches the server:

```bash
PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.loop.export_stage_stars --id 10081 --rounds 1
PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.loop.export_stage_stars --id 10532 --rounds 1
```

That writes `extracted.star`, `class2d_accepted.star` and `survivors.star` into
`$RAPICK_WORK/loop/<id>/round<n>/`, next to the two the loop already wrote there. The
overlay expects the contamination stage's survivors under `masked.star`; if your loop
wrote them under another name, pass `--stage-star mask=<name>`.

## 2. Render the strips

```bash
python fb_stage_overlays.py --id 10081 --rounds 1 \
    --mic HCN1apo_0343_2xaligned --panel-width 640 --quality 92
python fb_stage_overlays.py --id 10532 --rounds 1 \
    --mic FoilHole_24139658_Data_24136393_24136395_20200224_135555_Fractions_patch_aligned \
    --panel-width 640 --quality 92
```

They land under `$RAPICK_FIGURES_OUT/stage_overlays/fb/round1/<id>/stage/stage_<mic>.jpg`.
The image carries no caption, so its path is what identifies it. Copy each one to
`<strips>/stage_<entry>_round1.jpg` for the next step.

No CryoSPARC connection is needed here: the STAR files are read from disk, and the
background is the denoised micrograph under `$RAPICK_WORK/denoised/<id>/`.

`--select density` (the default) chooses micrographs by annotated particle count alone,
so the same micrographs come out for every round and arm and the rounds can be flipped
through. `--select mask` and `--select selection` instead rank by what that stage
discarded in this round: use them to find a failure, not to compare rounds.

## 3. Cut the panels and build the deck

```bash
python ../lib/prepare_overlay_panels.py --strips <dir> --only pick_fates
python build_pick_fates.py
soffice --headless --convert-to pdf --outdir <dir> <dir>/pick_fates.pptx
```

Then crop the PDF to the drawing exactly as `../pipeline_overview/README.md` describes,
including the warning against cropping with `gs -sDEVICE=pdfwrite`.

## Reading the figure

The strip holds five panels and the figure uses four: the `after 2D classification` panel
is dropped, and the per-panel counts are not printed, so the figure reads as the
qualitative narrowing rather than as a second copy of the particle table.

Consecutive panel counts differ by exactly what the stage between them removed, so the 2D
selection's own share is read off the last two panels and is not inflated by the
extraction edge and the class_2D rejects, which the panel before it has already taken out.
Those two are not the selection's doing: together they are 6,243 particles against the
selection's 21,192 on EMPIAR-10081 round 0.

The stage names are set once under the bottom row and the entry names run sideways in a
column left of the panels, so neither costs a band of its own per row. That is what keeps
the slide at 7.01 in rather than 8.59. Sideways text is `vert270` on the text box, which
python-pptx has no setter for, so it is written as a raw attribute.

## The full-set strip

`fullset_stage_overlay.py` draws the same two leading panels for the full micrograph set.
The full set is not a round and leaves no round directory, so it reads the two STAR files
the full-set arm does leave and calls the same panel renderer.

```bash
python fullset_stage_overlay.py --id 10081 --tag round1 --mic HCN1apo_0343_2xaligned
python ../lib/prepare_overlay_panels.py --strips <dir> --only protocol
```

Those four panels go into the supplementary protocol figure, which is drawn in TikZ in the
manuscript. Both blocks of it show the same field of view, because both strips are cut
from the same micrograph; the picks differ because the annotated subset is picked with the
round's checkpoint and the full set with the one the round delivers.

## Not reproducible as it stands

Only EMPIAR-10081 and 10532 have ever been rendered this way. The other two entries need a
run against a checkout that holds their round outputs.
