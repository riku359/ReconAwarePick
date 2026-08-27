# Fig. 6: where the contamination mask fails

Two mask overlays of EMPIAR-10532 side by side at single-column width: a typical case,
where the mask removes a handful of the annotated particles on the micrograph, and a
failure case, where the mask has inverted and removes nearly all of them. EMPIAR-10532 is
the entry the mask removes annotated particles at the highest rate, and the removals
concentrate on a few micrographs, where the contamination is left as bright holes in a
masked particle field rather than as the dark blobs the mask covers elsewhere. The entry
is not in the training data of the released MicrographCleaner weights, which is the
domain shift Sec. 6 draws from the figure.

```bash
envs/figures/.venv/bin/python results/figures/lib/prepare_overlay_panels.py \
    --strips <dir> --only cleaner_failure
envs/figures/.venv/bin/python results/figures/cleaner_failure/build_cleaner_failure.py
soffice --headless --convert-to pdf --outdir <dir> <dir>/cleaner_failure.pptx
```

Then crop the PDF to the drawing exactly as [`../pipeline_overview/README.md`](../pipeline_overview/README.md)
describes, including the warning against cropping with `gs -sDEVICE=pdfwrite`.

`build_cleaner_failure.py` takes `--panels` (default `$RAPICK_FIGURES_OUT/overlay_panels`,
which is where the previous step writes) and `--out-dir` (default `$RAPICK_FIGURES_OUT`).
`prepare_overlay_panels.py` takes `--strips`, `--out` and a repeatable `--only`.

| Reads | Writes | Needs |
| --- | --- | --- |
| `<strips>/cleaner_10532_typical.jpg` and `<strips>/cleaner_10532_inverted.jpg`, then the bare panels of the same names under `$RAPICK_FIGURES_OUT/overlay_panels/` | `$RAPICK_FIGURES_OUT/overlay_panels/cleaner_10532_*.jpg`, then `$RAPICK_FIGURES_OUT/cleaner_failure.pptx` | opencv, python-pptx, LibreOffice |

## The two overlays

They are contamination overlays of two EMPIAR-10532 micrographs, drawn by the single
overlay renderer of the contamination stage,
[`src/rapick/cleaner/overlay_panel.py`](../../../src/rapick/cleaner/overlay_panel.py):
denoised background, mask in red alpha, the 0.5 contour in yellow, a circle per particle
in the colour of its fate, and a header bar. The galleries the two frames were taken from
are rendered by the contamination-detection driver of the research repository, which is
not part of this release; `src/rapick/cleaner/classify_gt_overlap.py` records where they
sit. What this release renders on its own is the validation overlay of
`filter_star_by_contamination.py --overlay-limit`, the same panel for the first few
anomaly micrographs of the released arm, over that arm's picks rather than over the
annotated particles.

## Traps

- **The two captions are baked into the script**, as `(a) typical case: 6 of 324` and
  `(b) failure case: 181 of 183`. They are properties of those two frames, so they stay
  right only as long as the panels do: swapping in a different micrograph means editing
  the captions with it.
- **The panels have to be cut before the deck is built.** `build_cleaner_failure.py` reads
  the panel's aspect ratio out of the JPEG header, so it stops with `missing panel ...;
  run prepare_overlay_panels.py first` rather than laying out a slide of the wrong shape.
- The renderer burns a black title bar into every frame and draws the yellow 0.5 contour
  on the mask, and neither survives being shrunk to column width. `prepare_overlay_panels.py`
  strips the bar by scanning row means for the first row that is bright and stays bright,
  and inpaints the contour away with a loose yellow tolerance, because JPEG leaves a halo
  around it. Run it on the JPGs the renderer wrote, not on re-encoded copies.
- Unlike the stage panels of Fig. 2, these two are placed whole: `--only cleaner_failure`
  does not take the upper-right quarter.
- The deck is one slide 6.5 in wide, sized so the exported PDF lands at about half scale
  on the page, which is what keeps the 18 pt caption reading as roughly 9 pt in print.
