# Fig. S2 and Fig. S3: mask post-processing

Both figures of Sec. S3, from one script. **Fig. S2** puts the released post-processing
next to this repository's triangular blending on one real micrograph of EMPIAR-10532, the
stripe example: two panels, the mask in red over the denoised micrograph, no contour and
no pick markers. **Fig. S3** is the synthetic explanation of why they differ: four windows
at 50% overlap predicting the same smooth field plus a window-specific offset, assembled
once with uniform weights and once with triangular ones, so the step at the window border
is visible in the first and absent in the second.

```bash
# both figures
envs/figures/.venv/bin/python results/figures/mask_postproc/build_mask_postproc_figs.py \
    --assets <dir>

# Fig. S3 alone, which needs nothing
envs/figures/.venv/bin/python results/figures/mask_postproc/build_mask_postproc_figs.py
```

Three flags: `--assets`, `--stem` (default `24136458`, the micrograph the assets are named
after) and `--out-dir` (default `$RAPICK_FIGURES_OUT`).

| Reads | Writes | Needs |
| --- | --- | --- |
| Fig. S2: `<assets>/mic_<stem>.jpg`, `<assets>/mask_off_<stem>_gray.png`, `<assets>/mask_tri_<stem>_gray.png`. Fig. S3: nothing | `$RAPICK_FIGURES_OUT/mask_postproc_real.pdf`, `$RAPICK_FIGURES_OUT/mask_blend_weights.pdf` | matplotlib, pillow, numpy |

The three assets are images rather than code, so they are not committed.
[`src/rapick/cleaner/`](../../../src/rapick/cleaner/README.md) regenerates both masks from
the micrograph: `compare_official_vs_triangular.py` predicts both for the same micrograph
and writes the per-micrograph arrays, and `plot_mask_postproc_figures.py` draws the
mechanism version of the same comparison (`C_fixjump_mechanism`, `D_triangular_window`)
under `$RAPICK_WORK/figures/postproc/`. The two PDFs the manuscript includes are the ones
built here.

## Traps

- **Without `--assets` the script does not fail.** It prints one line, skips Fig. S2 and
  builds Fig. S3 alone. A run that was meant to build both looks like a success. With
  `--assets` given, a missing file raises and names all of the missing ones.
- **The stored masks are upside down relative to the denoised jpg.** They are in the raw
  mask frame, so the script applies `flipud` to each, the same flip the overlay renderer
  in `src/rapick/cleaner/overlay_panel.py` applies. An asset that was already flipped when
  it was exported comes out inverted with no error.
- **The two masks are made at different scales**: the released post-processing's from the
  stored full-resolution npz, the triangular one at the 376 px model scale. Both are
  resampled to the 1600 px display size, so the figure compares post-processings and not
  resolutions.
- The micrograph is EMPIAR-10532 `FoilHole_24136458_..._021514`, identified by correlation
  (0.87) against the saved default overlays of the contamination stage. It is not the
  micrograph `plot_mask_postproc_figures.py` defaults to, which is the flooded-band
  example of the failure gallery, so the arrays that script writes for its own default are
  not these assets.
- **`split_filtercmp` is kept but never called.** It splits a saved filter-comparison jpg
  into its two panels for an alternative version of Fig. S2 that is not built: the same
  micrograph with the 0.5 contour in yellow and the picks as circles. Its baked header
  strip carries removed/kept counts that no table in the paper backs, which is why that
  version drops the header and, in the end, was not used.
- Fig. S3's step lands at x = 2 and x = 3, where same-parity windows disagree; the panel is
  clipped to that interior so the image edges, where every assembly is one-sided, stay out
  of view.
