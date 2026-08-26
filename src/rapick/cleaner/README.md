# Contamination masking

This stage is the first of the two purification steps in the selection operator
(paper Sec. 3.3, with the mask assembly detailed in Sec. S3). Contamination such
as carbon film, ice and aggregates carries more contrast than the particles, so
candidates taken from it would otherwise reach 2D classification and survive
there as false-positive classes. The pretrained model of
[MicrographCleaner](https://github.com/rsanchezgarc/micrograph_cleaner_em)
predicts contamination as a per-pixel probability map, and a candidate is
discarded when that map, resized to the full micrograph resolution, reaches a
probability of 0.5 at the candidate's centre.

## Mask assembly (Sec. S3)

MicrographCleaner predicts on fixed-size 256x256 windows, so the per-window
predictions have to be assembled into one mask per micrograph. The released
assembly averages the overlapping windows with **uniform** weights: a window
contributes at full weight up to its own border, so the assembled mask carries a
step wherever adjacent windows disagree, and `fixJumpInBorders` repairs those
steps afterwards. That routine cannot tell a genuine intensity step in the
micrograph from a window-border artefact, and when it misreads one it overwrites
everything between the border and the image edge with the values just before the
border, flooding a rectangular region of the mask. The released assembly also
averages over eight rotations of the input.

This stage replaces that assembly with **triangular blending**
(`triangular_mask.extract_blended`). Windows overlap by half their size and the
predictions are averaged with weights that fall linearly from one at the window
centre to zero at the border, so adjacent windows hand over smoothly and no step
forms even where they disagree. No seam correction is needed. The rotation
averaging is dropped and each micrograph is predicted in a single orientation.

`filter_star_triangular.py` is the production filter. `filter_star_by_contamination.py`
is the same filter over the released assembly, kept so the two can be compared.

## Environment

Everything except the figure scripts runs in `envs/micrograph_cleaner`: Python
3.10, TensorFlow 2.16 + Keras 3, numpy < 2. It needs the upstream
`micrograph_cleaner_em` checkout on the **`tf2` branch** — the default `master`
branch is TF1 and will not work:

```bash
git clone -b tf2 https://github.com/rsanchezgarc/micrograph_cleaner_em.git \
    "${RAPICK_THIRD_PARTY:-$PWD/third_party}/micrograph_cleaner_em"

src/rapick/cleaner/build_env.sh      # uv sync --locked of envs/micrograph_cleaner
src/rapick/cleaner/download_model.sh # the pretrained weight, into $RAPICK_DATA/checkpoints
```

`plot_mask_postproc_figures.py` and `classify_gt_overlap.py` need matplotlib,
which this environment does not have; run them from any environment that has
numpy and matplotlib.

`envs/micrograph_cleaner/{pyproject.toml,uv.lock}` install the upstream package
editable from `../../third_party/micrograph_cleaner_em`, which is where
`scripts/00_setup.sh` puts it. That path is relative to the lockfile, so if you
point `RAPICK_THIRD_PARTY` somewhere else, symlink `third_party/` back at it
before syncing.

The scripts are run directly (`python src/rapick/cleaner/<script>.py`) and import
each other flat, so they work from any working directory.

## Files

| File | What it does |
| --- | --- |
| `triangular_mask.py` | The core. `extract_blended` slides 256-px windows at 50% overlap, weights each by a Bartlett window and overlap-adds them; `build_extractor` taps the mask and the last-layer embedding out of the loaded model. |
| `overlay_panel.py` | The single overlay renderer: denoised background, mask in red alpha, threshold contour, per-pick circles, header bar. Also holds `denoise_flip_frame`, the CryoSegNet-identical background denoise. |
| `denoise_pipeline.py` | The denoising chain itself, vendored from CryoSegNet (`standard_scaler` -> NlMeans -> Wiener -> CLAHE -> guided filter). |
| `cleaner_env.py` | Paths from the environment variables, the model location, the per-entry box size, the in/out-of-distribution split, the ptxas fix and the denoised-JPG index. |
| `save_fullset_triangular_masks.py` | Precompute the triangular mask of every micrograph of an entry into the npz store. Resumable. |
| `save_triangular_masks.py` | The same precompute restricted to the micrographs the released arm removed the most annotated particles from. Needs the released arm's mask store. |
| `filter_star_triangular.py` | **The production filter.** Predicts the triangular mask per micrograph and writes `<prefix>_clean_tri.star` / `_removed_tri.star`. Resumable. |
| `filter_star_from_masks.py` | The same decision from the precomputed npz store — no TensorFlow, no GPU. This is what the feedback loop calls each round. |
| `filter_star_by_contamination.py` | The released-post-processing arm of the same filter (upstream `predictMask`), for the Sec. S3 comparison. Also holds the shared `parse_star` / `keep_flags` / `load_micrograph`. |
| `compare_official_vs_triangular.py` | Predicts both masks for the same micrographs at model scale and writes `comparison.csv` plus the per-micrograph arrays behind Fig. S2. |
| `plot_mask_postproc_figures.py` | Draws the two mask post-processing figures (Fig. S2, Fig. S3), applying the upstream `fixJumpInBorders` to a synthetic mask and to the real arrays. |
| `classify_gt_overlap.py` | Counts, per micrograph, the annotated particles that fall inside each method's mask, plus the masked area, and plots the comparison. |
| `count_gt_in_contamination.py` | Aggregates those per-micrograph counts by entry and by distribution class into a markdown table. |
| `build_env.sh` | Builds the venv from `envs/micrograph_cleaner`. |
| `download_model.sh` | Fetches the pretrained weight from Zenodo. The package's own `cleanMics --download` only gunzips an archive that is really a tar, leaving a file Keras cannot read; this script gunzips *and* untars. |

## Inputs and outputs

Paths come from the environment variables in
[docs/CONFIGURATION.md](../../../docs/CONFIGURATION.md). A script that cannot
resolve one fails naming it, rather than falling back to a default.

Inputs, under `$RAPICK_DATA`:

```
$RAPICK_DATA/cryoppp/<id>/micrographs/                          the 300 annotated .mrc
$RAPICK_DATA/cryoppp/<id>/ground_truth/empiar-<id>_particles_selected.star
$RAPICK_DATA/cryoppp_fullset/<id>/micrographs/                  the full deposition
$RAPICK_DATA/checkpoints/micrograph_cleaner_defaultModel.h5     download_model.sh
$RAPICK_DATA/cryosegnet_dataset/                                optional, overlay backgrounds only
```

Outputs, under `$RAPICK_WORK`:

```
$RAPICK_WORK/masks/<id>/<mic>_tri.npz          triangular masks (float16, model scale) + meta
$RAPICK_WORK/picks/<id>/<prefix>_clean_tri.star   the filtered picks the reconstruction reads
$RAPICK_WORK/picks/<id>/<prefix>_removed_tri.star the discarded candidates
$RAPICK_WORK/picks/<id>/filter_stats_tri.csv      per-micrograph counts
$RAPICK_WORK/picks/<id>/summary_tri.json          totals
$RAPICK_WORK/picks/<id>/decisions_tri.jsonl       per-micrograph checkpoint, for resuming
$RAPICK_WORK/mask_compare/                        comparison.csv + arrays/ (Fig. S2)
$RAPICK_WORK/figures/                             the rendered figures
```

The stored mask is exactly what the filter would compute at run time
(`extract_blended(extractor, preprocessMic(image, box), 2, 1)`); the only
difference is the rounding from float16 storage, whose step near 0.5 is about
0.0005. The mask depends only on the micrograph, never on the picks, so it is
computed once and reused for every condition and every feedback round.

## Commands

### (a) Precompute the masks for one entry

Every micrograph of the full deposition. The 300 annotated micrographs are a
subset of it under the same file names, so this covers both; the run resumes, so
whatever is already in the store is skipped.

```bash
python src/rapick/cleaner/save_fullset_triangular_masks.py --ids 10081 --gpu 0
```

For the annotated micrographs alone, point it at the other root:

```bash
python src/rapick/cleaner/save_fullset_triangular_masks.py --ids 10081 --gpu 0 \
    --mic-root "$RAPICK_DATA/cryoppp"
```

### (b) Filter a STAR

From the precomputed store (no GPU, no TensorFlow) — this is the normal path:

```bash
python src/rapick/cleaner/filter_star_from_masks.py \
    --star "$RAPICK_WORK/picks/10081/cryotransformer.star" \
    --empiar-id 10081
# -> $RAPICK_WORK/picks/10081/cryotransformer_clean_tri.star
```

Predicting the mask on the fly instead, when there is no store:

```bash
python src/rapick/cleaner/filter_star_triangular.py \
    --star "$RAPICK_WORK/picks/10081/cryotransformer.star" \
    --mic-dir "$RAPICK_DATA/cryoppp_fullset/10081/micrographs" \
    --empiar-id 10081 --gpu 0
```

The released-post-processing arm, for the Sec. S3 comparison, writes
`cryotransformer_clean.star` (no `_tri` suffix) into the same directory:

```bash
python src/rapick/cleaner/filter_star_by_contamination.py \
    --star "$RAPICK_WORK/picks/10081/cryotransformer.star" \
    --mic-dir "$RAPICK_DATA/cryoppp_fullset/10081/micrographs" \
    --empiar-id 10081 --gpu 0
```

### (c) Table 3 — mistaken rate of the contamination mask

Table 3 splits the candidates the mask removed on the annotated micrographs by
whether they overlap a CryoPPP-annotated particle. Its two rows are differences
of the 2D matching statistics between the *baseline* and *+mask* conditions
(total picks, and matches against the annotations), so the table itself is
assembled by the evaluation stage from the two conditions' metrics. What this
stage supplies is the `+mask` condition — the filtered STAR from (b) — and the
per-micrograph accounting of which annotated particles the mask covers:

```bash
# per-micrograph counts and the method comparison figure
python src/rapick/cleaner/classify_gt_overlap.py --no-copy

# the same counts aggregated per entry and per distribution class
python src/rapick/cleaner/count_gt_in_contamination.py
```

Both read stores written by the contamination-detection driver of the research
repository, which is not part of this release: `classify_gt_overlap.py` needs the
released arm's masks (`--official-root`) and its per-micrograph manifest
(`--manifest`) alongside `$RAPICK_WORK/masks`, and `count_gt_in_contamination.py`
reads that manifest's `n_gt` / `n_gt_in_contam` columns
(`empiar_id, dist_class, micrograph, height, width, box_size, deep_thr,
contam_fraction, max_mask, has_anomaly, n_gt, n_gt_in_contam, overlay_path,
status`). `--no-copy` skips the overlay classification, which needs the overlay
galleries that driver renders.

### (d) Fig. S2 and Fig. S3

First compute both masks for the micrographs in the store, which writes
`comparison.csv` and the per-micrograph arrays:

```bash
python src/rapick/cleaner/compare_official_vs_triangular.py --ids 10532 --gpu 0
# -> $RAPICK_WORK/mask_compare/{comparison.csv,arrays/<id>__<mic>.npz}
```

Then draw the figures from those arrays (matplotlib environment, no GPU):

```bash
python src/rapick/cleaner/plot_mask_postproc_figures.py
# -> $RAPICK_WORK/figures/postproc/C_fixjump_mechanism.{png,pdf}
#    $RAPICK_WORK/figures/postproc/D_triangular_window.{png,pdf}
```

`C_fixjump_mechanism` applies the upstream `fixJumpInBorders` to a synthetic mask
with a step at a window border and shows the flooded result next to the real
released and triangular masks of one EMPIAR-10532 micrograph; those two real
panels are the material for Fig. S2. `D_triangular_window` is the blending
schematic behind Fig. S3. The real panels default to the micrograph named by
`REAL_EXAMPLE`; `--example` selects a different npz from `--arrays`.

## Skipping this stage

The precomputed masks for all four entries are published in the Hugging Face
dataset [`rikrikrik/recon-aware-pick-data`](https://huggingface.co/datasets/rikrikrik/recon-aware-pick-data)
under the path prefix `triangle_mask_overlay/anomaly_mask_npy/`. Download those
into `$RAPICK_WORK/masks/<id>/` and this stage reduces to (b): no GPU, no model
weight, and no upstream checkout are needed to reproduce the filtered picks.

## Notes

- **The 0.5 test is on the centre pixel.** `keep_flags` flips the mask with
  `flipud` to line it up with the top-left-origin STAR coordinates and reads
  `mask[round(y), round(x)]`. A coordinate outside the frame is kept, which is
  the safe side. `filter_star_triangular.py`, `filter_star_from_masks.py` and
  `filter_star_by_contamination.py` all call the same function, so the three
  differ only in how the mask was made.
- **Failures keep every pick.** A micrograph that cannot be read, or whose mask
  is missing from the store, is recorded as `status=error` with all its picks
  kept. The run does not stop.
- **`decisions.jsonl` is a checkpoint, not an output.** The STAR is assembled from
  it at the end, so a resumed run reproduces the same STAR. On resume, `error`
  records are retried and the corrected record wins.
- **The box size only sets the preprocessing downsample**, never the model
  weights, so one loaded model serves every entry. The nominal diameters are in
  `cleaner_env.DIAMETERS`; an unknown entry gets 180 px.
- **A movie-stack `.mrc`** (shape `(frames,H,W)`) is collapsed by averaging the
  frames. Contamination is large-scale structure, so an unaligned average is
  enough.
- **MicrographCleaner's in/out-of-distribution split** (`cleaner_env.TRAIN_EMPIAR`)
  is the model's own training set and has nothing to do with any picker's
  training set. Do not conflate the two.
- **GPU selection goes through `--gpu`.** `MaskPredictor` overwrites
  `CUDA_VISIBLE_DEVICES` from its `gpus` argument before TensorFlow is imported,
  so setting that variable outside has no effect.
