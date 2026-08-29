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

Everything runs in `envs/micrograph_cleaner`: Python
3.10, TensorFlow 2.16 + Keras 3, numpy < 2. It needs the upstream
`micrograph_cleaner_em` checkout on the **`tf2` branch** — the default `master`
branch is TF1 and will not work:

```bash
git clone -b tf2 https://github.com/rsanchezgarc/micrograph_cleaner_em.git \
    "${RAPICK_THIRD_PARTY:-$PWD/third_party}/micrograph_cleaner_em"

src/rapick/cleaner/build_env.sh      # uv sync --locked of envs/micrograph_cleaner
src/rapick/cleaner/download_model.sh # the pretrained weight, into $RAPICK_DATA/checkpoints
```


`envs/micrograph_cleaner/{pyproject.toml,uv.lock}` install the upstream package
editable from `../../third_party/micrograph_cleaner_em`, which is where
`scripts/setup.sh` puts it. That path is relative to the lockfile, so if you
point `RAPICK_THIRD_PARTY` somewhere else, symlink `third_party/` back at it
before syncing.

The scripts are run directly (`python src/rapick/cleaner/<script>.py`) and import
each other flat, so they work from any working directory.

## Files

| File | What it does |
| --- | --- |
| `triangular_mask.py` | The core. `extract_blended` slides 256-px windows at 50% overlap, weights each by a Bartlett window and overlap-adds them; `build_extractor` taps the mask and the last-layer embedding out of the loaded model. |
| `overlay_panel.py` | `denoise_flip_frame`, the CryoSegNet-identical background denoise the overlay draws on. The overlay itself is rendered by `filter_star_by_contamination.render_validation`. |
| `denoise_pipeline.py` | The denoising chain itself, vendored from CryoSegNet (`standard_scaler` -> NlMeans -> Wiener -> CLAHE -> guided filter). |
| `cleaner_env.py` | Paths from the environment variables, the model location, the per-entry box size, the in/out-of-distribution split, the ptxas fix and the denoised-JPG index. |
| `save_fullset_triangular_masks.py` | Precompute the triangular mask of every micrograph of an entry into the npz store. Resumable. |
| `filter_star_triangular.py` | **The production filter.** Predicts the triangular mask per micrograph and writes `<prefix>_clean_tri.star` / `_removed_tri.star`. Resumable. |
| `filter_star_from_masks.py` | The same decision from the precomputed npz store — no TensorFlow, no GPU. This is what the feedback loop calls each round. |
| `filter_star_by_contamination.py` | The released-post-processing arm of the same filter (upstream `predictMask`), for the Sec. S3 comparison. Also holds the shared `parse_star` / `keep_flags` / `load_micrograph`. |
| `compare_official_vs_triangular.py` | Predicts both masks for the same micrographs at model scale and writes `comparison.csv` plus the per-micrograph arrays, for the Sec. S3 comparison. |
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
$RAPICK_WORK/picks/<id>/<prefix>_clean_tri.star   the filtered picks, under the filter's own name
$RAPICK_WORK/picks/<id>/<prefix>_removed_tri.star the discarded candidates
$RAPICK_WORK/picks/<id>/filter_stats_tri.csv      per-micrograph counts
$RAPICK_WORK/picks/<id>/summary_tri.json          totals
$RAPICK_WORK/picks/<id>/decisions_tri.jsonl       per-micrograph checkpoint, for resuming
$RAPICK_WORK/mask_compare/                        comparison.csv + arrays/ (Fig. S2;
                                                  no script here draws it)
```

`<prefix>` defaults to `cryotransformer`, which is also the name the published
Hugging Face artifacts carry, so a downloaded mask arm and a locally derived one
land under the same filename. That is the filter's **own** name for its output.
`scripts/contamination_removal.sh` renames it to the name the rest of the pipeline
addresses picks by, which records the stages they have been through:
`$RAPICK_WORK/picks/<id>/cryotransformer_mask.star`, or `<name>_mask.star` for any
other `--star` it is handed. The rename is what makes the two names meet, and it
happens there and nowhere else. The `_removed_tri.star` and the two counts files
keep the filter's names: they are diagnostics, not pipeline inputs.

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
whether they overlap a CryoPPP-annotated particle. Its two rows are differences of
the 2D matching statistics between the *baseline* and *+mask* conditions (total
picks, and matches against the annotations), so the table is assembled by the
evaluation stage from those two conditions' metrics. What this stage supplies is
the `+mask` condition, the filtered STAR from (b). The per-micrograph accounting
behind the split is not part of this release: it reads a manifest written by the
research repository's contamination-detection driver, which is not published.

### (d) The two post-processings, side by side

Compute both masks for the micrographs in the store, which writes `comparison.csv`
and the per-micrograph arrays behind the Sec. S3 comparison:

```bash
python src/rapick/cleaner/compare_official_vs_triangular.py --ids 10532 --gpu 0
# -> $RAPICK_WORK/mask_compare/{comparison.csv,arrays/<id>__<mic>.npz}
```

## Skipping this stage

The precomputed masks for all four entries are published, and

```bash
bash scripts/download/08_published_artifacts.sh
```

places them at `$RAPICK_WORK/masks/<id>/` along with the already-filtered picks. With
them this stage reduces to (b): no GPU, no model weight, and no upstream checkout are
needed to reproduce the filtered picks. Where each file comes from:
[docs/DATA.md](../../../docs/DATA.md).

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
