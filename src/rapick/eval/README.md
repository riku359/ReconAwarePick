# 2D evaluation

Three scripts. `calc_common_2d_metrics.py` scores picks against the CryoPPP
annotations, `convert_star_to_gt.py` puts any picker's STAR into the format that scorer
reads, and `vis_star_overlay.py` draws the scored coordinates on the micrograph so a
number can be checked by eye.

They are plain scripts with no package layout: run them directly, or put this directory
on `PYTHONPATH` and import `calc_common_2d_metrics` (the head-repair scripts under
[`../picker/overlay/head_repair/`](../picker/overlay/head_repair) do exactly that).

Requirements: Python 3 and the standard library for the scorer and the converter;
`vis_star_overlay.py` additionally needs `opencv-python`.

## The metric

Precision, recall and F1 over particle centres, with no notion of pixels, masks or
segmentation. Table S2 and Table 6 both report the **macro** figures: the score is
computed per micrograph and then averaged over micrographs, so a micrograph with 30
particles counts as much as one with 300. Micro figures (TP, FP and FN pooled across
micrographs before the ratio) are computed as well and printed alongside.

### How matching works

Per micrograph, with `R = diameter / 2` (the diameter is the nominal particle diameter
of the entry, from the CryoPPP README's "Particle Diameter (px)" column):

1. Every (prediction, annotation) pair whose centres lie within `R` of each other
   becomes a candidate edge, keyed by their squared Euclidean distance.
2. The edges are sorted by **increasing distance** and walked in that order. An edge is
   accepted when neither of its two endpoints has been used yet.
3. What comes out is a **one-to-one** assignment. A matched pair is a TP, an unmatched
   prediction is an FP, an unmatched annotation is an FN. One annotation therefore
   absorbs at most one prediction: a second prediction on the same particle is an FP.

The matching criterion is Topaz's "maximum allowed radius for matching prediction to
labeled target", the one such criterion stated explicitly in a picker publication;
CryoTransformer's supplement uses the same wording.

### Why the same code can score every picker

**Ordering by distance, rather than by confidence, is what makes the procedure
picker-independent.** CryoSegNet emits no confidence at all, so its STAR has no score
column and a confidence-ordered greedy match is not even definable for it. Ordering by
distance removes score from the matching entirely, so all four pickers go through a
procedure that is identical instruction for instruction.

The same choice is why no score-dependent secondary metric is reported. There is no AP
and no best-F1 sweep: the operating point being scored is whatever each picker's STAR
already contains, which is the operating point that also feeds reconstruction.

Two further conventions keep the comparison fair:

- **The denominator is every annotated micrograph.** If a picker returned nothing on a
  micrograph, all of that micrograph's annotations become FN rather than the micrograph
  being dropped.
- **Predictions on unannotated micrographs are not scored**, because there is nothing to
  score them against. Feeding the full deposition's picks to the scorer therefore
  restricts it to the annotated subset on its own.

### The orientation check

The annotations use a top-left origin and so does the GT-aligned STAR, so predictions
are **not** flipped. Because a native, bottom-origin STAR passed in by mistake would
silently produce a plausible-looking low score rather than an error, the scorer also
matches the first few micrographs with the predictions flipped as `H - y` and prints
both counts. A flipped count much larger than the unflipped one is flagged
`!FLIPPED?`. `H` comes from the mrc header, so the check is skipped when the
micrographs are not on disk. Disable it with `--no-check-orientation`.

## The GT-aligned STAR format

This is the only input format the scorer accepts, and it is exactly the format of the
CryoPPP ground truth at
`$RAPICK_DATA/cryoppp/<ID>/ground_truth/empiar-<ID>_particles_selected.star`:

- **Block**: `data_particles`.
- **Columns**, in this order: `_rlnMicrographName`, `_rlnCoordinateX`,
  `_rlnCoordinateY`, and optionally `_rlnAutopickFigureOfMerit`. Three columns when the
  picker has no confidence to report (CryoSegNet), four when it has. CTF and optics
  columns are never fabricated.
- **`_rlnMicrographName`**: `<micrograph>.mrc`. Comparison strips the directory, a
  leading run of digits followed by `_` (the CryoSPARC import hash), and the `.mrc`
  extension, so `>J1/imported/000...371_stack_0001_DW.mrc` and `stack_0001_DW.mrc`
  resolve to the same micrograph.
- **Coordinates**: **integers** at mrc scale (not the model's input scale, not a
  normalized fraction).
- **Origin**: **top-left**. All four pickers write a bottom origin natively, so the
  conversion is `y_gt = round(H - y_native)` with `H` the `ny` field of the mrc header.

```
data_particles

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
_rlnAutopickFigureOfMerit #4
stack_0001_DW.mrc 1204 2871 0.9913
stack_0001_DW.mrc 3310 1502 0.9887
```

Granularity is free: one aggregated STAR per entry, or one STAR per micrograph in a
directory. A per-micrograph STAR without `_rlnMicrographName` takes its file stem as the
micrograph name. `--pred` accepts a file, a directory, or a glob in all cases.

Two ways to produce the format:

1. At generation time, where the picker offers it -- CryoTransformer's
   `predict.py --gt-format` (see [`../picker/`](../picker)).
2. After the fact from any picker's existing STAR, with `convert_star_to_gt.py`. It
   matches columns by their `_rln` name, so the block name and any extra columns
   (`_rlnClassNumber`, `_rlnAnglePsi`, `_rlnDiameter`, ...) are ignored.

```bash
# one aggregated STAR
python convert_star_to_gt.py IN.star --out-dir $RAPICK_WORK/picks/10081 --empiar 10081
# a directory of per-micrograph STAR files
python convert_star_to_gt.py 'native/10081/*.star' \
    --out-dir $RAPICK_WORK/picks/10081/cryolo --empiar 10081
```

`--no-flip-y` skips the Y flip for input that is already top-left; `--drop-fom` writes
the three-column form.

## Environment

| Variable | Used for |
| --- | --- |
| `RAPICK_DATA` | annotations (`cryoppp/<ID>/ground_truth/`) and micrographs, for the header height |
| `RAPICK_WORK` | `picks/<ID>/` in `--batch` mode, and the default overlay output directory |
| `RAPICK_TEST_DATA` | background jpgs for `vis_star_overlay.py`, as `<ID>/images/*.jpg` |

None has a default; see [docs/CONFIGURATION.md](../../../docs/CONFIGURATION.md).

## Commands

### Table S2 -- 2D detection scores of the four base pickers

Macro precision, recall and F1 on the 300 annotated micrographs, for crYOLO, Topaz,
CryoSegNet and CryoTransformer on each of 10081, 10093, 10345 and 10532. Every value
comes from one invocation of the scorer on that picker's GT-aligned STAR:

```bash
python calc_common_2d_metrics.py --id 10081 \
    --pred $RAPICK_WORK/picks/10081/cryotransformer.star
```

All sixteen cells at once, as a markdown table:

```bash
python calc_common_2d_metrics.py --batch --markdown \
    --out-json $RAPICK_WORK/picks/table_s2.json
```

`--batch` resolves `$RAPICK_WORK/picks/<ID>/<picker>.star`, or
`$RAPICK_WORK/picks/<ID>/<picker>/` when the picks are one STAR per micrograph.
`--ids` and `--pickers` override the defaults.

The table's `leak` column and its `avg (leak-free)` row use the code's `LEAK` table:
crYOLO's general model was trained on 10017/10028/10081, and the Topaz publication
reports 10028. **The paper is stricter about Topaz.** Table S2 greys every Topaz value,
because the training data of the *released* Topaz general model is undocumented and an
overlap with these four entries cannot be ruled out; the `LEAK` table encodes only what
the publication states, so `avg (leak-free)` does not reproduce the paper's greying for
Topaz.

### Table 6 -- per-round loop diagnostics

The `P / R / F1` column is the same macro score, applied to the picks of each round of
the feedback loop, on the same 300 annotated micrographs. One invocation per (entry,
round), against that round's GT-aligned picks under `$RAPICK_WORK/loop/<ID>/round<n>/`:

```bash
for n in 0 1 2 3; do
  python calc_common_2d_metrics.py --id 10081 \
      --pred $RAPICK_WORK/loop/10081/round$n/cryotransformer.star --json
done
```

`--json` prints the full result dict on one line, which is what to collect when the
numbers are being tabulated. The `after purify.` columns of that table are counts from
the purification stages, not products of this scorer.

The rows of Table 6 are picked over the whole deposition and scored here on the
annotated subset. No filtering step is needed for that: the scorer only ever evaluates
micrographs the annotations cover.

### Checking a number by eye

```bash
python vis_star_overlay.py --picker cryotransformer --id 10081 --n 2
python vis_star_overlay.py --gt --id 10081 --mic HCN1apo_0008_2xaligned
python vis_star_overlay.py --star $RAPICK_WORK/loop/10081/round1/cryotransformer.star --id 10081 --n 3
```

The overlay reuses the scorer's STAR resolution, STAR reading and diameter table, so the
boxes it draws are the coordinates that were scored. It draws **no Y flip**, because its
input is GT-aligned; handing it a native STAR puts every box in the wrong place. Output
goes to `$RAPICK_WORK/overlays/` unless `--out` says otherwise. Box colours are green for
the annotations and one colour per picker, so overlays rendered separately stay
comparable.
