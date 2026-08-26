# The picker: CryoTransformer

The picker is [CryoTransformer](https://github.com/jianlin-cheng/CryoTransformer)
(Dhakal et al.), MIT licensed, Copyright (c) 2023 Jianlin Cheng, pinned at commit
`a56f133f3f499562c32a6bc512eec5f115095b3b`.

This repository does **not** vendor a copy of it. `scripts/00_setup.sh` clones upstream
at that commit into `$RAPICK_THIRD_PARTY/cryotransformer/`, then copies everything under
[`overlay/`](overlay) over the clean checkout. What ships here is therefore only the
authors' own files plus the three upstream files they had to change.

```
overlay/                    copied over the clean upstream clone at setup time
├── predict.py              upstream, MODIFIED
├── train.py                upstream, MODIFIED
├── datasets/micrograph.py  upstream, MODIFIED
├── finetune.py             new
├── predict_fullset.py      new
└── head_repair/            new -- the head repair of Sec. S2
patches/                    the same three modifications as readable diffs
```

## Which files are which

**Upstream, unmodified** -- taken from the clone and never touched: `engine.py`,
`denoise_micrographs.py`, `environment.yml`, `models/` (`backbone.py`, `detr.py`,
`matcher.py`, `position_encoding.py`, `segmentation.py`, `transformer.py`), `util/`
(`box_ops.py`, `misc.py`), `datasets/` except `micrograph.py` (`__init__.py`, `coco.py`,
`coco_eval.py`, `transforms.py`, `annotation_preparation/`), plus upstream's own
`README.md`, `LICENSE` and `visuals/`.

**Upstream, modified** -- shipped in `overlay/`, with the change also readable as a
unified diff in [`patches/`](patches). Each diff was produced against the pinned
upstream blob and applies to a clean clone with `patch -p1`:

| File | Patch | What changed |
| --- | --- | --- |
| `predict.py` | `patches/predict.py.diff` | an `indices` argument threaded through `nms()`, so a surviving box can be traced back to the decoder query slot it came from; a `--gt-format` output mode; and the `--debug_dump` / `--dump_hs` diagnostic dumps. `--dump_hs` is what feeds the head repair. |
| `train.py` | `patches/train.py.diff` | `class_embed` and `query_embed` are reinitialised only when the checkpoint's shapes disagree with the model's, so resuming the project's own run no longer resets the trained head; and optimizer / lr\_scheduler / epoch are restored only on such a relay. |
| `datasets/micrograph.py` | `patches/datasets_micrograph.py.diff` | `category_id` is remapped to 0. |

Applying them by hand, if you are not using `scripts/00_setup.sh`:

```bash
cd $RAPICK_THIRD_PARTY/cryotransformer
for p in predict train datasets_micrograph; do
  patch -p1 --dry-run < <repo>/src/rapick/picker/patches/$p.py.diff
done
```

### Why `category_id` has to be remapped

The COCO annotations CryoTransformer distributes label **every** particle
`category_id=1`, in both the train and val splits. `models/detr.py`'s `build()` sets
`num_classes=1` for this dataset, and DETR's no-object index is `num_classes`, i.e. also
1. Without the remap, a query matched to a real particle is trained toward the
no-object class: the classification loss carries no usable signal. Subtracting 1 puts
particles on index 0 and leaves no-object alone on index 1. The head repair does not go
through this loader -- it reads ground truth from CryoPPP `.star` files -- so the remap
matters only for a full retrain.

**Authors' own files**, all under `overlay/`:

| File | What it is |
| --- | --- |
| `finetune.py` | The fine-tuning stage of Sec. 3.5 and Sec. S2. Trains from a GT-aligned STAR plus a micrograph directory, with no COCO annotation build step. |
| `predict_fullset.py` | Picking the whole deposition. |
| `head_repair/*.py` | The head repair of Sec. S2, which produces theta_0. |

## The operating point

The picks the paper reports come out of `predict.py` at its shipped defaults, and the
whole pipeline is built on that operating point rather than on a re-tuned one:

- **600 queries per micrograph** (`--num_queries 600`): the decoder emits 600 candidate
  boxes for every micrograph, and that count is the ceiling on how many particles a
  micrograph can yield.
- **keep the top 75% by score** (`--quartile_threshold 0.25`): candidates scoring below
  the 25th percentile of that micrograph's own scores are dropped. The cut is relative
  to the micrograph, not absolute.
- **NMS at overlap 0.7** (`--nms_threshold 0.7`): of two surviving boxes overlapping by
  more than 0.7, the lower-scoring one is discarded.

```bash
python predict.py --empiar 10081 \
    --resume $RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth \
    --gt-format
```

`--gt-format` writes the combined STAR in the format
[`../eval/README.md`](../eval/README.md) specifies, so the output is directly scorable
and directly reconstructable. Without it the output is upstream's native
bottom-origin STAR.

`predict_fullset.py` takes the same flags. It is `predict.py` without any of the
diagnostic machinery -- no `--debug_dump`, no `--dump_hs`, no forward hook on
`class_embed` -- so a multi-day run over a full deposition is not slowed or perturbed by
instrumentation it does not need. Point `RAPICK_TEST_DATA` at the full deposition rather
than the annotated subset.

## Where the micrographs come from

Upstream's `predict.py` reads `test_data/<EMPIAR id>/images/` relative to the working
directory, which on the authors' machines was a symlink into the data disk. **The layout
below the root is unchanged; only the root moved into the environment**:

```
$RAPICK_TEST_DATA/
└── <EMPIAR id>/
    └── images/          .mrc (or .jpg) micrographs -- a directory or a symlink
```

`RAPICK_TEST_DATA` is read by `predict.py`, `predict_fullset.py` and
`../eval/vis_star_overlay.py`. `--data_root` overrides it per run. Neither has a
default: with neither set, the scripts stop with an error naming the variable. Point it
at `$RAPICK_DATA/cryoppp` for the 300 annotated micrographs per entry, and at
`$RAPICK_DATA/cryoppp_fullset` for the full depositions -- both already have the
`<id>/` level, so only an `images` link per entry is needed.

`finetune.py` does not use this contract at all: it takes `--images_dir` and `--star`
explicitly.

`finetune.py` and `head_repair/` import `calc_common_2d_metrics` from
[`../eval/`](../eval), the repository's single STAR reader. They find it on
`PYTHONPATH` first and fall back to the in-repository path, which stops resolving once
the overlay has been copied over the clone. Export it before running them from
`$RAPICK_THIRD_PARTY/cryotransformer/`:

```bash
export PYTHONPATH=<repo>/src/rapick/eval:$PYTHONPATH
```

## The two training stages (Table S1)

### Stage 1 -- head repair, producing theta_0

The released CryoTransformer head is degenerate: training collapsed `category_id=1` onto
the no-object index, so the shipped `class_embed` ranks queries barely better than chance
(AUC 0.531 on the 22-entry held-out split, against a 0.601 positive rate). Fine-tuning on
top of that adapts a classifier that never learned to rank, and 40 micrographs cannot
repair it -- the repair needs the full 22-entry training data.

The repair refits `class_embed` alone on the frozen detector's features. Everything else
in the checkpoint -- backbone, encoder, decoder, `bbox_embed`, `query_embed` -- is
carried over untouched. **Settings (Table S1, "head repair"):**

| | |
| --- | --- |
| trained weights | head only (`class_embed`) |
| training set | 22 CryoPPP entries |
| loss | two-class softmax cross entropy |
| class weight | 1 |
| no-object weight | 0.1 |
| optimizer | Adam |
| learning rate | 2e-3 |
| weight decay | 1e-4 |
| batch size | 32,768 queries |
| epochs | 15 |

Because the decoder is frozen, no image ever passes through the model during this
training: the decoder's hidden states are dumped once and the head is fitted on them, on
CPU, in fp32.

Labels are **many-to-one**: a query is positive when its predicted box centre lies
within `R = diameter / 2` of any annotated particle. Hungarian one-to-one assignment is
deliberately not used here -- with the decoder frozen, several queries on the same
particle have near-identical hidden states, so electing a single winner would hand the
linear head contradictory labels.

The scripts, in the order they run:

| Script | Role |
| --- | --- |
| `build_train_stem_mapping.py` | Recovers which of the 22 entries each training micrograph belongs to, by matching filenames against the local CryoPPP ground truth. Writes the `stem,empiar_id` CSV. |
| `predict.py --dump_hs DIR` | Dumps, per micrograph, the last decoder layer's hidden state `hs_last` (600, 256), `pred_logits`, and `pred_boxes_px` in original-micrograph pixels. |
| `aggregate_hs_by_id.py` | Regroups those per-micrograph dumps into one npz per entry. |
| `linear_probe.py` | The diagnosis: a logistic-regression probe on `hs`. Same function class as `class_embed` itself, so its held-out AUC estimates the ceiling a head-only retrain can reach. Includes a control that scores each query slot by its historical positive rate alone, isolating how much of the AUC is fixed positional prior rather than per-image content. |
| `phase_d_train_heads.py` | Leave-one-entry-out CV over head architecture, loss form and no-object weight. Headline metric: query-level precision at recall >= 0.98. |
| `phase_e_writeback.py` | Refits the chosen configuration on all 22 entries and writes the repaired checkpoint. |
| `label_utils.py`, `cryoppp_gt.py` | Shared labelling and ground-truth access. |

```bash
export PYTHONPATH=<repo>/src/rapick/eval:$PYTHONPATH
cd $RAPICK_THIRD_PARTY/cryotransformer

python head_repair/build_train_stem_mapping.py \
    --train-dir <extracted>/train_val_test_data/train \
    --out-csv $RAPICK_WORK/head_repair/stem_to_id.csv

python predict.py --empiar 10081 --dump_hs $RAPICK_WORK/head_repair/hs_dumps
python head_repair/aggregate_hs_by_id.py \
    --dump-hs-dir $RAPICK_WORK/head_repair/hs_dumps \
    --mapping-csv $RAPICK_WORK/head_repair/stem_to_id.csv \
    --out-dir $RAPICK_WORK/head_repair/by_id

python head_repair/phase_d_train_heads.py --mode eos_sweep --epochs 15 \
    --hs-dumps-dir $RAPICK_WORK/head_repair/by_id \
    --out-dir $RAPICK_WORK/head_repair/cv

python head_repair/phase_e_writeback.py --arch linear --loss softmax \
    --eos-coef 0.1 --epochs 15 \
    --hs-dumps-dir $RAPICK_WORK/head_repair/by_id \
    --checkpoint-in $RAPICK_DATA/checkpoints/CryoTransformer_pretrained_model.pth \
    --checkpoint-out $RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth
```

Pass `--epochs 15` explicitly: both scripts' CLI defaults differ from the paper's
setting (`phase_e_writeback.py` defaults to 25).

The repaired head is written back as a two-class layer, not a one-logit one. The head is
trained as a single binary logit -- weighted BCE is numerically identical to two-class
softmax cross entropy once one output is pinned at 0 -- and then embedded as
`weight[0] = w`, `bias[0] = b`, `weight[1] = 0`, `bias[1] = 0`. Then
`softmax(-1)[..., 0] == sigmoid(w.h + b)`, so `predict.py`'s existing read-out works
unchanged. Feature standardization is folded into the weights, so inference needs no
normalization step.

**theta_0 is downloadable, so this whole stage can be skipped.** It is on the Hugging
Face model repository `rikrikrik/recon-aware-pick-weights` at
`weights/cryotransformer/eos_coef=0.1(default)/CryoTransformer_head_repaired.pth`.
Put it at `$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth`.

### Stage 2 -- fine-tuning

`finetune.py` trains from a GT-aligned STAR plus a directory of micrographs, building
targets directly. Labels are assigned as 0 = particle at this point and never read from
a COCO `category_id`, so the collision `datasets/micrograph.py` works around cannot
recur through this path. **Settings (Table S1, "fine-tuning"):**

| | |
| --- | --- |
| trained weights | all (`--finetune_mode head_decoder_encoder_resnet`) |
| training set | 40 micrographs |
| loss | Hungarian set loss |
| class weight | 1 |
| L1 box weight | 5 |
| GIoU weight | 2 |
| no-object weight | 0.1 |
| auxiliary decoder losses | 5 layers |
| optimizer | AdamW |
| learning rate | 1e-4 |
| backbone lr | 1e-5 |
| lr decay | x0.1 at epoch 24 |
| weight decay | 1e-4 |
| gradient clip | 0.1 |
| batch size | 8 micrographs |
| epochs | 50 |

"All weights" still leaves resnet layer1 frozen: `build_backbone()` in
`models/backbone.py` freezes it unconditionally, in every mode.

```bash
export PYTHONPATH=<repo>/src/rapick/eval:$PYTHONPATH
cd $RAPICK_THIRD_PARTY/cryotransformer

python finetune.py \
    --finetune_mode head_decoder_encoder_resnet \
    --resume $RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth \
    --images_dir $RAPICK_DATA/cryoppp/10081/micrographs \
    --star $RAPICK_WORK/loop/10081/round0/teacher.star \
    --box_size 154 --num_train_mrcs 40 \
    --output_dir $RAPICK_WORK/loop/10081/round1
```

`--resume` **must be the head-repaired checkpoint.** `finetune.py` loads every weight
as-is, `class_embed` and `query_embed` included, and reinitialises nothing; that is only
sound because the repaired head already uses the same 0 = particle scheme the fine-tuning
targets use. Pointing `--resume` at the released weights instead trains on top of the
degenerate head. The script guards against a *partial* checkpoint -- one missing
`class_embed` loads silently under `strict=False` and leaves the head at its random
init, which looks like a run that merely failed to converge -- but it cannot tell a
released checkpoint from a repaired one, and its own `--resume` default is upstream's
`pretrained_model/CryoTransformer_pretrained_model.pth`. Always pass `--resume`
explicitly.

`finetune.py` also implements a `lora` mode (`--finetune_mode lora`, plus `--lora_r`,
`--lora_alpha`, `--lora_dropout`), which injects rank-r adapters into the encoder and
decoder FFN. **The paper does not use it**, and the tooling for merging such adapters
back into a checkpoint is not part of this release.

## Environment

| Variable | Used for |
| --- | --- |
| `RAPICK_THIRD_PARTY` | where the upstream clone lives; the overlay is copied over it |
| `RAPICK_TEST_DATA` | root holding `<EMPIAR id>/images/`, the micrographs to pick |
| `RAPICK_DATA` | checkpoints, and the CryoPPP annotations the head repair labels against |
| `RAPICK_WORK` | dumps, the ground-truth parse cache, fine-tuning output |

None of them has a default. See [`../../../docs/CONFIGURATION.md`](../../../docs/CONFIGURATION.md).

Prediction output still lands under `output/predictions/` relative to the working
directory, which is upstream's own convention and is left alone. Run `predict.py` from a
directory you are willing to have written into, or move the result afterwards.
