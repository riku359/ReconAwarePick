# Bounded validation without CryoSPARC

These checks exercise code paths, not convergence or the paper's numerical results.
Do not start the default download: all assets together are approximately 1.6 TB.
Use a separate data/work directory, retain logs, and inspect GPU occupancy before
running one GPU process at a time. No CryoSPARC credentials are needed below.

## Install on Linux

Install `uv`, `git`, `curl`, and a Python 3.7 interpreter for the pinned picker
requirements. **uv cannot currently download Python 3.7 automatically.** For example,
create an isolated conda prefix with `conda create -p "$PWD/../python37" python=3.7`,
and prepend its `bin` directory to PATH for setup. Other stages use Python 3.10/3.12.
This preserves the existing dependency pins rather than silently upgrading PyTorch.
If the default uv interpreter/cache location is unwritable, set
`UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` to writable local directories.

```bash
bash scripts/setup.sh > setup.log 2>&1
```

The script verifies existing upstream checkout SHAs before applying the overlay.
If a checkout has another SHA, use a fresh `RAPICK_THIRD_PARTY` directory; do not
reset somebody else's checkout. When `RAPICK_THIRD_PARTY` is external, put
`<repo>/src/rapick/eval` on PYTHONPATH for head-repair scripts.
`RAPICK_ENVS` can move environments to a writable local disk.
Baseline environments are separate and are not needed for these checks.

## Small deterministic CPU checks

After setup, from the repository root:

```bash
PYTHONPATH=src envs/micrograph_cleaner/.venv/bin/python \
  -m unittest discover -s tests -p test_validation.py -v
envs/cryotransformer/.venv/bin/python \
  -m unittest discover -s tests -p test_query_selection.py -v
envs/cryotransformer/.venv/bin/python \
  -m unittest discover -s tests -p test_picker_training.py -v
bash scripts/check_release.sh
```

The tests use tiny generated fixtures. They check required-mask failures and
preservation of existing outputs, corrupt/nonfinite masks, exact 0.5 rejection,
the mask's Y flip, deterministic teacher sampling, import-prefix matching,
saved-particle coordinate restoration, and explicit legacy query indices.
Fixtures are retained in the system temporary directory for inspection.
They do not claim to exercise a neural network.

## One real picker image

Set `RAPICK_DATA` and `RAPICK_WORK` to separate writable input/output directories.
Fetch only the named assets (about 930 MB together), without authentication:

```bash
mkdir -p "$RAPICK_DATA/checkpoints" "$RAPICK_DATA/smoke-images"
curl -fL --retry 2 \
  'https://huggingface.co/rikrikrik/recon-aware-pick-weights/resolve/main/weights/cryotransformer/eos_coef%3D0.1(default)/CryoTransformer_head_repaired.pth' \
  -o "$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth"
curl -fL --retry 2 \
  'https://ftp.ebi.ac.uk/empiar/world_availability/10081/data/micrographs/HCN1apo_0002_2xaligned.mrc' \
  -o "$RAPICK_DATA/smoke-images/HCN1apo_0002_2xaligned.mrc"
```

Resolve the roots to absolute paths before entering the upstream checkout.
Run `nvidia-smi` and choose an available GPU; below, `GPU` is its physical index.
Use a 24 GB GPU such as an RTX A5000; allow a few minutes and roughly 2 GB
of output/cache space. The upstream backbone may additionally download ImageNet
weights into `TORCH_HOME` even though the full checkpoint replaces them.

```bash
REPO="$PWD"
GPU=0  # replace after checking occupancy
cd "${RAPICK_THIRD_PARTY:-$REPO/third_party}/cryotransformer"
CUDA_VISIBLE_DEVICES="$GPU" "$REPO/envs/cryotransformer/.venv/bin/python" predict.py \
  --image-dir "$RAPICK_DATA/smoke-images" --empiar 10081 \
  --resume "$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth" \
  --selection legacy_idxfix --gt-format --save_micrographs_with_encircled_proteins N \
  --debug_dump "$RAPICK_WORK/smoke/idxfix" --remarks smoke_idxfix
```

`--image-dir` bypasses the `<data_root>/<entry>/images` layout, including for
head-repair training JPGs. An absent/empty input fails before constructing the model.
Prediction STAR files remain under the upstream `output/predictions` directory.
Repeat with `--selection legacy`, a different `--remarks`, and a different dump
folder to inspect the algorithm difference.

The two selection modes can produce different query IDs and post-NMS coordinates
on the same image. Compare the diagnostics rather than expecting identical picks.
GPU/library differences can also perturb boundary decisions. This image is outside
the annotated subset, so this check does not measure accuracy.

## Selection semantics and round 0

`pick.sh` now uses `predict.py --selection legacy_idxfix`, matching `run_loop.py`
and `repick_fullset.py`, with 600 queries, quantile 0.25 and NMS 0.7.
`pick.sh --selection legacy` preserves explicit upstream reproduction;
`predict_fullset.py` also remains a legacy implementation.

Legacy applies a score-rank Boolean mask to raw query slots; idxfix gathers the
ranked query IDs and deduplicates them. These can select different boxes even
when the kept count before deduplication agrees. Neither constitutes a new paper
result. Do not label a legacy run as interchangeable with an idxfix run.
The existing round-0 numerical gate is retained and still stops the loop on drift.
Its historical reference has **not** been re-established over the full annotated
sets in this validation. A gate failure needs investigation of selection mode,
checkpoint, input coverage and preprocessing; do not replace its reference values
with smoke-test numbers or disable it to proceed.

## Cached masks and failures

`filter_star_from_masks.py` requires every referenced mask to be readable and
finite, including on reruns, and exits nonzero before replacing STAR outputs if
validation fails. Old decisions are recomputed for the current candidates and
threshold; they must not be reused across feedback rounds merely by image name.
The input masks, not the decisions JSONL, are the reusable cache.
`--allow-missing-masks` explicitly restores permissive retention, with error counts
in the summary; this is not successful strict contamination removal.

At the centre, the stored mask is flipped vertically, coordinates are rounded with
Python's `round`, and a value **>= 0.5** removes a pick. Out-of-frame coordinates
retain the existing permissive boundary behavior. Float16 storage can change
near-threshold decisions. Compare direct and saved-mask decisions on the same
candidates when checking a cache; one matching image does not establish identical
decisions for every image.

## Saved class averages and selected-particle exports

Use an existing completed export directory containing matching
`J<N>_<iteration>_class_averages.mrc`, `_class_averages.cs`, and `_particles.cs`.
No particle image stack is needed for scoring:

```bash
PYTHONPATH=src envs/cryosift/.venv/bin/python -m rapick.select2d.score_class2d \
  --job J15 --job-dir /path/to/saved/J15 --out-root "$RAPICK_WORK/scores"
```

With `--job-dir`, the CLI does not read a credentials file or require a project.
Use a saved **selected** particle export for the feedback teacher:

```bash
PYTHONPATH=src envs/cryosift/.venv/bin/python -m rapick.loop.export_teacher_star \
  --particles-cs /path/to/selected.cs --empiar 10081 --seed 1 --num-mics 2 \
  --out-dir "$RAPICK_WORK/smoke/teacher"
```

Optionally pass `--extract-cs` and `--input-star` together to check coordinate
restoration against the imported picks. Byte-string paths in raw `.cs` files are
decoded. Sampling records `train_mics.txt`; use this same list with
`rapick.loop.make_gt_teacher --mics-from` for the GT arm. A missing GT image is
reported in `dropped_empty_gt_mics` rather than counted as a labelled image.
Server-side reclassification, selection jobs and reconstruction remain untested.

For legacy MRC exports rejected for missing MAP/machine-stamp headers, keep the
originals and write canonical copies to a **new** directory before inference.
The following runs with the picker environment and preserves pixel values (it does
not repair corrupt pixel data):

```python
from pathlib import Path
import mrcfile
import numpy as np

source = Path("/path/to/original.mrc")
target = Path("/path/to/new-directory/original.mrc")
target.parent.mkdir(parents=True, exist_ok=True)
with mrcfile.open(str(source), permissive=True) as original:
    pixels = original.data.copy()
assert pixels.ndim == 2 and np.isfinite(pixels).all()
with mrcfile.new(str(target), overwrite=False) as canonical:
    canonical.set_data(pixels)
with mrcfile.open(str(target)) as check:
    np.testing.assert_array_equal(check.data, pixels)
```

## Short training and head repair

`scripts/repair_head.sh --train-dir DIR` now sends DIR to feature extraction via
`--image-dir`; `--empiar all` is only an output label, not a training-data resolver.
The documented full repair still requires all 22 training entries and their GT,
followed by grouping, CV and writeback. Do not overwrite the published theta_0
with weights fitted on a smoke fixture.
For an explicitly separate small head-fit test, `phase_e_writeback.py --ids ID ...`
loads only the named feature groups, requires at least two images per entry and
records `ids` and `smoke_subset` in the output checkpoint/JSON. The default remains
all 22 training entries. A held-out evaluation entry can test mechanics but is
not evidence of reproducing the original training split or head quality.

For fine-tuning, provide two or more **real annotated images** and a matching teacher
STAR, an explicit output directory and `--batch_size 1 --num_workers 0 --epochs 1
--val_fraction 0.5`. Use `--finetune_mode head_decoder_encoder_resnet` and the
published theta_0 via `--resume`. This is one train image and one validation image,
not the paper's 40/10 split. Use an available 24 GB GPU and allow several minutes
and several GB per checkpoint; inspect finite train/validation losses and reload
`checkpoint.pth` for inference. The first residual stage must remain unchanged.

For example, with matched images in `smoke-images` and the generated GT teacher:

```bash
REPO="$PWD"
cd "${RAPICK_THIRD_PARTY:-$REPO/third_party}/cryotransformer"
CUDA_VISIBLE_DEVICES="$GPU" "$REPO/envs/cryotransformer/.venv/bin/python" finetune.py \
  --images_dir "$RAPICK_DATA/smoke-images" \
  --star "$RAPICK_WORK/smoke/teacher/teacher_gt.star" \
  --resume "$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth" \
  --output_dir "$RAPICK_WORK/smoke/gt-finetune" \
  --finetune_mode head_decoder_encoder_resnet --backbone resnet152 \
  --num_queries 600 --box_size 154 --batch_size 1 --num_workers 0 \
  --epochs 1 --val_fraction 0.5 --device cuda:0
```

Use the entry's real particle box size rather than assuming 154 for other entries.
Images absent from the teacher do not supply training samples; ensure at least two
matched images. Replace the STAR/output paths for the selected-export arm.

`--resume` alone initializes a **new** feedback training run from weights.
`--resume-training` additionally restores optimizer/scheduler state and starts at
checkpoint epoch + 1; set `--epochs` to the desired total. This supports continuation
at a saved epoch boundary, not recovery of an interrupted partial batch or a claim
of bitwise-identical RNG continuation. Never resume another feedback round from
its preceding round by accident.

## Validation scope and input integrity

The checks above cover execution with small inputs. They do not establish
convergence, the full 22-entry head fit/CV, all 300-image round-0 gates,
full-deposition accuracy, or reproduction of the published GT-arm result.
CryoSPARC job creation, reclassification and reconstruction require a separate
server-enabled validation. Historical baseline versus round-0 metrics still
require confirmation against the original experiment records; keep the existing
round-0 references and drift checks.

Reference SHA-256 checksums for the public assets used in the one-image command:

- Repaired checkpoint: `0e29e53509e45ad846e5a68c07dc25c854adef5f980b0c0f286689767d217fc3`.
- `HCN1apo_0002_2xaligned.mrc`: `53001e4166b1e8ec9ab39fcd7bba6b99a4f878c1ec13fdbf8378de6b2f7e8648`.

Public asset URLs use `main`; verify checksums rather than assuming the remote
contents never change.
