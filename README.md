# ReconAwarePick

<p align="center">
  <a href="https://huggingface.co/rikrikrik/recon-aware-pick-weights"><img src="https://img.shields.io/badge/Hugging%20Face-weights-yellow.svg" alt="Weights"></a>
  <a href="https://huggingface.co/datasets/rikrikrik/recon-aware-pick-data"><img src="https://img.shields.io/badge/Hugging%20Face-data-yellow.svg" alt="Data"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<div align="center">

**Reconstruction-Aware Cryo-EM Particle Picking**

Riku Itsuji<sup>1</sup> · Yuanhao Wang<sup>2</sup> · Xingjian Li<sup>2</sup> · Seonghui Min<sup>2</sup> · Hideo Saito<sup>1</sup> · Min Xu<sup>2</sup>

<sub>
<sup>1</sup> Keio University &nbsp;
<sup>2</sup> Carnegie Mellon University
</sub>

</div>

Extracting a clean particle stack from cryo-EM micrographs is usually split into
three sub-tasks — particle picking, contamination removal, and 2D class selection —
each trained and evaluated on its own, none optimized for the 3D reconstruction that
consumes their output. We treat them as one selection problem posed against
reconstruction quality.

## Overview

![pipeline](assets/pipeline.png)

CryoTransformer picks permissively, a MicrographCleaner mask discards the candidates
that fall on contamination, CryoSift selects 2D classes by a continuous quality score,
and the surviving particles fine-tune the picker. The pipeline reaches a better
resolution than every picker we compare, on all four EMPIAR entries of
CryoTransformer's independent test set: EMPIAR-10081, 10093, 10345 and 10532.

## Requirements

- **CryoSPARC v4.7.x** — not installed by this repository. See
  [docs/CRYOSPARC.md](docs/CRYOSPARC.md).
- **One GPU** — every stage occupies a single GPU. The paper's runs used NVIDIA
  RTX A5000 cards with 24 GB each.
- `git`, `curl`, and [`uv`](https://docs.astral.sh/uv/).

## Installation

The stages need mutually incompatible Python environments, so there is one per stage
rather than one for the repository. `scripts/setup.sh` clones the pinned upstream
code and builds them all from the committed lockfiles under `envs/`.

```bash
git clone https://github.com/riku359/ReconAwarePick.git
cd ReconAwarePick
cp .env.example .env                  # CryoSPARC licence id, e-mail, password
export RAPICK_DATA=/path/to/data      # inputs
export RAPICK_WORK=/path/to/work      # outputs
bash scripts/setup.sh
```

| Stage | Environment | Python | Framework |
| --- | --- | --- | --- |
| picking, fine-tuning | `envs/cryotransformer` | 3.7 | torch 1.13.1+cu117 |
| contamination masking | `envs/micrograph_cleaner` | 3.10 | TensorFlow 2.16 + Keras 3 |
| 2D class selection | `envs/cryosift` | 3.12 | torch 2.6 (CPU) |
| reconstruction | `envs/recon` | 3.10 | cryosparc-tools 4.7 |

Every path comes from an environment variable, and a script that cannot resolve one
fails naming it rather than falling back to a default; the contract is in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md). Keep the `uv` cache on a local SSD:
its file locks hang on NFS.

## Data

```bash
bash scripts/download.sh
```

It takes no arguments and fetches everything. One script per source lives under
[scripts/download/](scripts/download/), and `download.sh` runs them in name order;
run one directly to re-fetch just that source. All four entries at full-set scale
come to about 1.6 TB — `export RAPICK_ENTRIES=10081` restricts every step to one
entry.

```
$RAPICK_DATA/
├── cryoppp_tools/cryoppp/                  the CryoPPP catalogue
├── cryoppp/<id>/
│   ├── micrographs/                        300 annotated .mrc
│   └── ground_truth/
│       └── empiar-<id>_particles_selected.star
├── cryoppp_fullset/<id>/
│   └── micrographs/                        the full deposition, 997-1,873 .mrc
└── checkpoints/
    ├── CryoTransformer_pretrained_model.pth        released upstream weights
    ├── CryoTransformer_head_repaired.pth           theta_0 (Sec. S2)
    ├── micrograph_cleaner_defaultModel.h5          contamination network
    └── loop_fb_round1_empiar_<id>.pth              the fb arm's checkpoint

$RAPICK_WORK/
├── masks/<id>/                             contamination masks, one .npz per micrograph
├── picks/<id>/                             picks, one STAR per stage they have been through
│   ├── cryotransformer.star                theta_0's candidates
│   ├── cryotransformer_mask.star           the same, after contamination removal
│   ├── {cryolo,topaz,cryosegnet}.star      the comparison pickers' candidates
│   └── fb.star, fb_mask.star               a run's own, once it has picked
├── select2d/<project>_<job>_iter/          CryoSift cycle state and scores
├── loop/<id>/round<n>/                     a round's stars, teacher labels, checkpoint
└── empiar_<id>/<setting>/<name>/
    ├── manifest.json                       which CryoSPARC jobs were run
    └── metrics.json                        resolution, particle counts, job uids
```

`<id>` is one of 10081, 10093, 10345 and 10532; `<setting>` and `<name>` are defined in
[docs/CONDITIONS.md](docs/CONDITIONS.md). `masks/` and `picks/` arrive with the download,
and a run fills in the rest as it goes. [docs/DATA.md](docs/DATA.md) has where each file
comes from, including the two ways a downloaded `.mrc` can be silently corrupt and how to
check.

## Quick Start

Each driver is one transform, with its input and its output named on the command line:

```
pick -> contamination_removal -> 2d_classification -> select2d -> reconstruct
-> finetune -> pick -> ...
```

### Ours: the full deposition, picked with the round-1 fine-tuned checkpoint

997 micrographs of EMPIAR-10081. `download.sh` supplied the checkpoint the feedback
loop produced, and the contamination masks.

```bash
P=$RAPICK_WORK/picks/10081
CKPT=$RAPICK_DATA/checkpoints/loop_fb_round1_empiar_10081.pth

# re-pick the deposition with the fine-tuned checkpoint (Sec. 3.2)
bash scripts/pick.sh --entry 10081 --checkpoint $CKPT --out $P/fb.star

# drop the picks that land on contamination, against the downloaded masks (Sec. 3.3)
bash scripts/contamination_removal.sh --entry 10081 --star $P/fb.star  # -> $P/fb_mask.star

# import -> patch CTF -> extract -> class_2D (K=50). Prints class_2D J<n>.
bash scripts/2d_classification.sh --entry 10081 --star $P/fb_mask.star

# CryoSift's three cycles over those classes. Prints the final select_2D J<m>. Resumable.
bash scripts/select2d.sh --entry 10081 --class2d J<n>

# ab-initio x3 -> refine x3 -> best by GSFSC 0.143 -> local resolution -> metrics.json
bash scripts/reconstruct.sh --entry 10081 --parent fb_mask --name fb
```

Result: `$RAPICK_WORK/empiar_10081/full/fb/metrics.json` — resolution, particle counts,
and the CryoSPARC job uids behind them.

## Training the checkpoint yourself: the feedback loop

On the 300 annotated micrographs (`--setting annot`), no reconstruction — at that scale
one does not resolve a round from the next (Sec. 4.3). One round is the five drivers in
order, and the checkpoint that comes out is what the next round picks with:

    theta_{n+1} = FineTune(theta_0; S_n)

theta_0 every round, never the checkpoint that just picked.

```bash
R=$RAPICK_WORK/loop/10081/round0

# 1. pick the 300 with the current checkpoint (round 0: theta_0)
bash scripts/pick.sh --entry 10081 --setting annot --out $R/cryotransformer.star

# 2. drop the picks that land on contamination
bash scripts/contamination_removal.sh --entry 10081 --setting annot \
    --star $R/cryotransformer.star                          # -> $R/cryotransformer_mask.star

# 3. classify what survives
bash scripts/2d_classification.sh --entry 10081 --setting annot \
    --star $R/cryotransformer_mask.star --name fb_r0

# 4. select 2D classes
bash scripts/select2d.sh --entry 10081 --setting annot --class2d J<n>

# 5. 50 micrographs of survivors become the labels; fine-tune theta_0 on them
bash scripts/finetune.sh --entry 10081 --select2d J<m> --parent fb_r0 \
    --round-dir $R --out $RAPICK_WORK/loop/10081/models/model_1.pth
```

Repeat from step 1 with `--checkpoint .../models/model_1.pth` and `R=.../round1`. The
paper runs three rounds and reports round 1.

`scripts/loop.sh` runs the same five steps with the bookkeeping around them — a per-round
directory, a resumable step record, a lock, and the per-round diagnostics of Table 6:

```bash
bash scripts/loop.sh --entry 10081 --rounds 0-2   # -> loop/10081/models/model_1.pth
```

Either way, re-pick the full deposition with the resulting checkpoint and follow the
**Ours** flow above.

## Optional

### Starting from the picker itself

`download.sh` supplies theta_0, the masks and the candidates.

```bash
# theta_0: refit CryoTransformer's degenerate classification head (Sec. S2).
# Needs the 22-entry CryoPPP training split, which this repository does not download.
bash scripts/repair_head.sh --train-dir <extracted>/train_val_test_data/train

# inference with theta_0 over the whole deposition (Sec. 3.2)
bash scripts/pick.sh --entry 10081                     # -> $P/cryotransformer.star

# predict the masks if $RAPICK_WORK/masks/10081/ is empty, then filter (Sec. 3.3)
bash scripts/contamination_removal.sh --entry 10081    # -> $P/cryotransformer_mask.star
```

### The ablation arms

The same picks as Ours, but from theta_0 instead of the fine-tuned checkpoint.
`download.sh` already fetched `cryotransformer_mask.star`, so the two masked rows need
no picking at all.

```bash
# +mask: the classified stack as it stands, no 2D class selection
bash scripts/2d_classification.sh --entry 10081 --star $P/cryotransformer_mask.star
bash scripts/reconstruct.sh       --entry 10081 --name cryotransformer_mask

# +both: the selection over that same class_2D
bash scripts/select2d.sh          --entry 10081 --class2d J<n>
bash scripts/reconstruct.sh       --entry 10081 --parent cryotransformer_mask

# the two unmasked rows start from cryotransformer.star (downloaded, or pick.sh above)
bash scripts/2d_classification.sh --entry 10081 --star $P/cryotransformer.star
bash scripts/reconstruct.sh       --entry 10081 --name cryotransformer
```

A run over all four entries and every arm takes weeks. The name of a STAR, and of the
directory a result lands in, says which stages the particles have been through:
`cryotransformer.star` → `cryotransformer_mask.star` → `cryotransformer_mask_select`.
The paper's own arms, and the three caveats that bound what a resolution here means, are
in [docs/CONDITIONS.md](docs/CONDITIONS.md).

### Comparison pickers

crYOLO, Topaz and CryoSegNet are built only by `bash scripts/setup.sh --baselines`, and
crYOLO needs conda. `download.sh` fetches their published candidates instead, so
Table 2 and Table S2 can be checked without installing any of them. See
[docs/BASELINES.md](docs/BASELINES.md).

## Documentation

| Document | Covers |
| --- | --- |
| [docs/CRYOSPARC.md](docs/CRYOSPARC.md) | installing CryoSPARC v4.7 and the job chain this repository drives |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | every environment variable, and the `.env` the CryoSPARC stages read |
| [docs/DATA.md](docs/DATA.md) | what is downloaded, and how a `.mrc` can be silently corrupt |
| [docs/CONDITIONS.md](docs/CONDITIONS.md) | the arm names, what a resolution here means, and the renames if you ran this before |
| [docs/BASELINES.md](docs/BASELINES.md) | the three comparison pickers |
| [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) | what is still pending, and what this repository deliberately does not contain |

Each stage's own README has the exact flags and the traps:
[picker](src/rapick/picker/README.md) ·
[cleaner](src/rapick/cleaner/README.md) ·
[select2d](src/rapick/select2d/README.md) ·
[loop](src/rapick/loop/README.md) ·
[recon](src/rapick/recon/README.md) ·
[eval](src/rapick/eval/README.md)

## License

Our own code is released under the [MIT License](LICENSE).

Three files of [CryoTransformer](https://github.com/jianlin-cheng/CryoTransformer)
(MIT, © 2023 Jianlin Cheng) ship in modified form under `src/rapick/picker/overlay/`,
because the paper depends on those modifications; the changes are readable as diffs in
`src/rapick/picker/patches/`. Everything else is cloned at a pinned commit by
`scripts/setup.sh` and is not redistributed here — see `repos.lock.yaml`:
[MicrographCleaner](https://github.com/rsanchezgarc/micrograph_cleaner_em)
(Apache-2.0), [CryoSift](https://cryosift.org) (Apache-2.0),
[Topaz](https://github.com/tbepler/topaz) (**GPL-3.0**),
[CryoSegNet](https://github.com/jianlin-cheng/CryoSegNet) (MIT), and
[crYOLO](https://cryolo.readthedocs.io/) (MPI Dortmund Complimentary Science Software
License, non-commercial, not redistributable).

Micrographs and annotations come from
[CryoPPP](https://github.com/BioinfoMachineLearning/cryoppp) (CC-BY-4.0) and from
[EMPIAR](https://www.ebi.ac.uk/empiar/). The STAR files and masks we publish on Hugging
Face are derived from them and are redistributed under CC-BY-4.0 with attribution.

## Citation

To be updated once the proceedings citation is assigned. If you use this code, please
also cite **CryoTransformer**, **MicrographCleaner**, **CryoSift**, **CryoPPP** and
**CryoSPARC**.
