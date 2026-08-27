# Configuration

Every path comes from an environment variable or `.env`. Nothing is hardcoded to a
machine: a script that cannot resolve one fails immediately, naming the variable.

## Environment variables

| Variable | Holds | Example |
| --- | --- | --- |
| `RAPICK_DATA` | Downloaded inputs: micrographs, annotations, pretrained weights. Read-mostly, and large (~1.6 TB for all four entries at full-set scale). | `/mnt/data/rapick-data` |
| `RAPICK_WORK` | Everything the pipeline produces: masks, filtered STAR, per-arm manifests, `metrics.json`. Grows during a run. | `/mnt/data/rapick-work` |
| `RAPICK_THIRD_PARTY` | Upstream checkouts fetched by `scripts/setup.sh`. | `<repo>/third_party` (default) |
| `RAPICK_ENVS` | Where the per-tool virtual environments are built. Point it at a local SSD: `uv` file locks hang on NFS. | `<repo>` (default, one `.venv` per env dir) |
| `RAPICK_GPU` | Default GPU index for the stages that take one. Every driver also accepts `--gpu`. | `0` |
| `RAPICK_TEST_DATA` | Root of the per-entry micrograph directories the picker reads, as `<id>/images/` — upstream CryoTransformer's contract, which we kept. `scripts/pick.sh` creates the links. | `$RAPICK_WORK/test_data` |

Set the main ones once, for example in `~/.rapick.env`, and source it before running
anything:

```bash
export RAPICK_DATA=/mnt/data/rapick-data
export RAPICK_WORK=/mnt/data/rapick-work
export RAPICK_GPU=0
```

### Escape hatches

Nothing below needs setting for a normal run. Each exists because one stage had a
reason to be overridable.

| Variable | Holds | Default |
| --- | --- | --- |
| `RAPICK_TOOL_*` | The path to one stage's entry point, when it is not where setup put it: `PREDICT`, `PREDICT_FULLSET`, `FINETUNE`, `MASK_FILTER`, `SCORER`, `SELECT_*`. | under `third_party/` or `src/` |
| `RAPICK_RECON_PROFILE` | A different CryoSPARC job-DAG profile. | `configs/cryosparc_v47.yaml` |
| `RAPICK_RECON_CONFIG` | The class_2D and reconstruction parameters, when `configs/` is laid out differently. One file, shared by every arm. | `configs/recon.yaml` |
| `RAPICK_RECON_MAX_INCOMPLETE_MICS`, `RAPICK_RECON_MAX_INCOMPLETE_CTF_MICS` | How many micrographs an import or a Patch CTF may silently drop before the preflight refuses to continue. **Raising this is how a run ends up reconstructing fewer micrographs than it reports**, which happened once here; raise it only with a reason. | `0` |
| `RAPICK_FT_MIN_FREE_MB`, `RAPICK_FT_MAX_WAIT_S` | How much free GPU memory a fine-tune waits for, and how long it waits before giving up. | 20000 MB, 7200 s |
| `RAPICK_LOCK_DIR` | Where the loop keeps its per-entry lock, so two rounds of the same entry cannot run at once. | `/tmp` |
| `RAPICK_ENTRIES` | Which entries `scripts/download.sh` and the scripts under `scripts/download/` fetch. Set it to one entry to avoid the 1.6 TB the four full depositions come to. | `10081 10093 10345 10532` |

## Layout under `RAPICK_DATA`

Created by `scripts/download.sh`. The four EMPIAR entries are 10081, 10093,
10345 and 10532.

```
$RAPICK_DATA/
├── cryoppp_tools/cryoppp/              the CryoPPP catalogue
├── cryoppp/<id>/
│   ├── micrographs/                    300 annotated .mrc
│   └── ground_truth/empiar-<id>_particles_selected.star
├── cryoppp_fullset/<id>/micrographs/   the full deposition, 997-1,873 .mrc
└── checkpoints/
    ├── CryoTransformer_pretrained_model.pth    released upstream weights
    ├── CryoTransformer_head_repaired.pth       theta_0 (Sec. S2)
    ├── micrograph_cleaner_defaultModel.h5      the contamination network
    └── loop_fb_round1_empiar_<id>.pth          the fb arm's checkpoint
```

## Layout under `RAPICK_WORK`

```
$RAPICK_WORK/
├── masks/<id>/                         triangular-blend contamination masks (.npz)
├── picks/<id>/<name>.star              GT-aligned picks, one file per stage they
│                                       have been through: cryotransformer.star,
│                                       cryotransformer_mask.star, fb.star, fb_mask.star
├── select2d/<project>_<job>_iter/      CryoSift cycle state and scores
├── loop/<id>/round<n>/                 a round's stars, teacher labels, checkpoint
└── empiar_<id>/<setting>/<name>/
    ├── manifest.json                   which CryoSPARC jobs were run
    └── metrics.json                    resolution, particle counts, job uids
```

`<setting>` is `annot` (the 300 annotated micrographs) or `full` (the whole
deposition). `<name>` is whatever a driver was told to record the run as; the ones the
paper reports are in [CONDITIONS.md](CONDITIONS.md).

## CryoSPARC connection: `.env`

Copy `.env.example` to `.env` at the repository root. It is git-ignored, is the only
file holding credentials, and every stage that talks to CryoSPARC reads it.

```
CRYOSPARC_LICENSE_ID=
CRYOSPARC_EMAIL=
CRYOSPARC_PASSWORD=
CRYOSPARC_HOST=localhost
CRYOSPARC_PORT=39000
CRYOSPARC_WORKER=
CRYOSPARC_PROJECT=
```

`CRYOSPARC_WORKER` is the worker hostname as CryoSPARC reports it
(`cryosparcm cli "get_scheduler_targets()"`); `CRYOSPARC_PROJECT` is the project uid to
write into, for example `P1`. Setup steps and the job chain are in
[CRYOSPARC.md](CRYOSPARC.md).
