# Configuration

Every path this repository needs comes from one of five environment variables
and a `.env` file. Nothing is hardcoded to a machine: a script that cannot
resolve one of these fails immediately with the variable's name, rather than
falling back to a default that happens to exist on the author's server.

## Environment variables

| Variable | Holds | Example |
| --- | --- | --- |
| `RAPICK_DATA` | Downloaded inputs: micrographs, annotations, pretrained weights. Read-mostly, and large (about 1.6 TB for all four entries at full-set scale). | `/mnt/data/rapick-data` |
| `RAPICK_WORK` | Everything the pipeline produces: masks, filtered STAR, per-condition manifests, `metrics.json`. Grows during a run. | `/mnt/data/rapick-work` |
| `RAPICK_THIRD_PARTY` | Upstream checkouts fetched by `scripts/00_setup.sh`. | `<repo>/third_party` (default) |
| `RAPICK_ENVS` | Where the per-tool virtual environments are built. Point it at a local SSD: `uv` file locks hang on NFS. | `<repo>` (default, one `.venv` per env dir) |
| `RAPICK_GPU` | Default GPU index for the stages that take one. Every driver also accepts `--gpu`. | `0` |
| `RAPICK_TEST_DATA` | Root of the per-entry micrograph directories the picker reads, laid out as `<id>/images/`. Upstream CryoTransformer resolves its input that way and we kept the contract; `scripts/03_pick.sh` creates the links. | `$RAPICK_WORK/test_data` |

Set them once, for example in `~/.rapick.env`, and source it before running
anything:

```bash
export RAPICK_DATA=/mnt/data/rapick-data
export RAPICK_WORK=/mnt/data/rapick-work
export RAPICK_GPU=0
```

## Layout under `RAPICK_DATA`

Created by `scripts/01_download_data.sh`. The four EMPIAR entries are 10081,
10093, 10345 and 10532.

```
$RAPICK_DATA/
├── cryoppp/<id>/
│   ├── micrographs/                    300 annotated .mrc
│   └── ground_truth/empiar-<id>_particles_selected.star
├── cryoppp_fullset/<id>/micrographs/   the full deposition, 997-1,873 .mrc
└── checkpoints/
    ├── CryoTransformer_pretrained_model.pth    released upstream weights
    └── CryoTransformer_head_repaired.pth       theta_0 (Sec. S2)
```

## Layout under `RAPICK_WORK`

```
$RAPICK_WORK/
├── masks/<id>/                         triangular-blend contamination masks (.npz)
├── picks/<id>/<condition>.star         GT-aligned picks, one per condition
├── select2d/<project>_<job>_iter/      CryoSift cycle state and scores
├── loop/<id>/round<n>/                 teacher labels, checkpoints, metrics
└── empiar_<id>/<setting>/<condition>/
    ├── manifest.json                   which CryoSPARC jobs were run
    └── metrics.json                    resolution, particle counts, job uids
```

`<setting>` is `annot` (the 300 annotated micrographs) or `full` (the whole
deposition). `<condition>` is one of the names in
[PAPER_TO_CODE.md](PAPER_TO_CODE.md).

## CryoSPARC connection: `.env`

Copy `.env.example` to `.env` at the repository root and fill it in. It is
git-ignored. This is the only file that holds credentials, and every stage that
talks to CryoSPARC reads it from here.

```
CRYOSPARC_LICENSE_ID=
CRYOSPARC_EMAIL=
CRYOSPARC_PASSWORD=
CRYOSPARC_HOST=localhost
CRYOSPARC_PORT=39000
CRYOSPARC_WORKER=
CRYOSPARC_PROJECT=
```

`CRYOSPARC_WORKER` is the name of your worker lane as CryoSPARC reports it
(`cryosparcm cli "get_scheduler_targets()"`). `CRYOSPARC_PROJECT` is the project
uid the pipeline writes into, for example `P1`; create it once in the CryoSPARC
web interface. See [CRYOSPARC.md](CRYOSPARC.md) for the version requirement and
the job chain.
