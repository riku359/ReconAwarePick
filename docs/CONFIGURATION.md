# Configuration

The variables every script resolves its paths through, and the `.env` the CryoSPARC
stages read. A script that cannot resolve one fails immediately, naming it.

## Environment variables

| Variable | Holds | Example |
| --- | --- | --- |
| `RAPICK_DATA` | Downloaded inputs: micrographs, annotations, pretrained weights. Read-mostly, and large. | `/mnt/data/rapick-data` |
| `RAPICK_WORK` | Everything the pipeline produces. Grows during a run. | `/mnt/data/rapick-work` |
| `RAPICK_ENTRIES` | Which entries every step covers. One entry avoids the 1.6 TB the four full depositions come to. | `10081 10093 10345 10532` |
| `RAPICK_GPU` | Default GPU index for the stages that take one. Every driver also accepts `--gpu`. | `0` |
| `RAPICK_THIRD_PARTY` | Upstream checkouts fetched by `scripts/setup.sh`. | `<repo>/third_party` (default) |
| `RAPICK_ENVS` | Where the per-tool virtual environments are built. Keep it on a local SSD. | `<repo>` (default, one `.venv` per env dir) |
| `RAPICK_TEST_DATA` | Root of the per-entry micrograph directories the picker reads, as `<id>/images/` — upstream CryoTransformer's contract, which we kept. `scripts/pick.sh` creates the links. | `$RAPICK_WORK/test_data` |

The first four are worth setting once, for example in `~/.rapick.env`, and sourcing
before running anything.

## Escape hatches

Nothing here needs setting for a normal run. Each exists because one stage had a reason
to be overridable.

| Variable | Holds | Default |
| --- | --- | --- |
| `RAPICK_RECON_MAX_INCOMPLETE_MICS`, `RAPICK_RECON_MAX_INCOMPLETE_CTF_MICS` | How many micrographs an import or a Patch CTF may silently drop before the preflight refuses to continue. **Raising this is how a run ends up reconstructing fewer micrographs than it reports**, which happened once here; raise it only with a reason. | `0` |
| `RAPICK_FT_MIN_FREE_MB`, `RAPICK_FT_MAX_WAIT_S` | How much free GPU memory a fine-tune waits for, and how long it waits before giving up. Worth raising the wait on a shared card. | 20000 MB, 7200 s |
| `RAPICK_RECON_PROFILE`, `RAPICK_RECON_CONFIG` | A different CryoSPARC job-DAG profile, and a different class_2D/reconstruction parameter file. One config, shared by every arm. | `configs/cryosparc_v47.yaml`, `configs/recon.yaml` |
| `RAPICK_TOOL_*` | The path to one stage's entry point, when it is not where setup put it: `PREDICT`, `PREDICT_FULLSET`, `FINETUNE`, `MASK_FILTER`, `SCORER`, `SELECT_*`. | under `third_party/` or `src/` |
| `RAPICK_LOCK_DIR` | Where the loop keeps its per-entry lock, so two rounds of the same entry cannot run at once. | `/tmp` |

## Credentials

CryoSPARC's licence id, login and project uid live in `.env` at the repository root,
which is git-ignored and is the only file holding credentials. Copy `.env.example` to
`.env` and fill it in as [CRYOSPARC.md](CRYOSPARC.md) describes.
