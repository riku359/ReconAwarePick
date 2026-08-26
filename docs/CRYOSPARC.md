# CryoSPARC

Every stage of this pipeline from particle extraction onward runs as a CryoSPARC job:
extraction, 2D classification, ab-initio reconstruction, homogeneous refinement, and
local resolution. **CryoSPARC is not installed by this repository, and it is not
optional.**

## Version: v4.7.x, and why not v5

The pipeline targets **CryoSPARC v4.7.x**. The reason is the GPU driver, not the science.

| CryoSPARC | Bundled CUDA | Minimum NVIDIA driver |
| --- | --- | --- |
| v4.4 – v4.7 | 11.8 | 520.61.05 |
| v5.0.x | 12.8 | 570.26 |

v5.0.x requires driver >= 570.26 (CUDA 12.8), which many shared machines and clusters do
not have and cannot be upgraded to without a maintenance window. v4.4 through v4.7 bundle
CUDA 11.8, so driver >= 520.61.05 is enough — and a CUDA 12.x driver runs the bundled
11.8 runtime fine, because NVIDIA drivers are forward compatible with older runtimes.
Targeting v4.7 therefore costs nothing and works on far more hardware.

`configs/cryosparc_v47.yaml` records the job types and input/output port names this
pipeline wires together, as they exist on v4.7.1. A different release can rename them.
After any upgrade, or on a new site, re-verify the profile before spending GPU time:

```bash
PYTHONPATH=src envs/recon/.venv/bin/python \
  src/rapick/recon/scripts/smoke_probe.py --env .env --project <throwaway project uid>
```

It builds each candidate job, reads its ports and params, prints them, and deletes it.
Nothing is queued, so it costs no GPU time — but it does create jobs, so point it at a
**throwaway project**, never one holding results.

## `cryosparc-tools` must match the server

The Python client talks to the server over an API whose shape changes between minor
versions. `cryosparc-tools` must match your server's **minor** version:

```
cryosparc-tools>=4.7,<4.8      # for a v4.7.x server
```

That pin is in `envs/recon/pyproject.toml` and locked in `envs/recon/uv.lock`. A v4.6 or
a v5.0 server needs the matching client and a re-locked environment; a mismatched client
fails in ways that look like data problems rather than version problems.

## Licence

CryoSPARC needs a licence id, which you request from Structura Biotechnology. **A free
non-commercial licence is enough** for everything in this repository. Follow the official
installation guide at <https://guide.cryosparc.com/> for the licence request and for the
master and worker installation; nothing here replaces it, and nothing here needs a
modified CryoSPARC.

## Pointing this repository at your instance

Copy `.env.example` to `.env` at the repository root and fill it in. `.env` is
git-ignored and is the only file in this repository that holds credentials.

```
CRYOSPARC_LICENSE_ID=
CRYOSPARC_EMAIL=
CRYOSPARC_PASSWORD=
CRYOSPARC_HOST=localhost
CRYOSPARC_PORT=39000
CRYOSPARC_WORKER=
CRYOSPARC_PROJECT=
```

`CRYOSPARC_HOST` and `CRYOSPARC_PORT` are your master's hostname and base port, from
`cryosparc_master/config.sh` (`CRYOSPARC_MASTER_HOSTNAME`, `CRYOSPARC_BASE_PORT`). The
default base port is 39000 and it is the port you reach the web interface on.

### The project

`CRYOSPARC_PROJECT` is the uid of the project this pipeline writes into, for example
`P1`. Create it once:

1. Open the web interface at `http://<CRYOSPARC_HOST>:<CRYOSPARC_PORT>/`.
2. **Projects** → **Create Project**. Give it a title and a directory on a disk with
   room for the particle stacks — they are much larger than the micrographs.
3. Read the uid off the project card (`P1`, `P2`, …) and put it in `.env`.

One project holds everything. Inside it the pipeline creates one workspace per
(entry, scale), titled `<empiar id>_<setting>` — `10081_full`, `10081_annot` — and
creates them itself; you do not need to make workspaces by hand.

The project uid is never written into a config file. `--project` overrides `.env` for a
single run, and that is the only other place it can come from.

### The worker lane

`CRYOSPARC_WORKER` is the name of the worker lane exactly as your CryoSPARC reports it.
Ask the server:

```bash
cryosparcm cli "get_scheduler_targets()"
```

Each entry has a `hostname` and a `lane`. `CRYOSPARC_WORKER` is the hostname the jobs
should be pinned to. Pinning matters: on an install with more than one worker, an
unpinned CPU-only job can be scheduled onto a node that is unreachable and fail at launch
with an ssh error, and an unpinned GPU job can land on a card another user has already
filled and die with `CUDA_ERROR_OUT_OF_MEMORY` mid-run — CryoSPARC's scheduler only knows
about GPUs reserved by its own jobs. `--worker` overrides `.env` for a single run.

## The job chain (Fig. S1)

One chain per condition. Import Micrographs and Patch CTF run once per (entry, scale) and
are reused by every condition of that entry, so all conditions are compared over identical
CTF estimates. Job types are the CryoSPARC keys, as declared in
`configs/cryosparc_v47.yaml`.

1. **Import Micrographs** — `import_micrographs`. The micrograph glob plus the entry's
   optics: pixel size, accelerating voltage, spherical aberration, dose. *Shared.* CPU.
2. **Patch CTF Estimation (Multi)** — `patch_ctf_estimation_multi`. Per-micrograph CTF.
   *Shared.* GPU.
3. **Import Particle Stack** — `import_particles` with "Ignore raw data", so only the
   coordinates are imported and connected to the micrographs from step 1. CPU.
4. **Extract From Micrographs (Multi)** — `extract_micrographs_multi`. Cuts each
   condition's boxes out of the CTF-estimated micrographs. GPU.
5. **2D Classification** — `class_2D`. 50 classes, 20 full iterations, one seed shared by
   all three reconstruction trials so 2D is never a source of difference between them.
   GPU.
6. **Select 2D Classes** — `select_2D`, *only for the conditions that select*
   (`select`, `both`, `fb`, `cryosegnet_both`). These do not come off step 5 directly:
   they reconstruct the final selection of CryoSift's iterative workflow, which sits
   several Select 2D / 2D Classification hops above it. That part is
   `src/rapick/select2d/`, not this stage.
7. **Ab-Initio Reconstruction x 3** — `homo_abinit`, one class, forked over seeds 0, 1
   and 2. GPU.
8. **Homogeneous Refinement x 3** — `homo_refine`, one per ab-initio, same seed. GPU.
9. **Best of three** — read each refinement's gold-standard FSC at the 0.143 threshold
   and keep the lowest. Not a CryoSPARC job; the pipeline reads the numbers and records
   the winner.
10. **Local Resolution Estimation** — `local_resolution`, on the winning refinement only.
    GPU. Non-fatal: if it fails the reconstruction still stands.

Optionally, and outside the chain: **Orientation Diagnostics** —
`orientation_diagnostics`, run over a chosen refinement by
`src/rapick/recon/scripts/run_orientation_diagnostics.py`. It is a CPU job, so it is
pinned to a worker without reserving a GPU.

Figures S6 and S7 are the FSC and viewing-direction panels CryoSPARC renders for the
refinement jobs of step 8; `collect` downloads them into each condition's `derived/`
directory. They are not drawn by any script here.

## What to watch for

The traps this chain has actually hit, and what they look like, are in
[`src/rapick/recon/README.md`](../src/rapick/recon/README.md#gotchas). The short list:
never set `compute_use_ssd: false` (2D classification dies with SIGFPE); a `completed`
job is not proof it was correct; `homo_abinit` clamps its input at a per-entry particle
cap; and a single-seed resolution is not trustworthy.
