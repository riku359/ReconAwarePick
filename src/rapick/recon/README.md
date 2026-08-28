# `rapick.recon` — the CryoSPARC v4.7 job chain

Section S1 of the paper. A config-driven driver that takes one arm's picks (a
GT-aligned STAR) plus a micrograph set and runs CryoSPARC job by job, recording what it
did. It is what turns a set of coordinates into the resolution numbers in Table 2, the
Table 4 ablation, the particle counts of Table 5, Table 8, and the maps behind Fig. 3,
Fig. S6 and Fig. S7.

Nothing about *what* runs is in the Python. Job types, port names and default params are
in [`configs/cryosparc_v47.yaml`](../../../configs/cryosparc_v47.yaml); the micrographs,
optics and picks are in [`configs/datasets/`](../../../configs/datasets/); the class_2D
and reconstruction parameters are in [`configs/recon.yaml`](../../../configs/recon.yaml),
one file shared by every arm; credentials and the project uid are in the
repository-root `.env`.

**CryoSPARC v4.7.x is required and this repository does not install it.** See
[docs/CRYOSPARC.md](../../../docs/CRYOSPARC.md).

## The job chain

One chain per arm. Import Micrographs and Patch CTF are created once per
(entry, scale) and reused by every arm of that entry, so all arms are
compared over identical CTF estimates.

1. **`import_micrographs`** — the micrograph glob plus the entry's optics
   (`psize_A`, `accel_kv`, `cs_mm`, dose). *Shared.* CPU.
2. **`patch_ctf_estimation_multi`** — per-micrograph CTF. *Shared.* GPU.
3. **`import_particles`** — the arm's STAR, coordinates only (`ignore_blob`),
   connected to the imported micrographs. Y is flipped to `ny - Y` first (see
   `coords.py`). CPU.
4. **`extract_micrographs_multi`** — cut `box_size_pix` boxes out of the CTF'd
   micrographs. GPU.
5. **`class_2D`** — K = 50, 20 full iterations, one seed. GPU.
   *Arms with 2D class selection stop here and hand off to
   [`src/rapick/select2d/`](../select2d/), then reconstruct from its final Select 2D
   Classes job.*
6. **`homo_abinit` x 3** — ab-initio, forked over seeds 0, 1, 2. GPU.
7. **`homo_refine` x 3** — homogeneous refinement, one per ab-initio. GPU.
8. **best-of-3** — read each refinement's GSFSC 0.143 and keep the lowest.
9. **`local_resolution`** — on the winner only. GPU. Non-fatal: if it fails the
   reconstruction still stands.
10. **`collect`** — a separate command; derives `metrics.json` from the manifest.

`orientation_diagnostics` is optional and is not part of the chain. Run it over a chosen
refinement with [`scripts/run_orientation_diagnostics.py`](scripts/run_orientation_diagnostics.py).

## Installing and running

The environment supplies dependencies only; it installs no code. Run the package from
`src/` with `PYTHONPATH`, the same way every other stage of this repository is run.

```bash
cd envs/recon && UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --locked && cd ../..
alias rapick-recon='PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.recon.cli'
```

`envs/recon/pyproject.toml` keeps the project name it was locked under and declares no
packages of its own.

## CLI

Four subcommands. All take `--dataset` and `--setting`, and `--condition` defaults to
`configs/recon.yaml`; the CryoSPARC project comes from `CRYOSPARC_PROJECT` in `.env` and
the worker lane from `CRYOSPARC_WORKER`, so neither appears on the command line or in a
config file. `scripts/` drives all four; these are what it runs.

| command | what it does |
| --- | --- |
| `check-setup` | preflight, read-only: connection, project access, micrograph health, and that no two arms share a STAR file. Never creates a job. |
| `prepare` | get or create the workspace for this (entry, scale). Its title is `<empiar id>_<setting>`. |
| `run` | run one arm's whole chain, from `import_particles` through `local_resolution`. Resumable. |
| `collect` | rebuild `metrics.json` and the derived CSVs from jobs that already finished. Re-runs nothing. |

```bash
STAR=$RAPICK_WORK/picks/10081/cryotransformer_mask.star

rapick-recon check-setup \
  --dataset configs/datasets/empiar_10081.yaml --setting full \
  --source  cryotransformer_mask --star "$STAR"

rapick-recon run \
  --dataset configs/datasets/empiar_10081.yaml --setting full \
  --source  cryotransformer_mask --star "$STAR" --seeds 0,1,2

rapick-recon collect \
  --dataset configs/datasets/empiar_10081.yaml --setting full \
  --source  cryotransformer_mask
```

`--setting` is `annot` (the 300 CryoPPP-annotated micrographs) or `full` (the whole
deposition). Every reconstruction-level result of the paper uses `full`; `annot` exists
for the `gt` arm, whose annotations cover only those 300.

Other useful flags on `run`:

| flag | |
| --- | --- |
| `--seeds 0,1,2` | which seeds to fork ab-initio and refinement over. **Pass it every time** — the default is a single seed, deliberately, so that a one-seed run is never accidental. |
| `--source NAME` | the name this run is recorded under, and the key the dataset config declares its STAR by. |
| `--star PATH` | that STAR, when the dataset config does not name it. Declared for the length of the run; nothing is written back to the config. |
| `--no-local-res` | skip the local-resolution estimate on the best-of-3 winner. |
| `--gpus 0` | pin GPU jobs to these cards. Without it, each GPU job picks a physically-free card at queue time. |
| `--extract-gpus N` | fan the I/O-bound extract step over up to N free cards. |
| `--micrographs` | override the micrograph glob for a fast smoke run. |
| `--force` | proceed past a failed preflight. Read the failure first. |
| `--project` / `--worker` | override `CRYOSPARC_PROJECT` / `CRYOSPARC_WORKER` for one run. |

## Where the output lands

Under `$RAPICK_WORK`:

```
$RAPICK_WORK/empiar_<id>/<setting>/
├── _shared/manifest.json          import_micrographs + patch_ctf, reused by every arm
└── <name>/
    ├── manifest.json              which CryoSPARC jobs ran, per seed, with input hashes
    ├── metrics.json               resolution, particle counts, best seed, map references
    ├── normalized.star            the Y-flipped STAR that was imported
    └── derived/                   ctf.csv and the plots CryoSPARC rendered for its jobs
```

`manifest.json` is an experiment record, not a cache. It holds the project and workspace
uids, every step's job uid / job type / output ports / status, one entry per seed
(ab-initio uid, refine uid, its GSFSC 0.143), the winning seed, the local-resolution job,
and the input STAR's path, Y-flip and sha256. A rerun reuses a recorded step only when
that job is still `completed` **and** its recorded input fingerprint still matches, so
replacing a STAR silently invalidates the reuse. It never scans the workspace for "the
latest completed job", because that mis-wires conditions and seeds into each other.

`metrics.json` holds the particle count after each stage (including how many ab-initio
actually used), the per-seed resolutions, the best seed, project-relative paths to the
refined and local-resolution maps, and a median summary of the micrographs' CTF
covariates. Large files — particle stacks, half maps, volumes, logs — are never copied;
the CryoSPARC job directory stays the source of truth.

## Gotchas

None of these announces itself.

**Never set `compute_use_ssd: false`.** CryoSPARC's SSD particle cache stays on. Turning
it off to save disk makes `class_2D` die with SIGFPE ("Floating point exception"): the
HDD-direct read path drives one 2D class to zero members, and the per-class average then
divides by zero. The same particles classify without incident with the cache on. Manage
SSD space at the worker instead — a quota plus `cache_reserve` protect the physical disk,
and the old cache is LRU-evicted.

**A `completed` entry in the manifest is not proof the job was correct.** The pipeline
reuses recorded jobs, so a step that ran over an incomplete micrograph set stays "done"
forever. This is not hypothetical: a Patch CTF job here reported `completed` while
silently dropping micrographs, and everything downstream saw only the shrunken
`exposures` output. `patch_ctf` and `extract` therefore both fail hard on any
`*_incomplete` output (one extract job lost 243 of its 300 micrographs this way and still
said `completed`). Before spending GPU hours on top of a reused step, verify the
micrograph count and the STAR hash — `scripts/verify_patch_ctf.py` does the CTF half —
and never auto-continue past a preflight failure.

**`homo_abinit` clamps its input at a per-entry particle cap.** Asked for a single class,
it reconstructs a randomly chosen subset large enough to converge and leaves the rest in
`particles_unused`, so the refine particle count can be lower than the extract count. The
cap is a per-entry constant (measured 219,900 on 10081 and 176,700 on 10532) and does not
grow with the input. Two conditions are comparable on particle count only when neither is
clamped; a comparison where one arm straddles the cap and the other does not is
confounded. `collect` records the used/unused split so the clamp is visible in
`metrics.json`.

**Run three seeds.** The paper's protocol (Sec. 4.2) runs the reconstruction three
times with different random seeds and reports the best of the three by GSFSC 0.143, so a
single-seed run reproduces something the paper does not report. `--seeds` must be passed
explicitly on every `run` and `collect`; the `reconstruction.seeds` block in the recon
YAML is documentation, not enforced config. When an ab-initio dies with a SIGSEGV, retry
the same seed at most twice and then advance the seed number (`--seeds 0,1,2` ->
`--seeds 0,1,3`); completed jobs are reused from the manifest, so a retry resumes the trio
rather than restarting it. Say which seeds were actually used — do not report a best-of-2
as a best-of-3. In one measured night 7 of 19 runs died, and one entry's seed 2 died four
times in a row while seeds 0, 1 and 3 completed.

**`import_particles` dies on the first micrograph its STAR names but cannot find**, and a
`*.mrc` glob happily imports a partial download. Confirm the expected micrograph count
(each dataset config declares it) before running.

**One job at a time per worker.** Two concurrent jobs on a box with shared storage evict
each other's page cache, and a `class_2D` iteration can then inflate several-fold at 0%
GPU utilisation.

## Layout

```
cli.py            check-setup / prepare / run / collect
config.py         profile + recon + dataset + .env -> ResolvedConfig, with ${VAR} expansion
api.py            the only module that talks to cryosparc-tools
pipeline.py       the chain, best-of-N, resume
manifest.py       the experiment record, and safe reuse
artifacts.py      collect: job outputs -> metrics.json and derived CSVs
coords.py         Y flip and micrograph-header reading, before import
gpu_select.py     pick a physically-free GPU, so a shared card does not OOM the run
setup_check.py    the preflight body
jobs/             one thin module per CryoSPARC job: type, inputs, params, outputs
scripts/          standalone utilities, see below
```

| script | |
| --- | --- |
| `scripts/smoke_probe.py` | re-verify `configs/cryosparc_v47.yaml` against a live server. Build-only, queues nothing. Needs a **throwaway** project. Run it on a new site and after any CryoSPARC upgrade. |
| `scripts/verify_patch_ctf.py` | inspect a finished Patch CTF before anything consumes it: output count against the expected count, missing or non-converged fits, the defocus distribution, astigmatism. |
| `scripts/run_orientation_diagnostics.py` | run CryoSPARC's Orientation Diagnostics over an existing refinement (cFAR, SCF*, the viewing-direction sphere). CPU job. |
| `scripts/recut_picks.py` | re-cut a picks STAR at a different operating point: `--score-min` (absolute score cut) or `--subsample-n` (the count-matched control). |

Two escape hatches exist as environment variables, both off by default and both for the
case where a small, independently verified micrograph loss is acceptable:
`RAPICK_RECON_MAX_INCOMPLETE_CTF_MICS` and `RAPICK_RECON_MAX_INCOMPLETE_MICS`. Leave them
unset unless you have checked what was lost.

`jobs/` holds one module per job the pipeline actually creates. Contamination removal is
not among them: in this paper it is the MicrographCleaner mask of
[`src/rapick/cleaner/`](../cleaner/), applied to the picks before they are imported, so
no CryoSPARC job does it.
