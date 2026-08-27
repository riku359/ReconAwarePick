# 2D class selection

Sec. 3.4 of the paper, with the full procedure in Sec. S4.

CryoSPARC 2D-classifies the picks that survived the earlier stages at K = 50. This
stage scores each class average with [CryoSift](https://cryosift.org)'s pretrained
CNN — a continuous score from 1.0 (a clean particle class) to 5.0 (a non-particle
class) — and then either applies a single cutoff, or runs CryoSift's iterative
workflow, which is what the paper reports.

Upstream is `Sandbox/particle_processor` in
[sstagg/Magellon](https://github.com/sstagg/Magellon) (Schaefer et al., bioRxiv
2025.07.28.667259), used **unmodified**. What lives here is the job-directory
resolution, the output formatting, the reproducibility of the threshold decisions, and
the workaround for an upstream sign bug (below). The model, the preprocessing (resize
to 210 px plus zero-padding), the feature assembly and the label recovery through
`unconvert_labels` are all upstream's.

**Existing `class_2D` jobs are never re-run for scoring.** Upstream's
`CryosparcPredictor` reads the job directory directly —
`J<N>_<iter>_class_averages.mrc`, the same-name `.cs`, and `J<N>_<iter>_particles.cs` —
and feeds the estimated resolution, the class distribution, the pixel size and three
mass deviations (from the mean, the median and the mode) to the network alongside the
image, so it applies to a completed job as it stands.

## The iterative workflow

The single-shot cutoff creates one `select_2D`. The iterative workflow instead cancels
the **attractor effect**: a few well-aligning classes pull in the particles of rare
viewing angles, so those particles get discarded along with them.

1. **Set aside.** Score the initial classification. Among the classes scoring
   **<= 2.5**, the **best 70 %** — counted in classes, not particles — are held out of
   the loop at threshold `l`, the score of the last class inside that quantile.
2. **Loop.** On the rest, repeat "discard every class scoring **>= 4.5** permanently,
   then re-classify" for **N = 3** cycles. A discarded particle never comes back. The
   loop stops early if a re-classification's worst score falls below 4.5.
3. **Final classification.** Merge the loop survivors with the set-aside classes and
   classify once more.
4. **Final selection.** Keep the classes scoring **< 3.5**. That is the stack the
   reconstruction stage takes. Selections at 2.5 and 4.5 are made at the same time, for
   the particle-count comparison only.

N follows from the extract box size, by upstream's rule: 2 above 300 px, 3 between 200
and 300 px, 5 below 200 px (and below 200 px the classification job becomes
`class_2D_new`, "2D Classification (Small Particle)"). All four datasets extract at 256
or 300 px, so **all four get N = 3** and the plain `class_2D`.

Two details are read off CryoSift's released implementation rather than off the prose of
the CryoSift publication, because the two disagree there. Its body text gives cycle
counts that contradict themselves, while its Fig. 2 caption and its code agree with each
other; and its body text describes the holdout as 30 % of the *particles* scoring better
than 2.5, while its code counts *classes* and holds out the best 70 % of them. Particles
per class are heavily skewed, so those two are not the same set. We follow the code in
both cases. Our own paper states the rule as implemented here, in Sec. S4.

## Environment

`envs/cryosift`, Python 3.12, **torch 2.6.0** with **cryosparc-tools 4.7.0** (matching a
CryoSPARC v4.7 server). Inference is **CPU-only** — upstream hardcodes `device='cpu'` —
and takes about **50 s for 50 classes**, so this stage never competes for a GPU. Only
the re-classification jobs inside the loop use one, and those run on the CryoSPARC
worker.

`envs/cryosift/requirements_exact.txt` is upstream's own frozen list, vendored
unchanged. That is why CUDA wheels appear in it: they are installed and never used.
**Expect the build to download about 6 GB and take ten minutes or more**, almost all
of it that unused CUDA stack. It is not stuck.

One deviation from that list, made by `scripts/build_env.sh` rather than by editing
the committed copy, so it stays byte-identical to upstream:

- `pysqlite3` is skipped. It has no wheel, so pip compiles it, and the compile needs
  `sqlite3.h`, which most machines do not have. Nothing imports it: not upstream's
  `particle_processor`, not this stage, and Python has carried `sqlite3` in its
  standard library for far longer than this environment pins.

### Setup

The **pretrained weights ship inside the Magellon checkout**, so there is no separate
weights download — the sparse clone is the whole of it. `scripts/setup.sh` fetches
the repository's upstream checkouts; by hand it is:

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/sstagg/Magellon.git "$RAPICK_THIRD_PARTY/magellon"
git -C "$RAPICK_THIRD_PARTY/magellon" sparse-checkout set Sandbox/particle_processor
git -C "$RAPICK_THIRD_PARTY/magellon" checkout 0d5c40a5c89efe9d7d977e832dcc94658a291ee1
```

The pin is `0d5c40a` on `main`. Even sparse and shallow the clone is 164 MB, 33 MB of
which is the two sets of weights (`final_model.pth` and `final_model_cont.pth`;
inference uses the latter).

Then build the environment:

```bash
src/rapick/select2d/scripts/build_env.sh
```

Set `RAPICK_ENVS` to a local SSD first if the code disk is small or on NFS — uv's file
locks hang on NFS, and the script also moves uv's cache off NFS by itself.

### Paths

Everything comes from the environment variables in
[docs/CONFIGURATION.md](../../../docs/CONFIGURATION.md); a variable that cannot be
resolved raises an error naming it.

| What | Where |
| --- | --- |
| Upstream checkout | `$RAPICK_THIRD_PARTY/magellon/Sandbox/particle_processor/` |
| Scores, cycle state, figures | `$RAPICK_WORK/select2d/<project>_<job>[_iter]/` |
| CryoSPARC credentials, project uid, worker | the repository-root `.env`, overridable with `--env` |
| Default GPU for the re-classifications | `$RAPICK_GPU`, overridable with `--gpu` |

This stage has to run on the CryoSPARC machine, or on one that sees the same shared
filesystem, because upstream opens the job directory's `.mrc` and `.cs` as files. The
CryoSPARC project directory itself is asked of the server, never hardcoded.

## Files

| File | What it does |
| --- | --- |
| `cryosift_env.py` | Shared: upstream import, `.env` reading, CryoSPARC connection, the sign-preserving `model.star` parser |
| `cryosift_jobs.py` | Shared: creating `select_2D` / `class_2D`, selecting classes, waiting on a queued job |
| `score_class2d.py` | Score an existing `class_2D` into `scores.csv`, with an optional montage |
| `purify_class2d.py` | Single-shot: read `scores.csv` and create one `select_2D` at a cutoff |
| `iterate_class2d.py` | The paper's iterative workflow: set aside, loop, final classify, final select |
| `scripts/build_env.sh` | Build `envs/cryosift` from the committed lockfile |

`scores.csv` carries `class_idx, cryosift_score, keep, n_particles, class_frac,
est_res_A, psize_A`. `class_idx` is 0-based and agrees both with what CryoSPARC's
`get_class_info` returns and with the row order of `*_model.star` (its 1-based running
number, minus one).

## Running it

Every command below runs from the repository root.

Score an existing `class_2D`, and look at the result before creating anything:

```bash
PYTHONPATH=src envs/cryosift/.venv/bin/python \
    -m rapick.select2d.score_class2d --job J15 --cutoff 3.5 --montage
```

The iterative workflow on that same starter job — `--dry-run` scores and reports the
three-way split without creating a single job:

```bash
PYTHONPATH=src envs/cryosift/.venv/bin/python \
    -m rapick.select2d.iterate_class2d --class2d J15 --dry-run

PYTHONPATH=src envs/cryosift/.venv/bin/python \
    -m rapick.select2d.iterate_class2d --class2d J15 --gpu 0
```

`--project` defaults to `CRYOSPARC_PROJECT` and `--worker` to `CRYOSPARC_WORKER`, both
from `.env`; pass them explicitly to override. The thresholds are all flags
(`--attract-threshold`, `--keep-fraction`, `--reject-threshold`, `--cutoffs`), defaulting
to the values above.

The run writes `$RAPICK_WORK/select2d/<project>_<class2d>_iter/`:

| File | Contents |
| --- | --- |
| `state.json` | Every job uid created, the thresholds, and the per-cycle particle counts |
| `round<N>/scores.csv`, `round<N>/*_montage.png` | That cycle's scores, and its class averages in ascending score order |
| `final/scores.csv` | The final classification's scores |
| `convergence.png` | Particle counts across the cycles, as stacked bars |

### The final `select_2D` uid

The job the reconstruction stage consumes is the cutoff-3.5 selection. The run prints
its uid when it finishes, and `state.json` holds it:

```bash
jq -r '.final_selects["3.5"].uid' \
    "$RAPICK_WORK/select2d/<project>_<class2d>_iter/state.json"
```

### Resuming

Every job is recorded in `state.json` **before** it is queued, so an interrupted run
never rebuilds one. This matters because each cycle's re-classification runs for hours
— measured starter jobs took from 1 h 25 min at 235 k particles to 6 h 49 min at 584 k
— and a run stacks three of them in series. Re-running the same command picks up where
it stopped.

If a recorded job is in any state other than `completed`, the run stops and says so:
clean that job up in CryoSPARC, delete its entry from `state.json`, and re-run.

## Two deviations from upstream

**Both selections hang off the initial `class_2D`, rather than being chained.**
Upstream feeds the subset emitted by the attractor-holdout `select_2D` into the next
`select_2D`; here two `select_2D` jobs hang off the initial classification and make the
same split.

```
upstream : initial classify -> select(score > l) -> select(score < 4.5) -> re-classify
here     : initial classify -> select(score <= l)                       ... set aside
           initial classify -> select(l < score < 4.5)                  ... loop pool
```

The composition is identical: what upstream's two stages leave is `l < score < 4.5`,
which is this pool, and its `particles_excluded` is `score <= l`, which is the set-aside
group. The reason to split them is that it is unverified whether `get_class_info`
returns the original `class_idx` or renumbers from 0 when the input is a subset.
Upstream looks its score dictionary up by the original numbering, so an implementation
that renumbers would break the first selection silently. Hanging every `select_2D`
directly off the full set of `class_2D` classes removes the question.

**K is carried through from the starter job.** Neither the paper nor upstream specifies
the number of classes; upstream simply reuses the starter job's `params_spec` for every
cycle, and so does this. The starter jobs here record
`{class2D_num_full_iter: 20, random_seed: 0}` with no `class2D_K`, because 50 is
CryoSPARC's own default — so K stays 50 through every cycle.

One smaller departure, in the set-aside step: upstream narrows the candidates to
`0 <= s <= 2.5`, and this drops the lower bound. A negative score is the *best* class,
and losing it from the candidate list shifts the quantile. See the sign bug below —
this repository reads those scores correctly, so there is no reason to keep the bound.
It only matters on a dataset that actually produces negative scores.

## Upstream bugs worth knowing

**Upstream's star parser silently discards the best class.**
`extract_class_scores.extract_scores_from_star` picks the score up with the regex
`\d+@.*\s(\d+\.\d+)`, which cannot match a negative number. Scores do go negative:
`unconvert_labels` returns `5 - 5*pred + weight`, so any model output above 1 gives a
negative score. That line then fails to match and the class is recorded as **5.0, the
worst possible score** — so the class that should have ranked first is the one thrown
away. `cryosift_env.parse_model_star` parses the last token on the line as a float
instead, keeping the sign, and `score_class2d.py` prints a `WARN` naming every class
where its result and upstream's diverge.

**Upstream's weights path is relative to the working directory.**
`extract_class_scores.get_class_labels` holds
`MODEL_PATH = "class_labeling/final_model/final_model_cont.pth"`, so it only resolves
when the CWD happens to be the upstream checkout. `cryosift_env.import_upstream` calls
`cryosparcpredict` directly instead, which takes the weights path as an argument.

**`compute_use_ssd: false` kills `class_2D` with SIGFPE.** `iterate_class2d.py` refuses
to start when the parent job carries that setting, rather than queueing three
re-classifications that will die.

**Upstream's `main.py` is not used.** It is an interactive full-auto pipeline that
takes everything from workspace selection to the particle source through `input()`, and
runs on to ab-initio. `iterate_class2d.py` is its non-interactive equivalent for the
case where the starter job already exists; the reconstruction is a separate stage.

## What this stage deliberately does not copy from the paper

**No 100 px Fourier crop, and no re-extraction after selection.** The crop is the
paper's standard preprocessing rather than a part of the algorithm, and the
re-extraction exists to undo it. Classification here runs on un-binned boxes from the
start, so dropping both is consistent. Adding the crop would change the classification
box relative to the baseline and manual-selection conditions, and the effect of the
iteration and the effect of the box size would then be confounded.

**Ab-initio keeps CryoSPARC's defaults**, rather than the paper's Detailed settings
(max res 3 A, initial minibatch 300, final minibatch 1000), because every other
condition in the comparison runs on the defaults.

**No mass estimation and no KMeans heterogeneous branch.** Every dataset here is a
single particle species, so only the homogeneous path is implemented.
