# `rapick.loop` — the reconstruction-aware feedback loop

Sec. 3.5 of the paper. The picker is retrained on the particles that survive its own
downstream cleanup: contamination removal and 2D class selection decide which of its
picks were real, and those become the labels for the next round.

## Fine-tuning is all weights, not LoRA

The paper fine-tunes **every weight except resnet layer1, which stays frozen** —
`finetune.py --finetune_mode head_decoder_encoder_resnet`. It is not a low-rank adapter,
and no result in the paper was produced with one.

That is the `fb` arm here, it is the default, and it is the only arm the paper reports.
The study this port comes from also carried two LoRA arms, and its driver defaulted to
one of them, so following its documentation reproduced something other than the paper.
**The LoRA arms are not ported.** They are gone rather than deprecated: no default, no
flag, no adapter-merging step. Every arm here runs the same full fine-tune and differs in
exactly one other thing: `fb_nomask` skips the contamination stage (an analysis cut from
the paper, and it says so on startup), and `fb_gt` replaces the teacher labels with the
CryoPPP annotations of the same micrographs for the perfect-teacher upper bound of
Table 7 (`--teacher gt`).

## One round

Round *n* picks with theta_n and produces theta_{n+1}:

1. **Pick** the 300 annotated micrographs with theta_n, at the fixed operating point
   (600 queries, top 75% by score, NMS 0.7). Round 0 picks with theta_0 itself.
2. **Score** the raw picks against the CryoPPP annotation. On round 0 this is a hard
   gate: theta_0 has to reproduce its recorded row (Sec. S2) or the run stops, because
   every number below would inherit the discrepancy.
3. **Discard** the picks whose centre lands on contamination, using the stored masks.
4. **Classify**: import_particles -> extract -> class_2D (K = 50) on what is left.
5. **Select** the good classes with the iterative 2D selection (cutoff 3.5). Its
   survivors, S_n, are the round's pseudo-labels.
6. **Sample** 50 micrographs and export the survivors on them as `teacher.star`.
7. **Fine-tune** theta_0 on those labels, all weights with resnet layer1 frozen:

   ```
   theta_{n+1} = FineTune(theta_0; S_n)                                    (Eq. 1)
   ```

8. **Promote** the result to `models/model_{n+1}.pth`, which round *n+1* picks with.

Every round restarts from theta_0. It never fine-tunes the checkpoint that just picked.
This follows TranSPHIRE, whose own implementation assigns its initial weights once at
session start and never updates them while the picking weights advance. Chaining instead
would let the picker's bias compound: it would be trained on the particles it chose,
having chosen them because it was trained on them.

## The teacher set

- **50 micrographs**, drawn from the micrographs that carry surviving particles. A
  micrograph whose every pick was rejected is not a zero-particle training example.
- **A fixed per-round seed** (round *n* uses seed *n+1*), so each round draws a different
  50 and every rerun of a round draws the same 50. The chosen names are written to
  `train_mics.txt` as a reproduction input.
- **Split 40 / 10** into training and validation, inside `finetune.py`
  (`--val_fraction 0.2`). The 10 validation micrographs **monitor the loss only** — they
  choose no threshold, gate no round, and select no model.

Sampling happens here rather than inside the fine-tuner because the fine-tuner subsamples
micrographs *after* its train/validation split, so 50 sampled there would not become
40 + 10.

## The loop runs no reconstruction

At 300 micrographs a reconstruction does not resolve one round from the next: the
seed-to-seed spread of GSFSC 0.143 is the size of the effect being looked for, so a
per-round trio of ab-initio + refinement would cost an hour a round and answer nothing.

Rounds are followed instead by numbers that are deterministic given the picks and need no
ground truth: pick count, what the contamination filter removed, what class_2D accepted,
what the selection permanently rejected, what survived. `round_metrics.py` prints them,
alongside P/R/F1 against the annotation.

3D happens once, at full-deposition scale, on the checkpoint the loop produced:
`repick_fullset.py`, and then `rapick.recon`.

## Three rounds; the paper reports round 1

`--rounds 0-2` is the default. Round 0 picks with theta_0 and trains model_1; round 1
picks with model_1 and trains model_2; round 2 picks with model_2 and trains nothing.
**The paper reports round 1, i.e. `models/model_1.pth`** — the checkpoint round 1 picks
with, which is the output of round 0's fine-tune.

## Commands

Run the loop for one entry:

```bash
PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.loop.run_loop \
    --id 10081 --rounds 0-2
```

Export the per-round metrics of Table 6:

```bash
PYTHONPATH=src python3 -m rapick.loop.round_metrics --csv results/tables/table6.csv
```

Re-pick the full deposition with the round-1 checkpoint, and publish it as the `fb`
condition's picks:

```bash
PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.loop.repick_fullset \
    --id 10081 --model "$RAPICK_WORK/loop/10081/models/model_1.pth"
```

That writes `$RAPICK_WORK/picks/10081/fb.star` and prints the `rapick.recon` command that
reconstructs it. Everything past that point is ordinary `configs/conditions/fb.yaml`
business and knows nothing about a checkpoint.

Useful while a run is in flight, both read-only and safe against a live loop:

```bash
PYTHONPATH=src python3 -m rapick.loop.status --id 10081
PYTHONPATH=src envs/recon/.venv/bin/python -m rapick.loop.export_stage_stars --id 10081
```

Every step records itself in `state.json` and is skipped when already done, so re-running
any driver resumes rather than rebuilds. `--stop-after <step>` stops at a step, `--redo
<step,step>` re-runs recorded ones. When resuming a run in pieces, pass `--last-round`
explicitly: a partial `--rounds` would otherwise treat its last round as final and skip a
fine-tune that is not.

## Checkpoints are not published yet

The round-1 checkpoints this loop produces are **not yet released**. They are intended for
the Hugging Face model repository `rikrikrik/recon-aware-pick-weights`. Until they are
there, reproducing any round-1 number means re-running the loop from theta_0, which
`scripts/01_download_data.sh` fetches.

## What runs where

The loop is a conductor: every step shells out to a tool that lives outside this package.
Each is resolved through `paths.TOOLS`, from an environment variable, and a missing one
raises an error naming the variable rather than guessing a path.

| tool | variable | default |
| --- | --- | --- |
| picker inference | `RAPICK_TOOL_PREDICT` | `$RAPICK_THIRD_PARTY/cryotransformer/predict.py` |
| picker inference, whole deposition | `RAPICK_TOOL_PREDICT_FULLSET` | `$RAPICK_THIRD_PARTY/cryotransformer/predict_fullset.py` |
| picker fine-tuning | `RAPICK_TOOL_FINETUNE` | `$RAPICK_THIRD_PARTY/cryotransformer/finetune.py` |
| 2D scorer | `RAPICK_TOOL_SCORER` | `src/rapick/eval/calc_common_2d_metrics.py` |
| contamination filter | `RAPICK_TOOL_MASK_FILTER` | `src/rapick/cleaner/filter_star_from_masks.py` |
| 2D class selection | `RAPICK_TOOL_SELECT_2D` | `src/rapick/select2d/iterate_class2d.py` |

The first three live in the upstream picker checkout that `scripts/00_setup.sh` clones
and copies `src/rapick/picker/overlay/` over; the last three are sibling stages of this
repository. Each runs under its own environment (`envs/cryotransformer`,
`envs/micrograph_cleaner`, `envs/cryosift`); the scorer is standard-library-only and runs
under whichever interpreter runs the loop. The 2D class selection is invoked as
`-m rapick.select2d.iterate_class2d`, because its package uses relative imports; the rest
are run by path. The picking and fine-tuning steps take their micrographs by explicit path
(`--data_root`, `--images_dir`) rather than through `RAPICK_TEST_DATA`, so nothing depends
on which set that variable happens to point at.

The reconstruction is not this package's job. `rapick.recon` runs it from the picks this
package publishes, under `configs/conditions/fb.yaml`. The only config this package reads
is that same file, for its `pipeline.class2d` block (K = 50, 20 full iterations): a round
of the loop *is* the `fb` condition's 2D classification at `annot` scale, so it reads the
one file rather than a copy. `RAPICK_CONDITION_FB` overrides its location, and
`RAPICK_RECON_PROFILE` overrides `configs/cryosparc_v47.yaml`.

## Paths

Everything comes from `docs/CONFIGURATION.md`'s contract; nothing is hardcoded to a host.

```
$RAPICK_DATA/cryoppp/<id>/micrographs           the 300 annotated micrographs
$RAPICK_DATA/cryoppp/<id>/ground_truth/*.star   the annotation the score step reads
$RAPICK_DATA/cryoppp_fullset/<id>/micrographs   the full deposition
$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth      theta_0

$RAPICK_WORK/masks/<id>/                        stored contamination masks
$RAPICK_WORK/loop/<id>/round<n>/                one round: state, stars, labels, logs
$RAPICK_WORK/loop/<id>/models/model_<n>.pth     the checkpoints the loop produces
$RAPICK_WORK/loop/<id>/fullset/<condition>/     one full-deposition re-pick
$RAPICK_WORK/picks/<id>/<condition>.star        where the re-pick is published
$RAPICK_WORK/select2d/<project>_<job>_iter/     the 2D selection's cycle state
$RAPICK_WORK/empiar_<id>/<setting>/<source>/    manifests and metrics.json
```

The picker appends `<id>/images` to the data root it is given, so each entry needs an
`images` directory or symlink beside its `micrographs`, under both
`$RAPICK_DATA/cryoppp/<id>/` and `$RAPICK_DATA/cryoppp_fullset/<id>/`. The loop checks
for it before spending a GPU minute, because without it the picker reads an empty set and
every count below is silently zero.

`CRYOSPARC_WORKER` and `CRYOSPARC_PROJECT` come from the repository-root `.env`; `--worker`
and `--project` override them. The GPU comes from `RAPICK_GPU`; `--gpu` overrides it.
A non-default arm writes to `$RAPICK_WORK/loop/<id>_<arm>/` and into its own CryoSPARC
workspace, so two arms of one entry never share state.

## Files

| file | what |
| --- | --- |
| `run_loop.py` | the driver: one entry, one arm, N rounds |
| `run_to_class2d.py` | import_particles -> extract -> class_2D, stopping before any 3D |
| `export_teacher_star.py` | a select_2D job -> `teacher.star` + `train_mics.txt` |
| `repick_fullset.py` | one checkpoint -> full-deposition picks -> `picks/<id>/<condition>.star` |
| `round_metrics.py` | Table 6: P/R/F1 and the pick-count funnel, per round |
| `status.py` | how far each arm has got, and what step is next |
| `export_stage_stars.py` | per-stage STARs, to attribute each discarded pick to a stage |
| `filter_star_by_micrograph_list.py` | restrict a STAR to (or exclude) a micrograph list |
| `entries.py` | the per-entry constants and the arms |
| `paths.py` | the path and environment contract |
| `common.py` | logging, subprocess, the lock, resumable state |
| `star.py` | reading and writing the GT-aligned STAR |

## Two things that will cost a run if forgotten

**One driver per entry at a time.** `common.acquire_lock` takes an exclusive `flock` and
refuses a second driver on the same entry, because the damage from an overlap is silent
rather than loud: two processes sharing one GPU, one CryoSPARC project and one
`state.json` will each run the round's fine-tune, each write the round's model, and each
build the next round its own extract and class_2D with a 2D selection cycle underneath.
Nothing fails; the records simply stop describing the files on disk. Two *different*
entries are fine side by side on separate cards. `repick_fullset.py` locks per (entry,
condition) for the same reason, and additionally refuses to publish a second checkpoint's
picks under a condition that already names one.

**A crashed ab-initio job still needs three seeds.** This bites in the reconstruction that
follows the re-pick, not inside the loop, but it is the trap most likely to turn a
reported number into a wrong one. When an ab-initio job dies with a SIGSEGV, retry the
same seed at most twice and then advance the seed number (`--seeds 0,1,2` ->
`--seeds 0,1,3`). Completed jobs are reused from the manifest, so a retry resumes the trio
rather than restarting it. **Which seeds were used must be stated with the resolution** —
a best-of-2 is not a best-of-3, and reporting one as the other is not a rounding error. In
one measured night 7 of 19 runs died, and one entry's seed 2 died four times in a row
while seeds 0, 1 and 3 completed.

## Known limits

- **The operating point is relative** (top 75% of candidates), not an absolute score
  threshold, because fine-tuning moves the scale the scores live on and a fixed number
  would mean a different operating point each round. The consequence: feedback that makes
  the picker propose less junk shows up as a lower pick count, but feedback that leaves
  the junk in place and only lowers its score does not move the 75% cut at all.
- **The 300 scored micrographs include the 50 a round trained on** (about 17%). The loop
  specialises to one entry rather than generalising to unseen data, so fitting its own
  training data is the mechanism and not a leak — but every P/R/F1 from round 1 onward
  carries that overlap and is reported with it.
- **K = 50 is coarse.** One class holds roughly 2,300 particles, so junk surviving inside
  a good class sets a ceiling on how much the picker's precision can improve. Tuning K is
  not a contribution of this work; the iterative selection loosens the constraint but does
  not remove it.
- **EMPIAR-10345's selection collapses.** The iterative selection permanently rejects
  88.5% of round 0's classes there, because its reject threshold is absolute and that
  entry's score distribution sits higher than the others'. Teacher labels downstream of
  that are suspect, and a round-over-round decline on 10345 may be the selector rather
  than the feedback.
- **EMPIAR-10345's pixel size follows CryoPPP** (0.673 A), which is understated by 2x
  against EMDB, so every resolution reported for that entry is half the physical one.
