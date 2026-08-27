# Reproducing the paper

[PAPER_TO_CODE.md](PAPER_TO_CODE.md) says what produces each table and figure and
defines the condition names. This page gives the commands.

Everything below assumes `scripts/00_setup.sh` has run, `.env` holds your
CryoSPARC credentials, and `RAPICK_DATA` and `RAPICK_WORK` are set
([CONFIGURATION.md](CONFIGURATION.md)).

## Two ways in

**From the published artifacts.** Reproducing one condition's reconstruction takes
a few hours instead of a few days, because the picks and the masks are downloaded
rather than derived.

```bash
bash scripts/01_download_data.sh --entry 10081 --intermediates
bash scripts/05_select2d.sh     --entry 10081 --condition both
bash scripts/07_reconstruct.sh  --entry 10081 --condition both
```

**From scratch.** Steps 02 to 07 in order, per entry. Budget weeks for all four
entries and all conditions; the per-stage timings are in the README.

## Tables

### Table 1, the datasets

Nothing to run. The values are `configs/datasets/empiar_<id>.yaml`, and
`scripts/01_download_data.sh` prints the micrograph counts it actually placed.

### Table 2, main results, and Table 4, the ablation

The same runs produce both: Table 4 is the five conditions on the CryoTransformer
candidates, and Table 2 puts its `baseline` and `fb` rows next to the three other
pickers. Every condition is reconstructed on the full deposition, three seeds,
best of three by GSFSC 0.143.

```bash
for c in baseline mask select both fb; do
  bash scripts/07_reconstruct.sh --entry 10081 --condition "$c"
done
```

`select`, `both` and `fb` take their particles from a 2D class selection rather
than from a STAR of their own, so their chain starts one step lower, at an
existing Select 2D Classes job. Build that selection first with
`scripts/05_select2d.sh`; `scripts/07_reconstruct.sh` then finds the job in the
state.json it wrote and hands it to the driver that can start there. `fb` also
needs the round-1 checkpoint, from `scripts/06_loop.sh`.

Each run writes `$RAPICK_WORK/empiar_<id>/full/<condition>/metrics.json`.
Collection is not optional and not automatic for the from-selection conditions:
`scripts/07_reconstruct.sh` runs it, but a hand-driven
`rapick-recon reconstruct-from-selection` leaves a manifest and no `metrics.json`
until you call `rapick-recon collect` yourself.

For the three comparison pickers, see [BASELINES.md](BASELINES.md): their picks
are published, so `--condition cryolo|topaz|cryosegnet` reconstructs without
installing them.

### Table 3, the mistaken rate of the mask

The difference between the `baseline` and `mask` conditions on the annotated
micrographs: removals are the drop in matched candidates, the particle row is the
drop in true positives.

```bash
python3 results/analysis/ablation_2d_metrics.py --entries 10081 10093 10345 10532
```

The two GT-overlap scripts in `src/rapick/cleaner/` measure a related but
different quantity, annotated particles whose centre falls inside the mask. They
do not produce this table.

### Table 5, particles retained

Read from the runs of Table 4: the `particle_counts` block of each condition's
`metrics.json`. Note that `homo_abinit` clamps its input at a per-entry particle
cap, so the refine count can be below the extract count; `collect` records both.

### Table 6, per-round loop diagnostics

```bash
bash scripts/06_loop.sh --entry 10081 --rounds 3
python3 -m rapick.loop.round_metrics --id 10081 --csv $RAPICK_WORK/loop/10081/rounds.csv
```

Scored against the CryoPPP annotations with the same code as every other 2D
number, `src/rapick/eval/calc_common_2d_metrics.py`.

### Table 7, pseudo-labels against a perfect teacher

The `fb` row is the run above. The second row replaces the teacher with the
CryoPPP annotations of the same 50 micrographs, holding everything else fixed:

```bash
bash scripts/06_loop.sh --entry 10081 --rounds 0 --teacher gt
bash scripts/03_pick.sh --entry 10081 --checkpoint $RAPICK_WORK/loop/10081_fb_gt/round0/model.pth
bash scripts/04_mask.sh --entry 10081 --out-name fb_gt
bash scripts/05_select2d.sh --entry 10081 --condition fb_gt
bash scripts/07_reconstruct.sh --entry 10081 --condition fb_gt
```

`--teacher gt` writes into its own arm, so it never overwrites the `fb` run it is
read against. **This path has not been run end to end in this form.** The scripts
that produced the published row lived on a lab server and were never committed;
`src/rapick/loop/make_gt_teacher.py` reimplements the procedure they are documented
to have followed, which is to restrict the entry's ground-truth STAR to the
micrographs of that round's training set and hand it to the fine-tune in place of
the surviving picks.

### Table 8, the same selection on CryoSegNet

```bash
bash scripts/07_reconstruct.sh --entry 10081 --condition cryosegnet
bash scripts/05_select2d.sh    --entry 10081 --condition cryosegnet_both
bash scripts/07_reconstruct.sh --entry 10081 --condition cryosegnet_both
```

### Table S1, hyperparameters

Nothing to run. The values are the defaults in `src/rapick/picker/`; its README
lists them next to the table.

### Table S2, 2D detection scores

```bash
python3 src/rapick/eval/calc_common_2d_metrics.py --batch --markdown
```

crYOLO on EMPIAR-10081 and Topaz everywhere are greyed in the paper and excluded
from its ranking, because their released models may have seen those entries in
training ([BASELINES.md](BASELINES.md)).

## Figures

`results/figures/` has one directory per figure, each with its own README and the
exact command. `bash scripts/08_tables_figures.sh` runs the ones that need
nothing but this repository.

| | Runs standalone? |
| --- | --- |
| Fig. 4, Fig. 5, and the loop-rounds figure | yes, from `results/tables/` |
| Fig. S2, Fig. S3 | yes, from committed inputs |
| Fig. 1, Fig. 2 | needs LibreOffice for the pptx to pdf step |
| Fig. 3 | needs ChimeraX and the reconstructed volumes |
| Fig. 6, Fig. S5 | needs a run's outputs |
| Fig. S1, Fig. S6, Fig. S7 | needs a live CryoSPARC instance: the panels are what CryoSPARC renders for its own jobs |

Fig. S1 and Fig. S4 are TikZ inside the manuscript, not drawn by any script here.

## Checking without running anything

Every number the paper prints is in `results/tables/`, as JSON, with the
unrounded per-seed values and the CryoSPARC job uid behind each one:

```bash
python3 -c "import json;d=json.load(open('results/tables/main_results.json'));print(json.dumps(d['values']['cryolo']['10081'],indent=2))"
```

## Three things that bound what the numbers mean

**EMPIAR-10345's resolutions are about half the physical figure.** The pixel size
follows CryoPPP's declared value, which is the super-resolution movie figure,
while the micrographs are 2×-binned. Conditions compare within that entry only.

**Resolution is best-of-three-seeds, and one seed is not enough.** The
seed-to-seed spread ranges from 0.011 to 2.073 Å across this project's runs, so on
some entries it exceeds the effect being measured. If a `homo_abinit` seed dies,
advance the seed number rather than reporting a best-of-two as a best-of-three.

**The 2D scores are not held out.** In each round, 50 of the 300 annotated
micrographs also train the picker.
