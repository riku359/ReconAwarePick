# Analysis

One-off scripts that produced numbers the paper quotes but that no table or figure
is built from directly. They were written against a single machine and held
absolute paths to it; here every path comes from the repository's contract
([CONFIGURATION.md](../../docs/CONFIGURATION.md)), plus `RAPICK_CRYOSPARC_PROJECT_DIR`
for the CryoSPARC project directory, because several of them read a job's `.cs`
and `job.json` off disk rather than through the API.

Each writes its result as JSON, so a number can be checked without re-running the
run it came from. Their outputs for the paper's own runs are committed under
[`../tables/revision/`](../tables/revision/).

| Script | What it computes | Backs |
| --- | --- | --- |
| `ablation_2d_metrics.py` | 2D precision, recall and F1 of the five ablation conditions on the annotated micrographs, scored with the same code as every other 2D number | **Table 3** (as the difference between the `baseline` and `mask` rows) and the CryoTransformer row of Table S2 |
| `contam_survival.py` | the fraction of the particles surviving 2D class selection that land on contamination, i.e. what the mask would still have removed | the claim that 2D selection removes most of what the mask removes |
| `mask_coverage.py` | how often the contamination mask inverts, counted over every micrograph of the four full sets rather than the three examples the paper draws | the mask failure discussion (Sec. 6.2) |
| `locres_common_mask.py` | local resolution re-measured inside one mask common to every condition of an entry, since each refinement generates its own and they differ in size | the local-resolution reading |
| `teacher_purity.py` | the precision of a round's teacher labels against the CryoPPP annotations, over the micrographs the teacher covers | the loop diagnostics |
| `finetune_rate.py` | true fine-tuning time per epoch, from the checkpoints' modification times | Sec. S9 |
| `collect_cost.py` | the compute cost of each stage, from logs that already exist | Sec. S9 |
| `analysis_env.py` | shared path resolution; not a script | |

## Running one

```bash
export RAPICK_CRYOSPARC_PROJECT_DIR=/path/to/CryoSPARC/PXX
envs/figures/.venv/bin/python results/analysis/ablation_2d_metrics.py \
    --ids 10081 10093 10345 10532 --setting annot \
    --out $RAPICK_WORK/analysis/ablation_2d_metrics.json
```

Every one takes `--ids` and most take `--out`; `--help` lists the rest. The ones
that read a job directory also take `--project-dir`, which defaults to
`RAPICK_CRYOSPARC_PROJECT_DIR`.

## Two things to know before reading the numbers

**Table 3 is a difference, not a direct measurement.** Its removals row is the
drop in matched candidates between the `baseline` and `mask` conditions, and its
particle row is the drop in true positives. The two GT-overlap scripts in
`src/rapick/cleaner/` measure a related but different quantity, annotated
particles whose centre falls inside the mask, and do not reproduce the table.

**The 2D scorer covers the micrographs that carry annotations, not always 300.**
CryoPPP deposits 300 per entry and the paper prints 300, but on EMPIAR-10093 and
10345 only 295 carry annotations, and those are what the scores are averaged over.
`../tables/datasets.json` records the counts per entry.
