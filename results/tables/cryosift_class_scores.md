# cryosift_class_scores.csv

The input to **Fig. 5** of the paper (`fig:cryosift_scores`), which shows the CryoSift
score of each of the 50 classes of each entry's first 2D classification, with the absolute
discard threshold of 4.5 drawn on it.

The figure reads only the rows with `stack == baseline`, four entries times 50 classes,
and histograms the `cryosift_score` column. The file also carries a second stack per entry
(`cleaner_tri`, the same classification after contamination masking), which the figure does
not use, and a set of common-line geometry columns from a study that did not enter the
paper.

## What produced it

`cryosift/summarize_geom_stage0.py` in the private research repository, which joins the
outputs of `run_geom_stage0.py` (common-line geometry), `collect_class_view_truth.py`
(true viewing directions from the refined poses) and `probe_class_is_projection.py`
(matching each class average against projections of a reference map) into one table. It
was written to `output/3d_recon/geom2d/stage0_classes.csv` and is copied here unchanged.
The figure itself is drawn by `results/figures/cryosift_scores/build_cryosift_scores.py` in the manuscript
repository, which reads this file directly.

The eight sets come from the authors' earlier processing project, not from the project
that produced the paper's tables, so no number in any table JSON here comes from this
file. What the paper takes from it is Fig. 5 and the two counts in Sec. 6.2, which the
file reproduces exactly: on EMPIAR-10345 the first classification discards 44 of the 50
classes and 88.557 % of the particles, which the paper rounds to 88.6 %.

## Layout

400 rows: 4 entries x 2 stacks x 50 classes, plus one header row.

| column | meaning |
| --- | --- |
| `set` | `<empiar_id>_<stack>`, the row's group |
| `empiar_id` | `10081`, `10093`, `10345` or `10532` |
| `stack` | `baseline` (raw picks) or `cleaner_tri` (the same classification after contamination masking). Fig. 5 uses `baseline` only |
| `class_idx` | class number within the classification, 0 to 49 |
| `n_particles` | particles assigned to the class |
| `cryosift_score` | CryoSift's appearance score for the class average, three decimals. Higher is worse; the threshold drawn in Fig. 5 is the absolute value 4.5 |
| `geom_corruption_median` | median common-line corruption of this class against the others, four decimals |
| `n_good_lines` | how many common lines passed the band and quality filters for this class |
| `view_group` | index of the group of classes that the common-line geometry places at the same viewing direction |
| `true_view_spread_deg` | angular spread in degrees of the true viewing directions of the class's particles, taken from the refined poses. Empty where the spread is undefined |
| `is_single_view` | 1 if `true_view_spread_deg` is below 15 degrees, else 0. Empty wherever the spread is |
| `projection_match` | best match score of the class average against projections of a reference map. **Empty on every `cleaner_tri` row**; the probe was run for the `baseline` stack only |
| `projection_peakedness` | how sharply that match peaks over the projection search. Empty on every `cleaner_tri` row |
| `keep_cryosift_absolute` | 1 if `cryosift_score` is below 4.5, else 0. This is the rule Fig. 5 illustrates |
| `keep_cryosift_quantile` | 1 if `cryosift_score` is below the median of the 50 scores of that set, else 0. Exactly 25 per set by construction |

Only `empiar_id`, `stack`, `class_idx`, `n_particles` and `cryosift_score` are needed to
redraw Fig. 5. The rest come from the common-line geometry study, which found that the
geometry carries no information independent of the appearance score and was not written
up.

## Per-set summary of the column Fig. 5 draws

Baseline stack only, the one the figure uses. Computed from this file, so a reader can
check that their copy parses the way the figure expects.

| entry | classes | score range | classes at or above 4.5 | share of particles discarded |
| --- | ---: | --- | ---: | ---: |
| 10081 | 50 | 2.015 to 5.172 | 17 | 23.40 % |
| 10093 | 50 | 1.856 to 5.427 | 21 | 46.44 % |
| 10345 | 50 | 1.978 to 5.652 | 44 | 88.56 % |
| 10532 | 50 | 1.358 to 5.692 | 29 | 57.56 % |
