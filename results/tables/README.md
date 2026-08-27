# Results tables

Every number the paper prints, in machine-readable form, together with the values behind
it: the three per-seed reconstructions each cell is the best of, the CryoSPARC job that
produced the winner, and the particle counts that reached it. The point is that a reader
can check the paper against the measurements without running anything, and can see which
cells rest on a comfortable margin and which do not.

Values were transcribed from three sources, in this order of authority:

1. the manuscript's `.tex` sources, which fix what the paper says;
2. the authors' provenance notes, which hold the unrounded values and the job uids;
3. the aggregated JSON exports in `revision/`.

Where the notes and the paper disagree, the paper wins for `published` and the notes'
value is kept alongside as `raw`. No value here was computed, interpolated or inferred:
anything that no source records is `null` or absent, and the accompanying `note` says so.

## Files

| file | paper table | label |
| --- | --- | --- |
| `datasets.json` | Table 1 | `tab:datasets` |
| `main_results.json` | Table 2 | `tab:main_results` |
| `mask_removals.json` | Table 3 | `tab:mask_removals` |
| `ablation.json` | Table 4 | `tab:ablation_res` |
| `particles.json` | Table 5 | `tab:particles` |
| `loop_rounds.json` | Table 6 | `tab:loop_rounds` |
| `teacher_quality.json` | Table 7 | `tab:teacher_quality` |
| `cryosegnet_purified.json` | Table 8 | `tab:cryosegnet_purified` |
| `hparams.json` | Table S1 | `tab:supp_hparams` |
| `detection_2d.json` | Table S2 | `tab:f1_vs_res` |
| `compute_cost.json` | Sec. S9, prose | none |

Also here:

- `cryosift_class_scores.csv` and `cryosift_class_scores.md`: the class scores behind
  Fig. 5, and a description of the columns.
- `revision/`: six aggregated JSON exports, copied unchanged except for one edit noted
  below. See the last section for what each one backs.

## Schema

Each file is one object:

```json
{
  "table": "tab:main_results",
  "paper_number": "Table 2",
  "caption": "<first sentence of the caption, plain text>",
  "units": "angstrom",
  "metric": "res_gsfsc_0143",
  "note": "<caveats that belong with the whole table>",
  "provenance": {"source": "<where the values came from>", "collected": "<date or null>"},
  "values": {"<condition>": {"<entry>": { ... }}}
}
```

`values` is keyed by condition and then by EMPIAR entry, `"10081"`, `"10093"`, `"10345"`,
`"10532"`. The condition names are the release vocabulary of
[`docs/PAPER_TO_CODE.md`](../../docs/PAPER_TO_CODE.md): `baseline`, `mask`, `select`,
`both`, `fb`, `fb_gt`, `cryolo`, `topaz`, `cryosegnet`, `cryosegnet_both`, `gt`. Where the
paper's row label differs from the release name, a `paper_row_names` block maps the two.

Inside a cell:

| field | meaning |
| --- | --- |
| `published` | the rounded number printed in the paper |
| `raw` | the unrounded value it was rounded from |
| `seeds` | the three per-seed values, seed 0, seed 1, seed 2 |
| `refine_job` | the CryoSPARC homogeneous refinement that produced the winning map |
| `particles` | particles delivered to the reconstruction |

Not every table has every field. `loop_rounds.json` is keyed by entry and then by round
`0` to `3`, each holding `picks`, `precision`, `recall`, `f1`, `after_purify_count` and
`after_purify_share`. `hparams.json` mirrors the two columns of Table S1 as `head_repair`
and `finetune`. Several files carry a `not_in_paper` block holding measurements the notes
record but the paper does not print, including rows that were drafted and then cut; those
values are never part of `values` and are labelled where they come from.

## Reading these numbers

### EMPIAR-10345 resolutions are about half the physical figure

The pixel size for EMPIAR-10345 follows CryoPPP's declared 0.673 A, which is the
super-resolution movie value rather than the physical one. Every resolution reported for
10345, in the paper and here, is therefore about half the figure you would compare against
EMDB. Multiply by about two to do that. Within the entry the conditions are still
comparable to each other, which is all the paper claims. This caveat is repeated in the
`note` of every file that carries a 10345 resolution.

### Best of three seeds

Every reconstruction was run three times with different random seeds, and every resolution
the paper prints is the best of the three by GSFSC 0.143 (Sec. 4.2). That convention holds
throughout the paper and throughout these files.

The three values behind each published one are recorded in the `seeds` field, so a cell can
be traced to the runs it came from. The paper reports the best of them and does not
characterise their spread; these files follow it and do the same.

### Job uids are the authors' own

Every `J`-prefixed identifier refers to the authors' CryoSPARC instance. They are recorded
so that a cell can be traced to the job that produced it, not because they can be resolved
anywhere else. A fresh run of the same pipeline produces the same chain of jobs with
different uids.

### Two tables share their numbers

The CryoTransformer row of Table 2 and the `baseline` row of Table 4 are the same
reconstructions, not two runs of the same condition. Likewise the Ours row of Table 2, the
`fb` row of Table 4, the first row of Table 7 and the last row of Table 8 are all the same
four cells, and the CryoSegNet row of Table 8 repeats Table 2. Each file's `note` says
which of its rows are repeats.

## What is in `revision/`

Aggregated exports the manuscript revision produced, copied unchanged. Filenames are the
authors' own.

| file | what it holds | which claim it backs |
| --- | --- | --- |
| `ablation_2d_metrics.json` | 2D precision, recall, F1 and candidate counts of the five ablation conditions on the annotated micrographs | **Table 3** in full: the removal counts, the annotated-particle counts and the mistaken rate are all differences between its `baseline` and `mask` blocks. Its `baseline` block also reproduces the CryoTransformer row of **Table S2** to three decimals, which is the check that the two protocols agree |
| `e1_purified_pickers.json` | twelve purified picker conditions, three seeds each, with refinement job uids | the `CryoSegNet + mask + select` row of **Table 8**, including its per-seed values |
| `cost.json` | per-stage wall clock for the loop rounds and the full-set chains, plus the node's hardware | every number in **Sec. S9**, the compute-time section |
| `contam_survival.json` | how much predicted contamination survives 2D selection, per entry, and the mask's calibration | the mask stage of **Table 5**: its `known_mask_pct` values, 5.56 / 0.13 / 1.91 / 4.63 %, are the per-stage removal rates behind that row's cumulative -5.6 / -0.1 / -1.9 / -4.6. Its `entries` block also backs a paragraph on contamination surviving selection that was cut from Sec. 5.3 |
| `locres.json` | local-resolution quartiles of every reconstruction inside its own refinement mask, with the refinement and local-resolution job uids | the local-resolution reading of **Fig. 3**, that our map resolves the most fine-grained structure of the five, and the colour scales shared within each row of that figure |
| `locres_common.json` | the same quartiles measured inside the intersection of an entry's five refinement masks, so that conditions are compared over the same voxels | the quantitative counterpart of the same Fig. 3 claim. Refinement masks within one entry differ in size by 3 % on 10081 and 65 % on 10093, which is why the common region exists at all. The subsection that read these quartiles cell by cell was drafted and then cut from Sec. 6 for length; on three of the four entries it followed the global ranking and on EMPIAR-10345 it contradicted it |

One field was edited. `ablation_2d_metrics.json` recorded `protocol.gt` as an absolute
path on the authors' server; it now reads
`CryoPPP ground-truth star of each entry: <cryoppp>/<id>/ground_truth/empiar-<id>_particles_selected.star (absolute server path replaced for release)`.
Nothing else in these files was changed.

The per-micrograph mask dumps and the contamination-survivor coordinate lists that sat
beside these files in the manuscript repository are not included. They back prose that was
cut from the paper.
