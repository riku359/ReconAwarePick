# Paper to code

Every table and figure of the paper, and what produces it. Commands are in
[REPRODUCE.md](REPRODUCE.md); this page is the map.

## Conditions

The paper names five conditions. They are the names used throughout this
repository: in `configs/conditions/`, in the `--condition` flag of every driver,
and in the output path `$RAPICK_WORK/empiar_<id>/<setting>/<condition>/`.

| Condition | Picks from | Contamination mask | 2D class selection | Appears as |
| --- | --- | :---: | :---: | --- |
| `baseline` | CryoTransformer, base checkpoint | | | Table 2 (CryoTransformer row), Table 4 row 1 |
| `mask` | same | yes | | Table 4 row 2 (+mask) |
| `select` | same | | yes | Table 4 row 3 (+select) |
| `both` | same | yes | yes | Table 4 row 4 (+both), Table 8 |
| `fb` | CryoTransformer, round-1 checkpoint | yes | yes | Table 2 (**Ours**), Table 4 row 5, Table 7 |

Three further conditions exist for the comparisons:

| Condition | What it is |
| --- | --- |
| `cryolo`, `topaz`, `cryosegnet` | the other three pickers, raw picks, no purification (Table 2) |
| `cryosegnet_both` | CryoSegNet's candidates through the same mask and 2D selection (Table 8) |
| `gt` | a reconstruction from the CryoPPP annotations of the 300 annotated micrographs (Fig. S6, Fig. S7) |
| `fb_gt` | one round of the loop with the CryoPPP annotations as the teacher (Table 7, lower row) |

`<setting>` is `annot` (the 300 CryoPPP-annotated micrographs, used by the loop
and by every 2D metric) or `full` (the whole deposition, used by every
reconstruction-level result).

### If you are reading the private research repository

The research monorepo this release was assembled from names the same conditions
differently, and the correspondence is not guessable. It is recorded here so the
two can be cross-checked.

| Here | In the research repo |
| --- | --- |
| `baseline` | `cryotransformer` |
| `mask` | `cryotransformer_clean_tri` |
| `select` | `cryotransformer_cryosift_iter` |
| `both` | `cryotransformer_clean_tri_cryosift_iter` |
| `fb` | `fbf_r1_clean_tri_cryosift_iter`, loop arm `general_full`, prefix `fbf_r` |
| `fb_gt` | loop arm driven by the GT teacher, prefix `fbgt_r` |

The loop arms `general` and `chained` in that repository fine-tune with LoRA and
are **not** the paper's method. The paper fine-tunes every weight except the
first residual stage of the backbone (`--finetune_mode head_decoder_encoder_resnet`).

## Tables

| | What it holds | Produced by |
| --- | --- | --- |
| Table 1 | The four EMPIAR entries, particle diameter, pixel size, micrograph counts | `configs/datasets/empiar_<id>.yaml`; the counts are checked by `scripts/01_download_data.sh` |
| Table 2 | GSFSC 0.143 resolution of four pickers and Ours, full sets | `src/rapick/recon`, one run per condition; collected into `results/tables/main_results.json` |
| Table 3 | Candidates the mask removes, split by whether they overlap an annotated particle | `results/analysis/ablation_2d_metrics.py`, as the difference between the `baseline` and `mask` conditions: removals are the drop in matched candidates, and the particle row is the drop in true positives |
| Table 4 | The ablation, five conditions x four entries | the same reconstruction runs as Table 2, conditions `baseline` / `mask` / `select` / `both` / `fb` |
| Table 5 | Particles retained through the pipeline | the `particle_counts` block of each condition's `metrics.json` |
| Table 6 | Per-round loop diagnostics on the 300 annotated micrographs | `src/rapick/loop/fb_round_metrics.py`, scoring with `src/rapick/eval/calc_common_2d_metrics.py` |
| Table 7 | Pseudo-labels against a perfect teacher | conditions `fb` and `fb_gt` |
| Table 8 | The same selection applied to CryoSegNet | conditions `cryosegnet`, `cryosegnet_both`, `both`, `fb` |
| Table S1 | Hyperparameters of the two training stages | `src/rapick/picker/` — see its README |
| Table S2 | 2D detection scores of the four base pickers | `src/rapick/eval/calc_common_2d_metrics.py` |

## Figures

| | What it shows | Produced by |
| --- | --- | --- |
| Fig. 1 | The pipeline | hand-drawn, `results/figures/pipeline_overview/` |
| Fig. 2 | What survives each stage, on one micrograph per entry | `src/rapick/loop/fb_export_stage_stars.py` then the overlay renderer |
| Fig. 3 | Local-resolution maps of 10081 and 10532 | ChimeraX, `results/figures/locres_maps/` |
| Fig. 4 | 2D F1 against resolution | `results/figures/f1_vs_resolution/`; reads Table 2 and Table S2 |
| Fig. 5 | CryoSift scores of the 50 classes of each entry | `results/figures/cryosift_scores/`; reads `results/tables/cryosift_class_scores.csv` |
| Fig. 6 | Where the contamination mask fails on 10532 | `src/rapick/cleaner/` overlays |
| Fig. S1 | The pipeline as the CryoSPARC jobs that run it | TikZ in the manuscript, over panels CryoSPARC renders for its own jobs |
| Fig. S2 | The two mask post-processings on a real micrograph | `src/rapick/cleaner/plot_mask_postproc_figures.py` |
| Fig. S3 | Uniform averaging against triangular blending | `src/rapick/cleaner/plot_mask_postproc_figures.py` |
| Fig. S4 | The iterative workflow of CryoSift | TikZ in the manuscript |
| Fig. S5 | What the first cycle of 2D class selection does | `src/rapick/select2d/plot_selection_panel.py` |
| Fig. S6 | Gold-standard FSC of every reconstruction | CryoSPARC's own FSC panel for each refinement job |
| Fig. S7 | Viewing directions of every reconstruction | CryoSPARC's own viewing-direction panel |

Fig. S1, S6 and S7 are assembled from the images CryoSPARC renders for its own
jobs. They are reproduced by running the conditions and then reading each job's
rendered panels out of the CryoSPARC project; they are not drawn by any script
here.

## What this repository cannot reproduce exactly

Stated plainly, because a reader will otherwise spend time on it.

- **Fig. 3, the local-resolution maps for 10081 and 10532.** The ChimeraX camera
  placements were approved by eye and frozen to JSON, and that JSON was lost. The
  rendered panels survive only inside the committed figure PDF. A re-render will
  produce the same maps at a different orientation. The placements for 10093 and
  10345 survive in `results/figures/locres_maps/poses/`.
- **Fig. S1, the class-average panels.** The tiles were fetched from the CryoSPARC
  project into scratch space and are gone. Re-running the conditions regenerates
  equivalent panels, from jobs with different uids.
- **Anything requiring the original CryoSPARC projects.** Job uids quoted in the
  manuscript's notes refer to the authors' instance. A fresh run produces the same
  chain with different uids.
