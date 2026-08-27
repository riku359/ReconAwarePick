# Paper to code

What produces each table and figure. Commands are in [REPRODUCE.md](REPRODUCE.md).

## Conditions

The paper's five conditions, named the same everywhere in this repository: in
`configs/conditions/`, in every driver's `--condition` flag, and in the output path
`$RAPICK_WORK/empiar_<id>/<setting>/<condition>/`.

| Condition | Picks from | Contamination mask | 2D class selection | Appears as |
| --- | --- | :---: | :---: | --- |
| `baseline` | CryoTransformer, base checkpoint | | | Table 2 (CryoTransformer row), Table 4 row 1 |
| `mask` | same | yes | | Table 4 row 2 (+mask) |
| `select` | same | | yes | Table 4 row 3 (+select) |
| `both` | same | yes | yes | Table 4 row 4 (+both), Table 8 |
| `fb` | CryoTransformer, round-1 checkpoint | yes | yes | Table 2 (**Ours**), Table 4 row 5, Table 7 |

Four more exist for the comparisons:

| Condition | What it is |
| --- | --- |
| `cryolo`, `topaz`, `cryosegnet` | the other three pickers, raw picks, no purification (Table 2) |
| `cryosegnet_both` | CryoSegNet's candidates through the same mask and 2D selection (Table 8) |
| `gt` | a reconstruction from the CryoPPP annotations of the 300 annotated micrographs (Fig. S6, Fig. S7) |
| `fb_gt` | one round of the loop with the CryoPPP annotations as the teacher (Table 7, lower row). A reimplementation — see the last section. |

`<setting>` is `annot` (the 300 CryoPPP-annotated micrographs, used by the loop and by
every 2D metric) or `full` (the whole deposition, used by every reconstruction-level
result).

### If you are reading the private research repository

It names the same conditions differently, and the correspondence is not guessable:

| Here | In the research repo |
| --- | --- |
| `baseline` | `cryotransformer` |
| `mask` | `cryotransformer_clean_tri` |
| `select` | `cryotransformer_cryosift_iter` |
| `both` | `cryotransformer_clean_tri_cryosift_iter` |
| `fb` | `fbf_r1_clean_tri_cryosift_iter`, loop arm `general_full`, prefix `fbf_r` |
| `fb_gt` | loop arm driven by the GT teacher, prefix `fbgt_r` |

Its loop arms `general` and `chained` fine-tune with LoRA and are **not** the paper's
method. The paper fine-tunes every weight except the first residual stage of the
backbone (`--finetune_mode head_decoder_encoder_resnet`).

## Tables

| | What it holds | Produced by |
| --- | --- | --- |
| Table 1 | The four EMPIAR entries, particle diameter, pixel size, micrograph counts | `configs/datasets/empiar_<id>.yaml`; counts checked by `scripts/01_download_data.sh` |
| Table 2 | GSFSC 0.143 resolution of four pickers and Ours, full sets | `src/rapick/recon`, one run per condition; collected into `results/tables/main_results.json` |
| Table 3 | Candidates the mask removes, split by whether they overlap an annotated particle | `results/analysis/ablation_2d_metrics.py`, as `baseline` minus `mask`: removals are the drop in matched candidates, the particle row the drop in true positives |
| Table 4 | The ablation, five conditions × four entries | the same runs as Table 2, conditions `baseline` / `mask` / `select` / `both` / `fb` |
| Table 5 | Particles retained through the pipeline | the `particle_counts` block of each condition's `metrics.json` |
| Table 6 | Per-round loop diagnostics on the 300 annotated micrographs | `src/rapick/loop/round_metrics.py`, scoring with `src/rapick/eval/calc_common_2d_metrics.py` |
| Table 7 | Pseudo-labels against a perfect teacher | conditions `fb` and `fb_gt` |
| Table 8 | The same selection applied to CryoSegNet | conditions `cryosegnet`, `cryosegnet_both`, `both`, `fb` |
| Table S1 | Hyperparameters of the two training stages | `src/rapick/picker/` — see its README |
| Table S2 | 2D detection scores of the four base pickers | `src/rapick/eval/calc_common_2d_metrics.py` |

## Figures

| | What it shows | Produced by |
| --- | --- | --- |
| Fig. 1 | The pipeline | hand-drawn, `results/figures/pipeline_overview/` |
| Fig. 2 | What survives each stage, on one micrograph per entry | `src/rapick/loop/export_stage_stars.py`, then `results/figures/pick_fates/` |
| Fig. 3 | Local-resolution maps of 10081 and 10532 | ChimeraX, `results/figures/locres_maps/` |
| Fig. 4 | 2D F1 against resolution | `results/figures/f1_vs_resolution/`; reads Table 2 and Table S2 |
| Fig. 5 | CryoSift scores of the 50 classes of each entry | `results/figures/cryosift_scores/`; reads `results/tables/cryosift_class_scores.csv` |
| Fig. 6 | Where the contamination mask fails on 10532 | `results/figures/cleaner_failure/`, over overlays `src/rapick/cleaner/` renders |
| Fig. S1 | The pipeline as the CryoSPARC jobs that run it | TikZ in the manuscript, over panels CryoSPARC renders for its own jobs |
| Fig. S2 | The two mask post-processings on a real micrograph | `results/figures/mask_postproc/`, over panels `src/rapick/cleaner/plot_mask_postproc_figures.py` cuts |
| Fig. S3 | Uniform averaging against triangular blending | `results/figures/mask_postproc/`; needs no input |
| Fig. S4 | The iterative workflow of CryoSift | TikZ in the manuscript |
| Fig. S5 | What the first cycle of 2D class selection does | `results/figures/first_cycle/` |
| Fig. S6 | Gold-standard FSC of every reconstruction | CryoSPARC's own FSC panel for each refinement job |
| Fig. S7 | Viewing directions of every reconstruction | CryoSPARC's own viewing-direction panel |

Fig. S1, S6 and S7 are assembled from images CryoSPARC renders for its own jobs: run the
conditions, then read each job's rendered panels out of the project. No script here
draws them.

## What this repository cannot reproduce exactly

Stated plainly, because a reader will otherwise spend time on it.

- **Fig. 3, the local-resolution maps for 10081 and 10532.** The ChimeraX camera
  placements were approved by eye and frozen to JSON, and that JSON was lost; the
  rendered panels survive only inside the committed figure PDF. A re-render produces the
  same maps at a different orientation. The 10093 and 10345 placements survive in
  `results/figures/locres_maps/poses/`.
- **Fig. S1, the class-average panels.** The tiles were fetched into scratch space and
  are gone. Re-running the conditions regenerates equivalent panels, from jobs with
  different uids.
- **Anything requiring the original CryoSPARC projects.** Job uids quoted in the
  manuscript's notes refer to the authors' instance.
- **The lower row of Table 7.** The two scripts that built its GT teacher and ran its
  fine-tune lived on a lab server and were never committed. `--teacher gt` and
  `src/rapick/loop/make_gt_teacher.py` reimplement what they are documented to have
  done, and have not been run end to end in this form.
