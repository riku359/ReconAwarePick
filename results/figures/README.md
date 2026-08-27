# Figures

One directory per figure, each holding the script or scripts that build it, a README
with the exact command, and any small input that is committed. Paths come from the
environment variables of [CONFIGURATION.md](../../docs/CONFIGURATION.md); a script that
cannot resolve one fails with the variable named.

Built figures are written to `$RAPICK_FIGURES_OUT`, which defaults to
`$RAPICK_WORK/figures`. Nothing here writes back into the repository, so "the figure in
the paper" and "the figure this checkout produces" stay separable.

## Index

| | Figure | Directory | Reads | Writes | Needs |
| --- | --- | --- | --- | --- | --- |
| Fig. 1 | The pipeline | [`pipeline_overview/`](pipeline_overview/) | photographic assets you supply with `--assets` | `pipeline_overview.pptx`, then a cropped PDF | python-pptx, LibreOffice, pypdf |
| Fig. 2 | What survives each stage | [`pick_fates/`](pick_fates/) | `$RAPICK_WORK/loop/<id>/round<n>/`, `$RAPICK_WORK/denoised/<id>/` | stage strips, then `pick_fates.pptx` | opencv, python-pptx, LibreOffice |
| Fig. 3 | Local-resolution maps | [`locres_maps/`](locres_maps/) | the refinement volumes named in each condition's `metrics.json` | `locres_maps*.pdf` | **ChimeraX** |
| Fig. 4 | 2D F1 against resolution | [`f1_vs_resolution/`](f1_vs_resolution/) | `../tables/main_results.json`, `../tables/detection_2d.json` | `f1_vs_resolution.pdf` | matplotlib |
| Fig. 5 | CryoSift class scores | [`cryosift_scores/`](cryosift_scores/) | `../tables/cryosift_class_scores.csv` | `cryosift_scores.pdf` | matplotlib |
| Fig. 6 | Where the mask fails | [`cleaner_failure/`](cleaner_failure/) | two contamination overlays | `cleaner_failure.pptx` | opencv, python-pptx, LibreOffice |
| Fig. S2, S3 | Mask post-processing | [`mask_postproc/`](mask_postproc/) | one micrograph and its two masks (S2); nothing (S3) | `mask_postproc_real.pdf`, `mask_blend_weights.pdf` | matplotlib, pillow |
| Fig. S5 | The first selection cycle | [`first_cycle/`](first_cycle/) | class tiles fetched from CryoSPARC | `first_cycle_<id>.png` | pillow, **CryoSPARC** (fetch) |
| Fig. S6, S7 | FSC and viewing directions | [`recon_diagnostics/`](recon_diagnostics/) | FSC and viewing-direction panels fetched from CryoSPARC | `recon/<id>/{fsc,viewing}_<condition>.png` | pillow, **CryoSPARC** (fetch) |
| (supplement) | Per-round loop diagnostics | [`loop_rounds/`](loop_rounds/) | `../tables/loop_rounds.json` | `loop_rounds.pdf` | matplotlib |

[`lib/`](lib/) holds what more than one figure uses: path and credential resolution, the
table reader, the CryoSPARC asset fetcher, the panel renderer, and the deck helpers.

`loop_rounds/` is the figure form of Table 6. The manuscript carries it in its candidate
supplement rather than in the main paper, so it has no figure number.

### Figures with no directory here

**Fig. S1** (the pipeline as the CryoSPARC jobs that run it) and **Fig. S4** (the
iterative workflow of the 2D class selection) are drawn in TikZ inside the manuscript, so
there is no script for either. Fig. S1 is drawn over panels that CryoSPARC renders for
its own jobs plus a few built here: `pick_fates/fullset_stage_overlay.py` and
`lib/prepare_overlay_panels.py --only protocol` cut its picking and mask panels, and
`first_cycle/build_protocol_cycles.py` and `first_cycle/build_teacher_strip.py` build its
class-average sheets and the strips that ride its arrows.

**Fig. S6 and Fig. S7** are assembled from CryoSPARC's own rendered panels rather than
drawn here; `recon_diagnostics/` picks the right job, keeps the last iteration and
shrinks the file.

## What cannot be reproduced byte-identically

- **Fig. 3, EMPIAR-10081 and 10532.** The ChimeraX camera placements for those two rows
  were approved by eye and frozen to JSON, and that JSON was lost. `fitmap search` starts
  from random placements, so a second freeze of the same row lands somewhere else: a
  re-render gives the same maps at a different orientation. The rendered panels survive
  only inside the committed figure PDF. The placements for EMPIAR-10093 and 10345 do
  survive, in [`locres_maps/poses/`](locres_maps/poses/), and that directory's README
  says how a row can be read back out of a PDF when its panels are gone.
- **Fig. S1, the class-average panels.** The fetched tiles were kept in scratch space and
  are gone. Re-running the conditions regenerates equivalent panels from jobs with
  different uids.
- **Every CryoSPARC job uid quoted in this directory.** They name jobs on the authors'
  instance. A fresh run produces the same job chain with different uids, which is why
  every uid here is a command-line argument with the paper's value as its default. The
  refine job behind each reconstruction panel is also recorded in
  `../tables/main_results.json` and `../tables/ablation.json`, so a panel and its table
  cell can be held to the same run.
- **Anything drawn over a photographic asset.** The micrograph thumbnails, particle
  crops, mask PNGs and class tiles are not committed: they are large binaries, and the
  repository commits code and numbers. Every script that needs them takes an `--assets`
  or `--strips` directory.

## Environment

`envs/figures/` is the environment these scripts run in. ChimeraX is a separate
application, not a Python dependency; `locres_maps/README.md` says how it is invoked. The
scripts that talk to CryoSPARC additionally need `cryosparc-tools`, which the `recon`
environment carries.
