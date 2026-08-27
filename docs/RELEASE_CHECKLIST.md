# Release checklist

This repository is complete enough to read and to run, but it has not been made
public. This page is what is left, in the order it makes sense to do it.

Delete this file when the repository goes public: by then it is either done or it
belongs in the README's TODO.

## Where it stands

The repository exists **private** on GitHub. It stays private: the paper is under
review, the abstract says the code will be available upon acceptance, and a public
repository under the authors' own account would break the anonymity of the
submission.

`bash scripts/check_release.sh` is the gate and passes. It checks paths and account
names from the experiment machines, credentials, files over 1 MB, leftover Japanese
prose, dead links, documented paths that do not exist, and the private
repository's condition names appearing outside the translation table.
`--links` additionally fetches every external URL.

## Before it goes public

- [x] **Check the repository on a machine that has never run this code.** Done on
      a fresh clone on the lab's GPU host. What was verified:
      - the hygiene gate passes on Linux (it did not at first: see below)
      - `00_setup.sh` clones all three upstreams at their pins and applies the
        CryoTransformer overlay, and the result matches the shipped overlay file
        for file
      - the `figures` and `recon` environments build from their lockfiles
      - `08_tables_figures.sh` rebuilds Fig. 4, Fig. 5 and the loop-rounds figure,
        and **Fig. 4 comes out identical to the one in the paper**
      - `rapick-recon check-setup` passes against the live CryoSPARC instance:
        connection, project, 300 healthy micrographs, distinct STARs
      - `--dry-run` works for both kinds of condition, and a condition whose parent
        has not run yet refuses with the command that would fix it
      - `01_download_data.sh --dry-run` resolves the real remote data and counts
        **997 micrographs for EMPIAR-10081, matching Table 1**
      - all five stage environments build, including the two that are not uv
        projects and the Python 3.7 one the picker needs, and every CLI runs
      - **47 of Table S2's 48 published cells reproduce exactly** from the published
        picks. The one that does not is Topaz's F1 on EMPIAR-10345, 0.483 against
        0.482, whose precision and recall both reproduce: a macro F1 landing on a
        rounding boundary, not a different measurement
- [ ] **Run one condition through to a reconstruction.** The above stops short of
      creating CryoSPARC jobs. The smallest useful test is EMPIAR-10081, condition
      `mask`, at `annot` scale, from the published intermediates.
- [x] Publish the four round-1 fine-tuned checkpoints to
      `rikrikrik/recon-aware-pick-weights`. Done, along with the four of the
      perfect-teacher arm. Each carries its own training arguments, and they read
      back as the paper's method: `finetune_mode=head_decoder_encoder_resnet`,
      50 epochs, lr 1e-4 with backbone lr 1e-5, 600 queries, resumed from theta_0.
- [x] Publish the four pickers' full-set picks to
      `rikrikrik/recon-aware-pick-data`. Done, and checked end to end: downloading
      them onto a fresh clone and scoring them reproduces **every value of Table S2
      for EMPIAR-10081 exactly**, all four pickers, all three metrics.
- [ ] Run the `fb_gt` path once. Its checkpoints are now published, so the
      reconstruction half of Table 7's lower row can be reproduced without it, but
      the loop half is still a reimplementation: the scripts that produced the
      published row were never committed, and `--teacher gt` follows their
      documented procedure without having been run in this form.
- [ ] Replace the citation block in `README.md` and `CITATION.cff` with the
      proceedings citation.
- [ ] Decide whether to archive a release on Zenodo for a DOI, as is usual for a
      paper's code.

## What running it caught

Every one of these was invisible to reading and only appeared when the repository
was cloned somewhere else and used. They are fixed; the list is here because it is
the argument for doing the same again before going public.

| What | Why it was invisible |
| --- | --- |
| The hygiene gate failed on Linux, flagging every emoji as Japanese prose | GNU grep under `LC_CTYPE=POSIX` compares a character range byte by byte; macOS's grep does not |
| Setup cloned Magellon whole, 2.2 GB instead of 166 MB | `repos.lock.yaml` recorded the subdirectory and the clone helper ignored it |
| Two of the five environments were never built | they are not uv projects, and setup skipped them with a message no one reads |
| The 2D selection environment would not build at all | upstream pins `pysqlite3`, which has no wheel and needs a system header nothing here imports |
| `--dry-run` died on an argparse error for six of the eleven conditions | `rapick-recon run` has no such flag, and only the from-selection driver was tested by hand |
| The released picker weights were referenced everywhere and downloadable from nowhere | every machine that ran the experiments already had them |
| `finetune.py --help` died: it could not find the repository's STAR reader | its fallback path was right only while the file sat in the repository, and setup copies it into the clone |
| `--help` needed the data roots set before it would print anything | nobody runs `--help` on a machine that has no data |

## Two things to settle in the manuscript, not here

Both were found while checking the paper's numbers against their sources. Neither
changes a reported result, and both are recorded in `results/tables/` rather than
fixed, because the manuscript is the right place to resolve them.

- **The 2D scores cover 295 annotated micrographs on EMPIAR-10093 and 10345, not
  the 300 the paper states.** CryoPPP deposits 300 per entry, but only 295 carry
  annotations on those two, and the scorer averages over the ones that do.
  `results/tables/datasets.json` records the count per entry in
  `micrographs_scored_2d`.
- **Table 6's round 0 and Table S2's CryoTransformer row disagree**, although both
  describe the base checkpoint on the annotated micrographs: 0.530 / 0.919 / 0.655
  against 0.469 / 0.954 / 0.610 on EMPIAR-10081, from 65,385 picks against 77,328.
  The difference is not the contamination mask, which the loop applies after the
  count that Table 6 prints. `results/tables/loop_rounds.json` records both under
  `not_in_paper`.

## What this repository deliberately does not contain

Recorded so that each omission reads as a decision rather than an oversight.

- **Upstream code.** Topaz is GPL-3.0 and crYOLO is distributed under a
  non-commercial licence, so neither can live inside an MIT repository.
  `scripts/00_setup.sh` clones every upstream at the commit `repos.lock.yaml`
  pins. The one exception is three CryoTransformer files that carry our changes;
  they are MIT, they ship in `src/rapick/picker/overlay/`, and the changes
  themselves are readable in `src/rapick/picker/patches/`.
- **The literature corpus.** The research repository holds full-text conversions
  of 158 published papers. They are third-party copyrighted works and are not
  redistributable.
- **Progress decks, machine notes, and abandoned directions.** The research
  repository's slide decks, its per-machine operational notes, and the
  common-lines geometry work that its own records close as a negative result are
  all outside what this paper needs.
- **Large binaries.** No micrograph, particle crop, mask, class tile or density
  map is committed. Every script that needs one takes an `--assets` directory, and
  the artifacts worth publishing go to Hugging Face.
