# Release checklist

This repository is complete enough to read and to run, but it has not been made
public. This page is what is left, in the order it makes sense to do it.

Delete this file when the repository goes public: by then it is either done or it
belongs in the README's TODO.

## Before the repository leaves this machine

- [ ] `bash scripts/check_release.sh` passes. It is the gate: paths and account
      names from the experiment machines, credentials, files over 1 MB, leftover
      Japanese prose, dead links, and the private repository's condition names
      appearing outside the translation table.
- [ ] The paper is still under review. The abstract says code will be available
      upon acceptance, and publishing a repository under the authors' own GitHub
      account would break the anonymity of the submission. **Create it private.**

```bash
gh repo create ReconAwarePick --private --source . --remote origin --push
```

## Before it goes public

- [ ] **Run one condition end to end on a machine that has never run this code.**
      Nothing else on this list matters as much. Everything here was ported and
      checked by reading; none of it has been executed against a live CryoSPARC
      instance. Suggested smallest useful test: EMPIAR-10081, condition `mask`,
      from the published intermediates.
- [ ] Publish the four round-1 fine-tuned checkpoints to
      `rikrikrik/recon-aware-pick-weights`. Without them the `fb` condition, which
      is the paper's headline row, can only be reproduced by re-running the loop.
- [ ] Publish the four pickers' full-set picks to
      `rikrikrik/recon-aware-pick-data`, so Table 2 and Table S2 can be reproduced
      without installing crYOLO, Topaz and CryoSegNet. `--picks` in
      `scripts/01_download_data.sh` already expects them and currently says so.
- [ ] `uv lock` in `envs/figures/` and commit the result. It is the one
      environment assembled for the release rather than during the experiments,
      so it is the one without a lockfile.
- [ ] Run the `fb_gt` path once, or mark Table 7's lower row as not reproducible
      here. The scripts that produced it were never committed; `--teacher gt`
      reimplements their documented procedure and has not been run in this form.
- [ ] Replace the citation block in `README.md` and `CITATION.cff` with the
      proceedings citation.
- [ ] Decide whether to archive a release on Zenodo for a DOI, as is usual for a
      paper's code.

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
