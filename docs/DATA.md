# Data

Inputs download into `$RAPICK_DATA` and the published artifacts of earlier stages into
`$RAPICK_WORK`, neither of which is ever inside this repository; both layouts are in the
README, and the variables themselves in [CONFIGURATION.md](CONFIGURATION.md).

```bash
bash scripts/download.sh                              # everything, all four entries
RAPICK_ENTRIES=10081 bash scripts/download.sh         # everything, one entry
bash scripts/download/04_full_depositions.sh          # one source on its own
```

`download.sh` takes no arguments. Each source has its own script under
[`scripts/download/`](../scripts/download/), and `download.sh` runs them in name order,
so what to fetch is chosen by which script you run rather than by a flag:

| Script | Fetches |
| --- | --- |
| `01_cryoppp_catalog.sh` | the CryoPPP catalogue every downloader after it reads |
| `02_annotated_micrographs.sh` | the annotated 300 per entry (~75 GB) |
| `03_expert_annotations.sh` | CryoPPP's expert particle annotations |
| `04_full_depositions.sh` | the whole EMPIAR deposition of each entry (~1.5 TB) |
| `05_verify_micrographs.sh` | recovers failed transfers, checks integrity, prints the counts |
| `06_cryotransformer_weights.sh` | CryoTransformer's released weights (~3 GB) |
| `07_micrograph_cleaner_weights.sh` | MicrographCleaner's contamination network (~127 MiB) |
| `08_published_artifacts.sh` | theta_0, the contamination masks, the masked picks |
| `09_picker_candidates.sh` | the four pickers' full-set candidates |
| `10_finetuned_checkpoints.sh` | the round-1 fine-tuned checkpoints (~870 MB each) |

The last three come from Hugging Face and are what makes a stage enterable without
re-deriving its input: `08` replaces `repair_head.sh` and `contamination_removal.sh`,
`09` replaces `pick.sh`, and `10` replaces `loop.sh`.

Resumable and idempotent: already-placed files are skipped, partial transfers resume
with `curl --continue-at -`, nothing is overwritten, and `sudo` is never used. A run
interrupted anywhere is restarted by running `download.sh` again.

## The four entries

CryoTransformer's independent test set.

| EMPIAR | Sample | Diameter (px) | Pixel size (Å) | Annotated | Full |
| --- | --- | ---: | ---: | ---: | ---: |
| 10081 | HCN1 | 154 | 1.30 | 300 | 997 |
| 10093 | NOMPC | 172 | 1.22 | 300 | 1,873 |
| 10345 | integrin αVβ8 | 149 | 0.673 † | 300 | 1,644 |
| 10532 | hemagglutinin | 174 | 1.03 | 300 | 1,556 |

† **10345's pixel size is CryoPPP's declared value, which is the super-resolution movie
figure.** The micrographs are 2×-binned at 1.345 Å — as the EMPIAR image set, the
authors' particle stack and EMD-20795's voxel size all agree. We follow CryoPPP so that
every entry uses CryoPPP-sourced optics, which means **every 10345 resolution here is
about half the physical figure**; multiply by about two before comparing against EMDB.
Coordinates and the extraction box are unaffected. The paper states this as a limitation.

The 300 annotated micrographs per entry supply the ground truth for the 2D detection
metrics and are the working set of the feedback loop. Every reconstruction-level result
uses the full deposition: 300 micrographs are not enough for a stable reconstruction.

## What gets downloaded

### From EMPIAR and CryoPPP

| Asset | Path under `$RAPICK_DATA` | Size | From |
| --- | --- | --- | --- |
| Annotated micrographs | `cryoppp/<id>/micrographs/*.mrc` | ~75 GB | [EMPIAR](https://www.ebi.ac.uk/empiar/), listed by the [CryoPPP](https://github.com/BioinfoMachineLearning/cryoppp) catalogue |
| Expert annotations | `cryoppp/<id>/ground_truth/empiar-<id>_particles_selected.star` | small | the `cryoppp_lite` archives at <https://calla.rnet.missouri.edu/cryoppp_lite/>, unpacked keeping only the `.star` files |
| Full depositions | `cryoppp_fullset/<id>/micrographs/*.mrc` | ~1.5 TB | [EMPIAR](https://ftp.ebi.ac.uk/empiar/world_availability/) |
| CryoTransformer's released weights | `checkpoints/CryoTransformer_pretrained_model.pth` | ~3 GB | <https://calla.rnet.missouri.edu/CryoTransformer/pretrained_model.tar.gz> |
| MicrographCleaner's released weights | `checkpoints/micrograph_cleaner_defaultModel.h5` | ~127 MiB | [Zenodo](https://zenodo.org/records/17093439), through `src/rapick/cleaner/download_model.sh` |
| CryoSift's weights | inside the Magellon checkout | 33 MB | bundled in the upstream clone; no separate download |

The full-set source directory differs per entry, and two need a filter. The table lives
in the `DATASETS` block of `src/rapick/data/download_empiar_fullset.py`:

| EMPIAR | EMPIAR directory | `.mrc` | Note |
| --- | --- | ---: | --- |
| 10081 | `10081/data/micrographs/` | 997 | |
| 10093 | `10093/data/NOMPC/` | 1,873 | |
| 10345 | `10345/data/Micrographs_18jam15a/` | 1,644 | |
| 10532 | `10532/data/02_Aligned_Micrographs/motioncorrected/` | 1,556 | keeps only `*_patch_aligned.mrc` |

### From Hugging Face

| Asset | Script | Repository | Path under `$RAPICK_DATA` / `$RAPICK_WORK` |
| --- | --- | --- | --- |
| theta_0, the repaired base checkpoint | `08` | [weights](https://huggingface.co/rikrikrik/recon-aware-pick-weights) | `$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth` |
| Triangular-blend contamination masks, all four entries at full-set scale (6,070 `.npz`) | `08` | [data](https://huggingface.co/datasets/rikrikrik/recon-aware-pick-data) | `$RAPICK_WORK/masks/<id>/` |
| Picks after contamination masking, all four entries | `08` | data | `$RAPICK_WORK/picks/<id>/cryotransformer_mask.star` |
| The four pickers' full-set candidates | `09` | data | `$RAPICK_WORK/picks/<id>/{cryotransformer,cryolo,topaz,cryosegnet}.star` |
| Round-1 fine-tuned checkpoints, one per entry | `10` | weights | `$RAPICK_DATA/checkpoints/loop_fb_round1_empiar_<id>.pth` |

The round-1 checkpoints are what the `fb` arm picks with, so with them the paper's
headline row needs no loop run at all. The perfect-teacher arm of Table 7's lower row is
not reachable from these scripts; fetch it directly:

```bash
uv run --with huggingface_hub python3 src/rapick/data/hf_assets.py download \
    --repo-weights rikrikrik/recon-aware-pick-weights \
    --data-root "$RAPICK_DATA" --ids 10081 --with-loop-checkpoints fb_gt
```

The picks are what Table 2 and Table S2 need, so having them means not installing
crYOLO, Topaz or CryoSegNet ([BASELINES.md](BASELINES.md)). Each lands under its
picker's name, and the masked CryoTransformer picks under
`cryotransformer_mask.star` — the same names
[`scripts/contamination_removal.sh`](../scripts/contamination_removal.sh) would have
written, so a downloaded stage and a locally derived one are interchangeable.

## Verifying a download

**A downloaded `.mrc` can be silently corrupt in two ways that checking for the file's
existence will not catch.** Parallel downloaders appending to one `.part` file produce a
truncated or oversized file; EBI's S3 endpoint sometimes returns a
`ConnectionClosedException` XML body that lands inside the `.mrc`, which then has the
right size and fails hours later at Patch CTF.

```bash
python3 src/rapick/data/verify_mrc_integrity.py --dataset fullset --ids 10093 --full
```

EMPIAR's older-format `.mrc` files lack the `MAP ` stamp at byte offset 208, so a strict
signature check reports valid files as broken; the verifier accounts for this.

Confirm the expected count before starting a run: `import_particles` dies on the first
missing micrograph, and a `*.mrc` glob happily imports a partial download.
`scripts/download/05_verify_micrographs.sh` does all of this, and `download.sh` runs it
after the micrographs are in: it recovers what failed, verifies the files, and prints
each entry's count against the expected one.

## Licensing

EMPIAR, CryoPPP and what we derive from them: the README's License section.
