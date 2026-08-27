# Data

Everything is downloaded into `$RAPICK_DATA`, which is never inside this
repository. See [CONFIGURATION.md](CONFIGURATION.md) for the layout.

```bash
bash scripts/01_download_data.sh                       # all four entries
bash scripts/01_download_data.sh --entry 10081         # one entry
bash scripts/01_download_data.sh --intermediates       # add the published artifacts
bash scripts/01_download_data.sh --dry-run             # enumerate and check free space
```

The scripts are resumable and idempotent: they skip what is already in place,
resume partial transfers with `curl --continue-at -`, and never overwrite an
existing file. `sudo` is never used.

## The four entries

These are the EMPIAR entries that form CryoTransformer's independent test set.

| EMPIAR | Sample | Diameter (px) | Pixel size (Å) | Annotated | Full |
| --- | --- | ---: | ---: | ---: | ---: |
| 10081 | HCN1 | 154 | 1.30 | 300 | 997 |
| 10093 | NOMPC | 172 | 1.22 | 300 | 1,873 |
| 10345 | integrin αVβ8 | 149 | 0.673 † | 300 | 1,644 |
| 10532 | hemagglutinin | 174 | 1.03 | 300 | 1,556 |

† **EMPIAR-10345's pixel size is CryoPPP's declared value, and it is the
super-resolution movie figure.** The micrographs themselves are 2×-binned at
1.345 Å, which the EMPIAR image set, the authors' own particle stack and
EMD-20795's voxel size all agree on. We deliberately follow CryoPPP so that every
entry uses CryoPPP-sourced optics, which means **every 10345 resolution in this
project is about half the physical figure**. Multiply by about two before
comparing against EMDB or against a paper that declared 1.345. Coordinates and
the extraction box are unaffected. The paper states this as a limitation.

The annotated 300 micrographs per entry supply the ground truth for the 2D
detection metrics and are the working set of the feedback loop. Every
reconstruction-level result uses the full deposition, because 300 micrographs are
not enough for a stable reconstruction.

## What gets downloaded

### From EMPIAR and CryoPPP

| Asset | Path under `$RAPICK_DATA` | Size | From |
| --- | --- | --- | --- |
| Annotated micrographs | `cryoppp/<id>/micrographs/*.mrc` | ~75 GB | [EMPIAR](https://www.ebi.ac.uk/empiar/), listed by the [CryoPPP](https://github.com/BioinfoMachineLearning/cryoppp) catalogue |
| Expert annotations | `cryoppp/<id>/ground_truth/empiar-<id>_particles_selected.star` | small | the `cryoppp_lite` archives at <https://calla.rnet.missouri.edu/cryoppp_lite/> |
| Full depositions | `cryoppp_fullset/<id>/micrographs/*.mrc` | ~1.5 TB | [EMPIAR](https://ftp.ebi.ac.uk/empiar/world_availability/) |
| CryoTransformer's released weights | `checkpoints/CryoTransformer_pretrained_model.pth` | ~3 GB | <https://calla.rnet.missouri.edu/CryoTransformer/pretrained_model.tar.gz> |
| MicrographCleaner's released weights | `checkpoints/micrograph_cleaner_defaultModel.h5` | 15 MB | [Zenodo](https://zenodo.org/records/17093439), fetched by `src/rapick/cleaner/download_model.sh` |
| CryoSift's weights | inside the Magellon checkout | 33 MB | bundled in the upstream clone; no separate download |

The full-set source directory differs per entry, and two of them need a filter.
The table is in the `DATASETS` block at the top of
`src/rapick/data/download_empiar_fullset.py`:

| EMPIAR | EMPIAR directory | `.mrc` | Note |
| --- | --- | ---: | --- |
| 10081 | `10081/data/micrographs/` | 997 | |
| 10093 | `10093/data/NOMPC/` | 1,873 | |
| 10345 | `10345/data/Micrographs_18jam15a/` | 1,644 | |
| 10532 | `10532/data/02_Aligned_Micrographs/motioncorrected/` | 1,556 | keeps only `*_patch_aligned.mrc` |

Annotations come from the `cryoppp_lite` archives, which are unpacked keeping only
the `.star` files.

### From Hugging Face

`--intermediates` adds the artifacts that this project produced, so a stage can be
entered without re-deriving its input.

| Asset | Repository | Path under `$RAPICK_DATA` / `$RAPICK_WORK` |
| --- | --- | --- |
| θ₀, the repaired base checkpoint | [rikrikrik/recon-aware-pick-weights](https://huggingface.co/rikrikrik/recon-aware-pick-weights) | `$RAPICK_DATA/checkpoints/CryoTransformer_head_repaired.pth` |
| Triangular-blend contamination masks, all four entries at full-set scale (6,070 `.npz`) | [rikrikrik/recon-aware-pick-data](https://huggingface.co/datasets/rikrikrik/recon-aware-pick-data) | `$RAPICK_WORK/masks/<id>/` |
| Picks after contamination masking, all four entries | same | `$RAPICK_WORK/picks/<id>/mask.star` |
| Round-1 fine-tuned checkpoints, one per entry | [rikrikrik/recon-aware-pick-weights](https://huggingface.co/rikrikrik/recon-aware-pick-weights) | `$RAPICK_DATA/checkpoints/loop_fb_round1_empiar_<id>.pth` |

The round-1 checkpoints are what the `fb` condition picks with, so with them the
paper's headline row needs no loop run at all. Add `--with-loop-checkpoints` to
fetch them, or `--with-loop-checkpoints fb_gt` for the perfect-teacher arm of
Table 7's lower row. Each is about 870 MB.

| The four pickers' full-set candidates | same | `$RAPICK_WORK/picks/<id>/{baseline,cryolo,topaz,cryosegnet}.star` |

Add `--picks` to fetch the pickers' candidates. They are what Table 2 and Table S2
need, and having them means not installing crYOLO, Topaz or CryoSegNet
([BASELINES.md](BASELINES.md)). CryoTransformer's land as `baseline.star`, which is
the condition's name here.

## Verifying a download

**A downloaded `.mrc` can be silently corrupt in two ways, and neither is caught
by checking that the file exists.** Parallel downloaders appending to the same
`.part` file produce a truncated or oversized file, and EBI's S3 endpoint
sometimes returns a `ConnectionClosedException` XML body that lands inside the
`.mrc`, which then has the right size and fails hours later at Patch CTF.

```bash
python3 src/rapick/data/verify_mrc_integrity.py --dataset fullset --ids 10093 --full
```

A further trap: EMPIAR's older-format `.mrc` files lack the `MAP ` stamp at byte
offset 208, so a strict signature check reports valid files as broken. The
verifier accounts for this.

`import_particles` in CryoSPARC dies on the first missing micrograph, and a
`*.mrc` glob happily imports a partial download, so confirm the expected count
before starting a run. `scripts/01_download_data.sh` does this at the end, and
each dataset config carries the expected count.

## Licensing

Micrographs come from [EMPIAR](https://www.ebi.ac.uk/empiar/). Annotations come
from [CryoPPP](https://github.com/BioinfoMachineLearning/cryoppp), whose data is
CC-BY-4.0. The masks and STAR files we publish are derived from them and are
redistributed under CC-BY-4.0 with attribution.
