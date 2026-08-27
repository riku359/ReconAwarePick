# Arms

A name says which stages the picks in it have been through, and the same name is used
everywhere: for the STAR under `$RAPICK_WORK/picks/<id>/`, for the `--name` a driver
records the run under, and for the output path
`$RAPICK_WORK/empiar_<id>/<setting>/<name>/`.

| Name | Picks from | Contamination mask | 2D class selection |
| --- | --- | :---: | :---: |
| `cryotransformer` | CryoTransformer, base checkpoint | | |
| `cryotransformer_mask` | same | yes | |
| `cryotransformer_select` | same | | yes |
| `cryotransformer_mask_select` | same | yes | yes |
| `fb` | CryoTransformer, round-1 checkpoint | yes | yes |

`fb` is the method: the picker re-picks with the checkpoint the feedback loop produced,
and its candidates go through the same two purification steps. By the rule above it
would be `fb_mask_select`; it keeps the short name because that is what the paper calls
the row. Its STARs do follow the rule — `fb.star`, then `fb_mask.star`.

Four more exist for the comparisons:

| Name | What it is |
| --- | --- |
| `cryolo`, `topaz`, `cryosegnet` | the other three pickers, raw picks, no purification |
| `cryosegnet_mask_select` | CryoSegNet's candidates through the same mask and 2D selection |
| `gt` | a reconstruction from the CryoPPP annotations of the 300 annotated micrographs |
| `fb_gt` | one round of the loop with the CryoPPP annotations as the teacher. A reimplementation: the scripts that produced this arm were never committed, and `--teacher gt` follows their documented procedure. |

`<setting>` is `annot` (the 300 CryoPPP-annotated micrographs, used by the loop and by
every 2D metric) or `full` (the whole deposition, used by every reconstruction-level
result).

## Where a name comes from

There is no list of arms in the code. A driver is handed a STAR and a `--name`, and an
arm is whatever comes out; `configs/recon.yaml` holds the parameters and is the same for
all of them, because the arms differ in which particles reach the chain and never in
what the chain does to them. So the table above is what the paper reports, not what the
repository permits — a STAR that no config names still runs:

```bash
bash scripts/2d_classification.sh --entry 10081 --star /some/other.star --name mine
bash scripts/reconstruct.sh       --entry 10081 --name mine
```

The names of the paper's own STARs are declared in `configs/datasets/empiar_<id>.yaml`
so that the preflight can check them and refuse a run in which two arms would import the
same file.

## What a resolution here means

Three caveats bound what a resolution produced by this repository means:

- Resolutions on EMPIAR-10345 follow CryoPPP's declared pixel size, which is the
  super-resolution movie value, so they are about half the physical figure and
  compare arms within that entry only.
- Resolution is best-of-three-seeds: the reconstruction runs three times with
  different random seeds and the best of the three by GSFSC 0.143 is reported.
- The 2D scores against the CryoPPP annotations are not held out: 50 of the 300
  annotated micrographs also train the picker in each round.

## If you are reading the private research repository

It names the same arms differently. Most of the correspondence is now readable, since
the names here were made to say the same thing:

| Here | In the research repo |
| --- | --- |
| `cryotransformer` | `cryotransformer` |
| `cryotransformer_mask` | `cryotransformer_clean_tri` |
| `cryotransformer_select` | `cryotransformer_cryosift_iter` |
| `cryotransformer_mask_select` | `cryotransformer_clean_tri_cryosift_iter` |
| `fb` | `fbf_r1_clean_tri_cryosift_iter`, loop arm `general_full`, prefix `fbf_r` |
| `fb_gt` | loop arm driven by the GT teacher, prefix `fbgt_r` |

Its loop arms `general` and `chained` fine-tune with LoRA and are **not** the paper's
method. The paper fine-tunes every weight except the first residual stage of the
backbone (`--finetune_mode head_decoder_encoder_resnet`).

## If you ran this repository before the rename

The names changed, and one of them changed meaning: `picks/<id>/fb.star` used to hold
the fb picks *after* contamination removal and now holds them before it. Every driver
refuses to run against a work directory written under the old names rather than read one
as the other; it prints the renames. They are:

| Old | New |
| --- | --- |
| `picks/<id>/baseline.star` | `picks/<id>/cryotransformer.star` |
| `picks/<id>/mask.star` | `picks/<id>/cryotransformer_mask.star` |
| `picks/<id>/fb.star` | `picks/<id>/fb_mask.star` |
| `picks/<id>/fb_raw.star` | `picks/<id>/fb.star` |
| `loop/<id>/round<n>/picks.star` | `loop/<id>/round<n>/cryotransformer.star` |
| `loop/<id>/round<n>/cryotransformer_clean_tri.star` | `loop/<id>/round<n>/cryotransformer_mask.star` |
| `loop/<id>/round<n>/survivors.star` | `loop/<id>/round<n>/cryotransformer_mask_select.star` |

Rename `fb.star` before `fb_raw.star`, or the second rename overwrites the first.
Directories under `empiar_<id>/<setting>/` keep whatever name they were written with;
nothing reads them by a fixed name.
