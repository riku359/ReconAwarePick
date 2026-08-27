# Conditions

The five conditions, named the same everywhere in this repository: in
`configs/conditions/`, in every driver's `--condition` flag, and in the output path
`$RAPICK_WORK/empiar_<id>/<setting>/<condition>/`.

| Condition | Picks from | Contamination mask | 2D class selection |
| --- | --- | :---: | :---: |
| `baseline` | CryoTransformer, base checkpoint | | |
| `mask` | same | yes | |
| `select` | same | | yes |
| `both` | same | yes | yes |
| `fb` | CryoTransformer, round-1 checkpoint | yes | yes |

`fb` is the method: the picker re-picks with the checkpoint the feedback loop produced,
and its candidates go through the same two purification steps.

Four more exist for the comparisons:

| Condition | What it is |
| --- | --- |
| `cryolo`, `topaz`, `cryosegnet` | the other three pickers, raw picks, no purification |
| `cryosegnet_both` | CryoSegNet's candidates through the same mask and 2D selection |
| `gt` | a reconstruction from the CryoPPP annotations of the 300 annotated micrographs |
| `fb_gt` | one round of the loop with the CryoPPP annotations as the teacher. A reimplementation: the scripts that produced this arm were never committed, and `--teacher gt` follows their documented procedure. |

`<setting>` is `annot` (the 300 CryoPPP-annotated micrographs, used by the loop and by
every 2D metric) or `full` (the whole deposition, used by every reconstruction-level
result).

## If you are reading the private research repository

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
