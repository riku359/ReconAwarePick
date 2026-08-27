# Comparison pickers

crYOLO, Topaz and CryoSegNet appear in exactly two places: Table 2, where their
raw picks are reconstructed with the same protocol as ours, and Table S2, where
their 2D detection scores are measured against the CryoPPP annotations. Nothing
else in the pipeline uses them.

**You probably do not need to install them.** All three are awkward to build,
crYOLO is not redistributable, and Topaz is GPL-3.0. Their picks are published as
GT-aligned STAR files, so both tables can be reproduced from the picks alone:

```bash
bash scripts/01_download_data.sh --intermediates --picks
bash scripts/07_reconstruct.sh --entry 10081 --condition cryolo   # Table 2
bash scripts/08_tables_figures.sh --tables                        # Table S2
```

Install them only if you want to re-derive the picks.

## Training-data overlap

The paper greys the affected cells in Table S2 and excludes them from that
table's ranking. This is why: a picker whose released model saw one of these
entries during training scores higher against that entry's annotations than a
picker that did not, and the comparison would otherwise read as a quality
difference.

| Picker | Overlap with the four entries |
| --- | --- |
| crYOLO | EMPIAR-10081 is in the training data of the released general model. |
| Topaz | The training data of the released general model is undocumented. The Topaz publication uses none of the four entries, but an overlap cannot be ruled out, so all four cells are treated as affected. |
| CryoSegNet | Trained on CryoPPP, the same data as CryoTransformer, and does not overlap the four entries. |
| CryoTransformer | These four entries are its held-out test set. |

Two other overlap questions come up and are unrelated to this one. MicrographCleaner
has its own in-distribution split of seven training entries, which includes 10081
but not 10532; that is why the paper attributes the mask's failure on 10532 to
domain shift. CryoSift's training set contains none of the four entries and none
of CryoTransformer's training entries either. Do not conflate the three splits.

Among the pickers with no known overlap, CryoTransformer has the highest recall on
every entry, which is what a pipeline whose purification stages can only discard
needs. That is the reason the paper builds on it.

## Installing them anyway

Each has its own environment, built from the lockfiles in `envs/`. Upstream code
is cloned by `scripts/00_setup.sh --baselines` at the commits pinned in
`repos.lock.yaml`; none of it is redistributed here.

| Picker | Environment | Python | Framework | Licence |
| --- | --- | --- | --- | --- |
| Topaz | `envs/topaz` (uv) | 3.10 | torch 2.0.1+cu118 | GPL-3.0 |
| CryoSegNet | `envs/cryosegnet` (uv) | 3.8 | torch 2.0.1+cu118 | MIT |
| crYOLO | `envs/cryolo` (**conda**) | 3.8 | NVIDIA TensorFlow 1.15.5 | proprietary, non-commercial |

Three things that will otherwise cost you an afternoon:

- **crYOLO cannot be built with `uv`**: its sdist has a malformed `extras_require`.
  Use conda or Miniforge, and use a local `$HOME` — conda hangs on an NFS home. The
  GitHub repository is documentation only; the software installs from PyPI as
  `pip install 'cryolo[c11]'` and needs Python < 3.9, plus conda-only GUI packages.
- **`scipy==1.9.1` in the CryoSegNet environment is load-bearing.** It imports
  `scipy.signal.gaussian`, which was removed in scipy 1.13. Do not bump it.
- CryoSegNet emits no confidence score. The 2D scorer matches by ascending
  distance rather than by score precisely so that every picker, including this one,
  goes through the identical procedure.

## Making their output comparable

The pickers write divergent STAR flavours, with different block names, different
columns, and Y measured from the bottom. Everything downstream consumes one
GT-aligned format instead:

- block `data_particles`
- columns `_rlnMicrographName`, `_rlnCoordinateX`, `_rlnCoordinateY`, and
  `_rlnAutopickFigureOfMerit` where the picker has a score
- integer coordinates at micrograph scale, Y flipped to a top-left origin via
  `round(H - y)` with `H` read from the `.mrc` header
- CTF and optics columns are never fabricated

Convert any picker's native output with:

```bash
python3 src/rapick/eval/convert_star_to_gt.py <input.star> --out-dir $RAPICK_WORK/picks/10081
```

See [src/rapick/eval/README.md](../src/rapick/eval/README.md) for the metric and
the matching rule.
