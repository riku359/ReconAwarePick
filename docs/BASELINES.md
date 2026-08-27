# Comparison pickers

crYOLO, Topaz and CryoSegNet are the three comparison pickers: their raw picks are
reconstructed with the same protocol as ours, and their 2D detection scores are
measured against the CryoPPP annotations.

**You probably do not need to install them.** Their picks are published as GT-aligned
STAR files, so a comparison run needs the picks alone:

```bash
bash scripts/01_download_data.sh --intermediates --picks
bash scripts/07_reconstruct.sh --entry 10081 --condition cryolo
```

Install them only to re-derive the picks.

## Training-data overlap

Table S2 greys the affected cells and excludes them from its ranking. A picker whose
released model saw an entry during training scores higher against that entry's
annotations, and the comparison would otherwise read as a quality difference.

| Picker | Overlap with the four entries |
| --- | --- |
| crYOLO | EMPIAR-10081 is in the training data of the released general model. |
| Topaz | The training data of the released general model is undocumented. The Topaz publication uses none of the four entries, but an overlap cannot be ruled out, so all four cells are treated as affected. |
| CryoSegNet | Trained on CryoPPP, the same data as CryoTransformer, and does not overlap the four entries. |
| CryoTransformer | These four entries are its held-out test set. |

Do not conflate this with two unrelated splits: MicrographCleaner's in-distribution
split of seven training entries includes 10081 but not 10532 — which is why the paper
attributes the mask's failure on 10532 to domain shift — and CryoSift's training set
contains none of the four entries and none of CryoTransformer's either.

Among the pickers with no known overlap, CryoTransformer has the highest recall on
every entry, which is what a pipeline whose purification stages can only discard needs.
That is why the paper builds on it.

## Installing them anyway

`scripts/00_setup.sh --baselines` clones each upstream at the commit `repos.lock.yaml`
pins; the environments build from the lockfiles in `envs/`. No upstream code is
redistributed here.

| Picker | Environment | Python | Framework | Licence |
| --- | --- | --- | --- | --- |
| Topaz | `envs/topaz` (uv) | 3.10 | torch 2.0.1+cu118 | GPL-3.0 |
| CryoSegNet | `envs/cryosegnet` (uv) | 3.8 | torch 2.0.1+cu118 | MIT |
| crYOLO | `envs/cryolo` (**conda**) | 3.8 | NVIDIA TensorFlow 1.15.5 | proprietary, non-commercial |

Three things that will otherwise cost you an afternoon:

- **crYOLO cannot be built with `uv`** — its sdist has a malformed `extras_require`.
  Use conda or Miniforge with a local `$HOME`; conda hangs on an NFS home. It installs
  from PyPI as `pip install 'cryolo[c11]'` and needs Python < 3.9 plus conda-only GUI
  packages. The GitHub repository is documentation only.
- **`scipy==1.9.1` in the CryoSegNet environment is load-bearing.** It imports
  `scipy.signal.gaussian`, removed in scipy 1.13. Do not bump it.
- CryoSegNet emits no confidence score. The 2D scorer matches by ascending distance
  rather than by score precisely so that every picker goes through the identical
  procedure.

## Making their output comparable

The pickers write divergent STAR flavours: different block names, different columns, Y
measured from the bottom. Everything downstream consumes one GT-aligned format instead:

- block `data_particles`
- columns `_rlnMicrographName`, `_rlnCoordinateX`, `_rlnCoordinateY`, and
  `_rlnAutopickFigureOfMerit` where the picker has a score
- integer coordinates at micrograph scale, Y flipped to a top-left origin via
  `round(H - y)` with `H` read from the `.mrc` header
- CTF and optics columns are never fabricated

```bash
python3 src/rapick/eval/convert_star_to_gt.py <input.star> --out-dir $RAPICK_WORK/picks/10081
```

The metric and the matching rule are in
[src/rapick/eval/README.md](../src/rapick/eval/README.md).
