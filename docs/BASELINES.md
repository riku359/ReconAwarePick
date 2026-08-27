# Comparison pickers

crYOLO, Topaz and CryoSegNet are the three comparison pickers: their raw picks are
reconstructed with the same protocol as ours, and their 2D detection scores are
measured against the CryoPPP annotations.

**You probably do not need to install them.** Their picks are published as GT-aligned
STAR files, so a comparison run needs the picks alone:

```bash
bash scripts/download.sh   # scripts/download/09_picker_candidates.sh is the picks
bash scripts/2d_classification.sh --entry 10081 --star $RAPICK_WORK/picks/10081/cryolo.star
bash scripts/reconstruct.sh       --entry 10081 --name cryolo
```

## Installing them anyway

`scripts/setup.sh --baselines` clones each upstream at the commit `repos.lock.yaml`
pins; the environments build from the lockfiles in `envs/`. No upstream code is
redistributed here.

| Picker | Environment | Python | Framework | Licence |
| --- | --- | --- | --- | --- |
| Topaz | `envs/topaz` (uv) | 3.10 | torch 2.0.1+cu118 | GPL-3.0 |
| CryoSegNet | `envs/cryosegnet` (uv) | 3.8 | torch 2.0.1+cu118 | MIT |
| crYOLO | `envs/cryolo` (**conda**) | 3.8 | NVIDIA TensorFlow 1.15.5 | proprietary, non-commercial |

Two things that will otherwise cost you an afternoon:

- **crYOLO cannot be built with `uv`** — its sdist has a malformed `extras_require`.
  Use conda or Miniforge with a local `$HOME`; conda hangs on an NFS home. It installs
  from PyPI as `pip install 'cryolo[c11]'` and needs Python < 3.9 plus conda-only GUI
  packages. The GitHub repository is documentation only.
- **`scipy==1.9.1` in the CryoSegNet environment is load-bearing.** It imports
  `scipy.signal.gaussian`, removed in scipy 1.13. Do not bump it.

Each picker is then run through its own upstream CLI. Convert whatever it writes into
the one GT-aligned format everything downstream consumes:

```bash
python3 src/rapick/eval/convert_star_to_gt.py <input.star> \
    --out-dir $RAPICK_WORK/picks/10081 --empiar 10081
```

That format, the metric and the matching rule are in
[src/rapick/eval/README.md](../src/rapick/eval/README.md).
