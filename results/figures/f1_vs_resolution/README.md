# Fig. 4: 2D F1 against resolution

One point per (picker, entry): the 2D macro F1 of the four base pickers on the 300
annotated micrographs, against the GSFSC 0.143 resolution the same stacks reach on the
full set. Sec. 6 of the paper reads it as the pickers reaching their highest F1 on
EMPIAR-10081, which is the one entry where the two rankings agree; on the other three the
2D score does not order the reconstructions. Colour is the entry, marker is the picker,
and the dashed line is the diagonal of the data range, a guide to the eye rather than a
fit.

```bash
envs/figures/.venv/bin/python results/figures/f1_vs_resolution/build_f1_vs_resolution.py
```

`--out` is the only flag, and it defaults to `$RAPICK_FIGURES_OUT/f1_vs_resolution.pdf`.

| Reads | Writes | Needs |
| --- | --- | --- |
| `results/tables/detection_2d.json` (Table S2, the `f1` field), `results/tables/main_results.json` (Table 2, the `published` field) | `$RAPICK_FIGURES_OUT/f1_vs_resolution.pdf` | matplotlib |

## Traps

- **The committed tables are read on purpose.** Both axes used to be literals transcribed
  into the script, so that the figure and the tables could not drift apart.
  [`results/tables/`](../../tables/README.md) now carries the same numbers together with
  their provenance, so the literals are gone and the property is kept. A missing file or a
  missing key raises with the path and the key in the message rather than plotting a hole.
- The pickers are keyed by the release condition names of
  [`docs/PAPER_TO_CODE.md`](../../../docs/PAPER_TO_CODE.md): the CryoTransformer row of
  the tables is the `baseline` condition.
- **Four of the sixteen points are values the paper greys out.** `detection_2d.json` marks
  crYOLO on EMPIAR-10081 and Topaz on all four entries with `possible_training_overlap`,
  because those scores may rest on entries inside a released general model's training
  data, and the paper excludes them from its ranking. The plot draws all sixteen points
  alike.
- **EMPIAR-10345 sits on its own resolution scale.** Its pixel size follows CryoPPP's
  declared 0.673 A, which is the super-resolution movie value, so its four points are
  about a factor of two off the rest on the shared y axis. The note in `main_results.json`
  says so; the figure carries no such marking.
- The axis limits (F1 0.22 to 0.86, resolution 3.2 to 7.6 A) and the two stacked legends
  are set for these sixteen points. The legends sit inside the axes because no point has
  F1 at or above 0.55 together with a resolution worse than 5.5 A; new numbers can land
  underneath them.
