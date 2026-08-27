# Fig. S5: the first selection cycle, and the class sheets of Fig. S1

**Fig. S5** is the 50 class averages of one entry's first classification, framed in the
colour of what the first CryoSift cycle does to each class and grouped by that fate: set
aside, re-classified in the next cycle, or discarded for good. Two entries are shown,
EMPIAR-10081 and 10345, both the `both` condition on the full micrograph set, and the
grouping is what makes the size of the discarded block the thing the eye lands on: 44 of
the 50 classes on EMPIAR-10345. The other two builders here draw the same kind of sheet
for every cycle, plus the short strips that ride the arrows, for **Fig. S1**, the
supplementary protocol figure, which is drawn in TikZ in the manuscript.

Every tile is an image CryoSPARC renders for one class inside the `select_2D` job that
reads the classification. Nothing here is redrawn, so the tiles have to be fetched first,
from a session that reaches the instance.

```bash
# 50 class tiles of one selection; cryosparc-tools lives in the recon environment.
# Repeat for J225 into /tmp/cls45, and for the eight protocol jobs into /tmp/tiles.
SPEC=$(python3 -c "print(','.join('J115=class2D_%d.png' % i for i in range(50)))")
envs/recon/.venv/bin/python results/figures/lib/cs_fetch_assets.py \
    --project P1 --spec "$SPEC" --out /tmp/cls81

# Fig. S5
envs/figures/.venv/bin/python results/figures/first_cycle/build_first_cycle_fig.py \
    --assets-10081 /tmp/cls81 --assets-10345 /tmp/cls45

# the eight sheets and ten carry strips of Fig. S1, from all eight selections at once
envs/figures/.venv/bin/python results/figures/first_cycle/build_protocol_cycles.py \
    --assets /tmp/tiles

# the two raw-particle stacks of Fig. S1
envs/figures/.venv/bin/python results/figures/first_cycle/build_teacher_strip.py \
    --assets /tmp/extract
```

`build_first_cycle_fig.py` takes `--assets-10081`, `--assets-10345` (both required),
`--job-10081` (default `J115`), `--job-10345` (default `J225`) and `--out-dir`.
`build_protocol_cycles.py` takes `--assets`, `--out-dir` and a repeatable
`--job <block>.<step>=<uid>`. `build_teacher_strip.py` takes `--assets`, `--out-dir`,
`--job-loop` and `--job-full`. `cs_fetch_assets.py` takes `--project`, `--spec`, `--out`
and `--env`.

| Reads | Writes | Needs |
| --- | --- | --- |
| `<assets>/<job>__class2D_<i>.png`, fifty per selection; `<assets>/<block>_extract_particles.jpg` for the two teacher stacks; the kept-class lists in the scripts, copied from `$RAPICK_WORK/select2d/<project>_<class2d>_iter/state.json` | `$RAPICK_FIGURES_OUT/selection/first_cycle_<id>.png`; `$RAPICK_FIGURES_OUT/protocol/<block>_{init,cycle1,cycle2,final}.png`, `<block>_carry_*.png` and `<block>_carry_teacher.png` | pillow; the fetch needs **a live CryoSPARC instance** and `cryosparc-tools` |

## Traps

- **Every job uid here is the authors' instance.** A fresh run produces the same job chain
  with different uids, which is why each one is a command-line argument with the paper's
  value as its default. Read yours out of your own project.
- **The kept-class lists are hardcoded next to the uids**, copied from the selector's own
  `state.json`. Nothing checks that a list belongs to the job it sits beside, so a uid
  overridden without its class list draws the right tiles under the wrong fates.
- **`--job-loop` and `--job-full` on `build_teacher_strip.py` only label the printed
  line.** That script reads `<block>_extract_particles.jpg`, named after the block, so
  overriding the uid does not change which file is cropped. Rename each fetched extraction
  render to that name in the assets directory.
- **The tiles that went into the paper are gone.** They were fetched into scratch space
  and not committed: they are large binaries, and this repository commits code and
  numbers. Re-running the conditions regenerates equivalent panels from jobs with
  different uids.
- The assets come out of CryoSPARC's GridFS rather than off disk, so `cryosparc-tools` is
  the only way to read them. Credentials come from the repository-root `.env`, the same as
  the rest of the repository; nothing prints one. Each asset lands as `<JOB>__<filename>`,
  which is the name every builder downstream looks for.
- A `--spec` filename may be given as a substring, and the first asset of that job whose
  name contains it is taken. Fifty tiles at a time is the usual case, so expand the list
  in the shell rather than typing it.
- The colours are shared with the pipeline figure of the main paper, so a kept class is
  the same green in both. `class_sheets.py` holds the palette, the border and the spacing,
  which is what lets Fig. S5 and the protocol sheets be read against each other.
- The teacher stacks are cropped from the middle row of CryoSPARC's 3x3 extraction render,
  whose particles are the most clearly centred, and resized to the class-tile size so the
  borders print at the same thickness.
