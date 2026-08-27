# Fig. 5: CryoSift class scores

Four stacked histograms, one per entry, of the CryoSift score of the 50 classes of that
entry's first 2D classification, with the absolute discard threshold of 4.5 drawn across
them. The point of the figure, in the failure-case analysis of Sec. 6, is that 4.5 is a
fixed number on a scale that shifts from entry to entry: it cuts the tail on three entries
and the bulk of the distribution on EMPIAR-10345, where 44 of the 50 classes fall above it
and 88.6% of the particles go with them. Each panel prints its own discarded count, and
the two ends of the scale are named under the ticks, particle at 1 and non-particle at 5.

```bash
envs/figures/.venv/bin/python results/figures/cryosift_scores/build_cryosift_scores.py
```

Two flags: `--csv` (default `results/tables/cryosift_class_scores.csv`) and `--out`
(default `$RAPICK_FIGURES_OUT/cryosift_scores.pdf`).

| Reads | Writes | Needs |
| --- | --- | --- |
| `results/tables/cryosift_class_scores.csv`, the rows with `stack == baseline` | `$RAPICK_FIGURES_OUT/cryosift_scores.pdf` | matplotlib |

## Traps

- **The committed table is read on purpose.** The CSV used to be read out of a sibling
  checkout of the research repository; it is committed here instead, so the figure and the
  numbers behind it cannot drift apart. An entry with no `baseline` rows raises rather
  than drawing an empty panel.
- **The file holds more than the figure draws.** Each entry also has a `cleaner_tri`
  stack, the same classification after contamination masking, and a set of common-line
  geometry columns from a study that did not enter the paper. The figure reads neither.
  [`results/tables/cryosift_class_scores.md`](../../tables/cryosift_class_scores.md)
  describes every column.
- **The bins have to reach past the largest score of any entry**, which is 5.692 on
  EMPIAR-10532. They run 1.0 to 5.75 in steps of 0.25. Widen them if a rerun scores
  higher: matplotlib drops the classes above the last edge and the panels quietly stop
  holding 50.
- One count scale is shared by all four panels, so a narrow distribution reads as narrow
  rather than as fewer classes. The top of the scale is the tallest bar plus 15%, which is
  the headroom the panel labels sit in.
- The colours are the same entry key the loop-rounds figure uses, so the two can be read
  against each other without re-learning it.
