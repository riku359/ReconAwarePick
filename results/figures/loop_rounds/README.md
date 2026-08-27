# Per-round loop diagnostics: the figure form of Table 6

Two panels against the feedback round: (a) the share of the particles reaching 2D
classification that the first CryoSift iteration rejects permanently, and (b) the share
that survives to the selection. One line per entry, four rounds, round 0 being the base
checkpoint. It is the plot that shows the loop is a one-round effect: round 1 moves the
fractions and rounds 2 and 3 stay where it left them. EMPIAR-10093 and 10532 move the way
the design predicts, EMPIAR-10081 moves the other way, EMPIAR-10345 does not move. The
manuscript carries this in its candidate supplement rather than in the main paper, so it
has no figure number.

```bash
envs/figures/.venv/bin/python results/figures/loop_rounds/build_loop_rounds.py
```

`--out` is the only flag, and it defaults to `$RAPICK_FIGURES_OUT/loop_rounds.pdf`.

| Reads | Writes | Needs |
| --- | --- | --- |
| `results/tables/loop_rounds.json`, the `not_in_paper` keys `permanent_discard_pct_unrounded` and `final_survival_pct_unrounded` | `$RAPICK_FIGURES_OUT/loop_rounds.pdf` | matplotlib |

## Traps

- **The committed table is read on purpose.** The per-round values used to be transcribed
  into the script so that the table and the plot could not drift apart; reading the one
  file keeps that and drops the duplication.
- **The two series live in `not_in_paper`, not in `values`.** Table 6 stopped printing
  these two rates when this plot was made, so they are numbers the table's own notes keep
  rather than numbers the paper prints. `tables.extra()` is what reaches them.
- **Watch the denominator.** Both rates divide by the particles that reach 2D
  classification, not by the round's raw picks. `after_purify_share` in the printed table
  divides by picks, so the two are not the same quantity under different names.
- The two series must be lists of one value per round and must be the same length for an
  entry; anything else raises, naming the entry and the two lengths.
- Round 0 picks with the base checkpoint, so a design that worked would pull panel (a)
  down and push panel (b) up as rounds accumulate. Reading the panels the other way round
  inverts the conclusion.
- The colours are the same entry key the CryoSift score figure uses.
