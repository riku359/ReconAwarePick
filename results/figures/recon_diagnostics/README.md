# Fig. S6 and Fig. S7: FSC and viewing directions

The gold-standard FSC curve (Fig. S6) and the viewing-direction distribution (Fig. S7) of
every reconstruction the paper reports: four entries by nine conditions, laid out in the
manuscript as one column per entry. CryoSPARC renders both plots inside each refinement
job, so nothing is drawn here. This picks the right job, keeps the last iteration and
shrinks the file. The panels have to be fetched first, from a session that reaches the
instance.

```bash
# cryosparc-tools lives in the recon environment; repeat per job, or list several jobs
envs/recon/.venv/bin/python results/figures/lib/cs_fetch_assets.py \
    --project P1 --out /tmp/cs_full \
    --spec 'J27=fsc_iteration,J27=viewing_direction_distribution_iteration'

envs/figures/.venv/bin/python results/figures/recon_diagnostics/build_recon_diagnostics.py \
    --assets /tmp/cs_full /tmp/cs_annot
```

`build_recon_diagnostics.py` takes `--assets` (required, one or more directories),
`--out-dir` (default `$RAPICK_FIGURES_OUT/recon`) and a repeatable
`--job <entry>.<condition>=<uid>`. `cs_fetch_assets.py` takes `--project`, `--spec`,
`--out` and `--env`.

| Reads | Writes | Needs |
| --- | --- | --- |
| `<assets>/<JOB>__<JOB>_fsc_iteration_<n>.png` and `<JOB>__<JOB>_viewing_direction_distribution_iteration_<n>.png`, for the 36 refine jobs named in the script | `$RAPICK_FIGURES_OUT/recon/<entry>/fsc_<condition>.png` and `viewing_<condition>.png`, 72 panels | pillow; the fetch needs **a live CryoSPARC instance** and `cryosparc-tools` |

## Traps

- **Every job uid in the script is the authors' instance**, and a fresh run produces the
  same chain with different uids. Read yours out of the `refine_job` field of
  `results/tables/main_results.json` and `results/tables/ablation.json`, or override one
  with `--job 10081.fb=J9`. The refine job of every panel is the one whose resolution the
  tables report, so a panel and its table cell cannot come from different runs. Panels
  taken from any other job of the same condition are a different run: an earlier picker
  run reaches other resolutions than the ones the paper prints, and its plots look exactly
  as plausible.
- **`gt` is the exception no table can check.** It is a reconstruction from the CryoPPP
  annotations of the 300 annotated micrographs, which no table reports, and it lives in
  another CryoSPARC project from the full-set conditions. That is why `--assets` takes
  several directories, and why the manuscript keeps that row below a dotted line.
- **The panel is found by name, not by asset id.** The pattern is exactly what
  `cs_fetch_assets.py` writes, `<JOB>__<JOB>_<stem>_iteration_<n>.png`, and the highest
  `<n>` wins. Renaming a fetched file, or fetching it any other way, makes the job look
  like it has no panel; the script then stops naming the job and the missing stem.
- The saved panels are quantized, to 64 colours for the FSC plots and 256 for the viewing
  directions. Both are flat-coloured plots, so the palette holds them at a fraction of the
  size of CryoSPARC's render, and the script prints the total it wrote in MB.
- The condition names are the release vocabulary of
  [`docs/PAPER_TO_CODE.md`](../../../docs/PAPER_TO_CODE.md): `baseline` is CryoTransformer
  on its own and `fb` is the Ours row, which is why an override is written as
  `<entry>.<condition>=<uid>` and is rejected when either name is not one of them.
