#!/usr/bin/env python3
"""Fig. S6 and Fig. S7: the FSC curve and the viewing-direction distribution of every
reconstruction the paper reports.

CryoSPARC renders both plots for each refinement, so this only picks the right job,
keeps the last iteration, and shrinks the file. Nothing is drawn here.

The refine job of every panel is the one whose resolution the tables report, so a panel
and its table cell cannot come from different runs. The uids below are the authors'
instance and a fresh run produces different ones: read yours out of the `refine_job`
field of `results/tables/main_results.json` and `results/tables/ablation.json`, or
override one with `--job <entry>.<condition>=<uid>`.

NEEDS A LIVE CRYOSPARC INSTANCE, once, to fetch the panels. This script itself only
reads what `lib/cs_fetch_assets.py` left in the asset directories.

    python ../lib/cs_fetch_assets.py --project P1 --out /tmp/cs_full \\
        --spec 'J27=fsc_iteration,J27=viewing_direction_distribution_iteration'
    python build_recon_diagnostics.py --assets /tmp/cs_full /tmp/cs_annot
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import figure_paths                                    # noqa: E402

# entry -> condition -> refine job, best of the three seeds.
#   baseline / mask / select / both / fb   the rows of the ablation table
#   cryolo / topaz / cryosegnet            the other pickers of the main table
#   gt                                     a reconstruction from the CryoPPP annotations
#                                          of the 300 annotated micrographs, which no
#                                          table reports
# `baseline` is CryoTransformer on its own, so the picker figure reuses it for that
# column, and `fb` is the Ours row of both tables. Everything but `gt` runs on the full
# micrograph set and lives in one CryoSPARC project; `gt` is the 300-micrograph series
# in another, which is why the figures keep it below a dotted line.
JOBS = {
    "10081": dict(baseline="J27", mask="J47", select="J102", both="J134", fb="J507",
                  cryolo="J83", topaz="J164", cryosegnet="J195", gt="J63"),
    "10093": dict(baseline="J171", mask="J269", select="J311", both="J352", fb="J577",
                  cryolo="J365", topaz="J374", cryosegnet="J387", gt="J156"),
    "10345": dict(baseline="J114", mask="J165", select="J221", both="J274", fb="J511",
                  cryolo="J92", topaz="J224", cryosegnet="J283", gt="J208"),
    "10532": dict(baseline="J26", mask="J62", select="J180", both="J162", fb="J557",
                  cryolo="J279", topaz="J323", cryosegnet="J335", gt="J315"),
}

# (file stem CryoSPARC uses, name here, colours to keep). The plots are flat-coloured, so
# a palette holds them at a fraction of the size of the render.
KINDS = [("fsc", "fsc", 64), ("viewing_direction_distribution", "viewing", 256)]


def parse_job_overrides(items):
    """`["10081.fb=J9"]` -> `{("10081", "fb"): "J9"}`, checking the names."""
    out = {}
    for item in items or ():
        target, _, uid = item.partition("=")
        entry, _, condition = target.partition(".")
        if not uid or entry not in JOBS or condition not in JOBS[entry]:
            raise SystemExit(
                f"--job expects <entry>.<condition>=<uid> with entry in {sorted(JOBS)} "
                f"and condition in {sorted(JOBS['10081'])}, got: {item}")
        out[(entry, condition)] = uid
    return out


def newest(assets, job: str, stem: str) -> Path:
    """The panel of the job's last iteration."""
    pattern = re.compile(rf"{job}__{job}_{stem}_iteration_(\d+)\.png$")
    hits = [(int(pattern.search(p.name).group(1)), p)
            for p in assets if pattern.search(p.name)]
    if not hits:
        raise SystemExit(f"no {stem} panel for {job}; fetch it with "
                         f"results/figures/lib/cs_fetch_assets.py")
    return max(hits)[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", required=True, type=Path,
                    help="directories of fetched CryoSPARC assets")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the panels go (default $RAPICK_FIGURES_OUT/recon)")
    ap.add_argument("--job", action="append", default=[], metavar="ENTRY.CONDITION=UID",
                    help="override one refine job uid, e.g. 10081.fb=J9")
    args = ap.parse_args()

    overrides = parse_job_overrides(args.job)
    out_dir = args.out_dir or figure_paths.figures_out("recon")
    assets = [p for d in args.assets for p in Path(d).glob("*.png")]

    total = 0
    for entry, panels in JOBS.items():
        (out_dir / entry).mkdir(parents=True, exist_ok=True)
        for condition, job in panels.items():
            job = overrides.get((entry, condition), job)
            for stem, kind, colors in KINDS:
                im = Image.open(newest(assets, job, stem)).convert("RGB")
                dst = out_dir / entry / f"{kind}_{condition}.png"
                im.quantize(colors=colors, dither=Image.NONE).save(dst, optimize=True)
                total += dst.stat().st_size
        print("%s  %d panels" % (entry, 2 * len(panels)))
    print("%.1f MB in %s" % (total / 1024 / 1024, out_dir))


if __name__ == "__main__":
    main()
