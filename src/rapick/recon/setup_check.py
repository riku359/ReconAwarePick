"""check-setup: preflight before any real run. Fails fast and loud.

Two groups of checks:
- environment (need a live server): connection, project access.
- data integrity (pure filesystem): micrograph health, star distinctness. These are
  what `run` enforces before spending GPU-hours (cli._cmd_run), because the failures
  they catch — a truncated micrograph, or a picker's star that is byte-identical to
  another method's — otherwise run to completion and quietly corrupt the benchmark.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from . import coords
from . import manifest as mf


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_connection(api) -> CheckResult:
    try:
        api.connect()
        return CheckResult("connection", True, f"{api.conn.host}:{api.conn.port}")
    except Exception as exc:
        return CheckResult("connection", False, str(exc))


def check_project_access(api, project_uid) -> CheckResult:
    if not project_uid:
        return CheckResult("project", False,
                           "no project UID — set CRYOSPARC_PROJECT in .env")
    try:
        api.find_project(project_uid)
        return CheckResult("project", True, project_uid)
    except Exception as exc:
        return CheckResult("project", False, f"{project_uid}: {exc}")


def check_micrographs(cfg, setting) -> CheckResult:
    """Fail if the glob matches no files, if any matched file is a truncated/corrupt
    mrc, or if the healthy count is below the dataset's declared minimum."""
    pattern = cfg.dataset.micrograph_glob(setting)
    files = sorted(glob.glob(pattern))
    if not files:
        return CheckResult("micrographs", False, f"no files match {pattern}")

    broken = [(os.path.basename(f), coords.micrograph_defect(f)) for f in files]
    broken = [(name, reason) for name, reason in broken if reason]
    healthy = len(files) - len(broken)
    expected = cfg.dataset.expected_count(setting)

    ok = not broken and (expected is None or healthy >= expected)
    detail = f"{len(files)} files, {healthy} healthy"
    if expected is not None:
        detail += f", expected >= {expected}"
    if broken:
        shown = ", ".join(name for name, _ in broken[:5])
        more = "..." if len(broken) > 5 else ""
        detail += f"; {len(broken)} BROKEN ({shown}{more})"
    return CheckResult("micrographs", ok, detail)


def check_star_distinctness(cfg, setting, source=None) -> CheckResult:
    """Every condition's star at this scale must be a distinct file. Two conditions
    sharing a star means one method is being compared against a copy of another (the
    exact '+mask == baseline' failure), so hash each and fail on any collision.

    Scoped twice over. To the setting being run, so an `annot` run is not blocked by
    full-deposition picks that have not been downloaded. And to the arms that exist:
    a dataset config names every arm of the paper, but `fb_mask` is written by the
    feedback loop and `fb_gt_mask` by the perfect-teacher arm, so demanding all of
    them would refuse the +mask run that is supposed to come first. An arm that has
    not been built is named in the detail and skipped; `source`, the one arm this run
    is about to import, is still required.
    """
    sources = cfg.dataset.sources(setting)
    by_hash: dict[str, list] = {}
    missing = []
    for name, src in sources.items():
        if not os.path.isfile(src.star):
            missing.append(name)
            continue
        by_hash.setdefault(mf.star_sha256(src.star), []).append(name)

    if source and source in missing:
        return CheckResult("star_distinctness", False,
                           f"no star for the arm being run, {source!r}: "
                           f"{sources[source].star}")
    if not by_hash:
        return CheckResult("star_distinctness", False,
                           f"none of this scale's picks exist yet: "
                           f"{', '.join(sorted(missing))}. Run scripts/download.sh.")
    collisions = [names for names in by_hash.values() if len(names) > 1]
    if collisions:
        pairs = "; ".join(" == ".join(sorted(g)) for g in collisions)
        return CheckResult("star_distinctness", False, f"identical stars: {pairs}")
    present = sum(len(names) for names in by_hash.values())
    detail = f"{present} distinct stars"
    if missing:
        detail += f" (not built yet, not checked: {', '.join(sorted(missing))})"
    return CheckResult("star_distinctness", True, detail)


def data_preflight(cfg, setting, source=None) -> list:
    """Filesystem-only input-integrity checks, enforced by `run` before any job is made.

    `source` is the arm about to run; its star is required where the other arms' are
    only checked if they happen to exist.
    """
    return [check_micrographs(cfg, setting),
            check_star_distinctness(cfg, setting, source)]


def check_setup(api, cfg, setting, source=None) -> list:
    """Run all checks. Connection first (others need it), then data integrity."""
    results = [check_connection(api)]
    if results[0].ok:
        results.append(check_project_access(api, cfg.project_uid))
    results.extend(data_preflight(cfg, setting, source))
    return results
