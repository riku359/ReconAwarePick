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


def check_star_distinctness(cfg, setting) -> CheckResult:
    """Every condition's star at this scale must be a distinct file. Two conditions
    sharing a star means one method is being compared against a copy of another (the
    exact '+mask == baseline' failure), so hash each and fail on any collision or
    missing file. Scoped to the setting being run, so an `annot` run is not blocked by
    full-deposition picks that have not been downloaded."""
    sources = cfg.dataset.sources(setting)
    by_hash: dict[str, list] = {}
    missing = []
    for name, src in sources.items():
        if not os.path.isfile(src.star):
            missing.append(name)
            continue
        by_hash.setdefault(mf.star_sha256(src.star), []).append(name)

    if missing:
        return CheckResult("star_distinctness", False, f"missing star for: {', '.join(sorted(missing))}")
    collisions = [names for names in by_hash.values() if len(names) > 1]
    if collisions:
        pairs = "; ".join(" == ".join(sorted(g)) for g in collisions)
        return CheckResult("star_distinctness", False, f"identical stars: {pairs}")
    return CheckResult("star_distinctness", True, f"{len(sources)} distinct stars")


def data_preflight(cfg, setting) -> list:
    """Filesystem-only input-integrity checks, enforced by `run` before any job is made."""
    return [check_micrographs(cfg, setting), check_star_distinctness(cfg, setting)]


def check_setup(api, cfg, setting) -> list:
    """Run all checks. Connection first (others need it), then data integrity."""
    results = [check_connection(api)]
    if results[0].ok:
        results.append(check_project_access(api, cfg.project_uid))
    results.extend(data_preflight(cfg, setting))
    return results
