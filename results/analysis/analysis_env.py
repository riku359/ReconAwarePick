"""Paths for the one-off analysis scripts in this directory.

These scripts produced numbers the manuscript quotes. They were written against one
machine and held absolute paths to it; here every path comes from the repository's own
contract instead (`docs/CONFIGURATION.md`), plus one variable of their own for the
CryoSPARC project directory, because several of them read a job's `.cs` and `job.json`
files straight off disk rather than through the API.

    RAPICK_DATA                     downloaded micrographs and annotations
    RAPICK_WORK                     everything the pipeline produced
    RAPICK_CRYOSPARC_PROJECT_DIR    the CryoSPARC project directory on disk, the
                                    parent of J1/, J2/, ...; `--project-dir` overrides

A missing variable raises with the variable named, rather than falling back to a
directory that happens to exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# results/analysis/analysis_env.py -> analysis -> results -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rapick.loop import paths as loop_paths            # noqa: E402

CORE_IDS = ("10081", "10093", "10345", "10532")

PROJECT_DIR_VAR = "RAPICK_CRYOSPARC_PROJECT_DIR"


def data_root() -> Path:
    """`$RAPICK_DATA`."""
    return loop_paths.data_root()


def work_root() -> Path:
    """`$RAPICK_WORK`."""
    return loop_paths.work_root()


def project_dir(explicit=None) -> Path:
    """The CryoSPARC project directory: `--project-dir`, else the environment variable.

    These scripts read `<project>/J<n>/*.cs` and `<project>/J<n>/job.json` directly, so
    they need the directory rather than a project uid. `cryosparcm cli
    "get_project('P1')"` prints it as `project_dir`.
    """
    value = str(explicit or os.environ.get(PROJECT_DIR_VAR, "")).strip()
    if not value:
        raise SystemExit(
            f"pass --project-dir, or set {PROJECT_DIR_VAR}: it must point at the "
            f"CryoSPARC project directory that holds J1/, J2/, ... See "
            f"docs/CONFIGURATION.md")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"{path} is not a directory")
    return path


def mask_dir(empiar: str) -> Path:
    """`$RAPICK_WORK/masks/<id>`: the stored triangular-blend masks, one npz per
    micrograph, key `tri`, float16 at model scale."""
    return loop_paths.mask_dir(str(empiar))


def gt_star(empiar: str) -> Path:
    """CryoPPP's annotation for one entry."""
    return loop_paths.gt_star(str(empiar))


def manifest_dir(empiar: str, setting: str, condition: str) -> Path:
    """`$RAPICK_WORK/empiar_<id>/<setting>/<condition>`: one reconstruction arm."""
    return loop_paths.manifest_dir(str(empiar), setting, condition)


def out_path(name: str, explicit=None) -> Path:
    """Where a result lands: `--out`, else `$RAPICK_WORK/analysis/<name>`.

    The committed copies of these outputs are under `results/tables/revision/`; a rerun
    writes next to the pipeline's other outputs rather than back into the repository.
    """
    if explicit:
        return Path(explicit).expanduser()
    path = work_root() / "analysis" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
