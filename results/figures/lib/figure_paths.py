"""Paths, credentials and tool locations shared by every figure builder.

`docs/CONFIGURATION.md` defines the repository's five environment variables and one
repository-root `.env`, and this module is the only place the figure code reads them.
Nothing falls back to a directory that happens to exist: a variable that cannot be
resolved raises `SystemExit` naming the variable, because a figure silently drawn from
the wrong tree looks exactly like a figure drawn from the right one.

    RAPICK_DATA         micrographs, annotations, pretrained weights (read-mostly)
    RAPICK_WORK         everything the pipeline produces, and where figures are written
    RAPICK_THIRD_PARTY  upstream checkouts             (default <repo>/third_party)
    RAPICK_GPU          default GPU index
    <repo>/.env         CryoSPARC credentials

Two more variables belong to the figures alone:

    RAPICK_FIGURES_OUT  where built figures go        (default $RAPICK_WORK/figures)
    RAPICK_CHIMERAX     the ChimeraX executable       (see chimerax_command below)
"""
from __future__ import annotations

import os
from pathlib import Path

# results/figures/lib/figure_paths.py -> lib -> figures -> results -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]

FIGURES_ROOT = REPO_ROOT / "results" / "figures"
TABLES_ROOT = REPO_ROOT / "results" / "tables"

DOC = "docs/CONFIGURATION.md"


def require_env(name: str, what: str) -> str:
    """Read an environment variable, naming it when it is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"environment variable {name} is not set ({what}); see {DOC}")
    return value


def data_root() -> Path:
    """`$RAPICK_DATA`: downloaded micrographs, annotations and pretrained weights."""
    return Path(require_env(
        "RAPICK_DATA",
        "downloaded inputs: micrographs, annotations, pretrained weights")).expanduser()


def work_root() -> Path:
    """`$RAPICK_WORK`: everything the pipeline produces."""
    return Path(require_env(
        "RAPICK_WORK", "where the pipeline writes its outputs")).expanduser()


def third_party_root() -> Path:
    """`$RAPICK_THIRD_PARTY`, defaulting to the in-repo checkout directory."""
    return Path(os.environ.get("RAPICK_THIRD_PARTY")
                or (REPO_ROOT / "third_party")).expanduser()


def gpu(override=None) -> str:
    """The GPU index to pin. An explicit argument wins, then `$RAPICK_GPU`."""
    if override not in (None, ""):
        return str(override)
    return require_env("RAPICK_GPU", "default GPU index; pass --gpu to override")


def figures_out(subdir: str = "") -> Path:
    """Where a built figure goes: `$RAPICK_FIGURES_OUT`, else `$RAPICK_WORK/figures`.

    Figures are outputs, so they are not written back into the repository: a rebuild
    that overwrote a committed PDF would make "the figure in the paper" and "the figure
    this checkout produces" indistinguishable without a diff.
    """
    root = os.environ.get("RAPICK_FIGURES_OUT", "").strip()
    base = Path(root).expanduser() if root else work_root() / "figures"
    return base / subdir if subdir else base


def src_root() -> Path:
    """`<repo>/src`, so that `rapick.*` is importable from a plain interpreter."""
    return REPO_ROOT / "src"


def add_src_to_path() -> None:
    """Make `import rapick.<...>` work without installing the package."""
    import sys

    path = str(src_root())
    if path not in sys.path:
        sys.path.insert(0, path)


# --- the repository-root .env -------------------------------------------

def env_file(explicit=None) -> Path:
    """The repository-root `.env` (git-ignored; `.env.example` is the template)."""
    return Path(explicit).expanduser() if explicit else REPO_ROOT / ".env"


def read_env(explicit=None) -> dict:
    """Parse `<repo>/.env` into KEY -> value, then layer the process environment on top.

    The parsed values are never printed by anything in this directory: they hold a
    licence id, an e-mail address and a password.
    """
    path = env_file(explicit)
    values = {}
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    elif explicit is not None:
        raise SystemExit(f"{path} not found")
    values.update(os.environ)
    return values


def require_credential(env: dict, key: str, path=None) -> str:
    """Fetch a `.env` key, naming the key but never echoing the value."""
    value = (env.get(key) or "").strip()
    if not value:
        raise SystemExit(
            f"{key} is not set in {env_file(path)}; copy .env.example to the "
            f"repository root as .env and fill it in -- see {DOC}")
    return value


def connect_cryosparc(env_path=None):
    """Open a CryoSPARC session from the credentials in `.env`.

    Every figure script that reaches CryoSPARC comes through here, so there is one
    place credentials are read and none of them takes a password on the command line.
    """
    from cryosparc.tools import CryoSPARC

    env = read_env(env_path)
    cs = CryoSPARC(
        license=require_credential(env, "CRYOSPARC_LICENSE_ID", env_path),
        email=require_credential(env, "CRYOSPARC_EMAIL", env_path),
        password=require_credential(env, "CRYOSPARC_PASSWORD", env_path),
        host=require_credential(env, "CRYOSPARC_HOST", env_path),
        base_port=int(require_credential(env, "CRYOSPARC_PORT", env_path)),
    )
    if not cs.test_connection():
        raise SystemExit("could not reach CryoSPARC with the credentials in "
                         f"{env_file(env_path)}")
    return cs


def cryosparc_project(override=None, env_path=None) -> str:
    """The CryoSPARC project uid to read from: `--project`, else `CRYOSPARC_PROJECT`."""
    if override:
        return str(override)
    return require_credential(read_env(env_path), "CRYOSPARC_PROJECT", env_path)


# --- ChimeraX ------------------------------------------------------------

# The macOS bundle installs under /Applications with the version in the directory name,
# so the default is a glob rather than one path. The bundle has no OSMesa, so it cannot
# render offscreen and must run windowed; windows flash up and close by themselves for
# every panel. On Linux use the headless wrapper next to the renderer, which supplies
# OSMesa (results/figures/locres_maps/chimerax_headless.sh).
MACOS_CHIMERAX_GLOB = "/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX"


def chimerax_command(override=None) -> str:
    """The ChimeraX executable: `--chimerax`, then `$RAPICK_CHIMERAX`, then the bundle.

    Returns a path; it is not checked for executability here, because the wrapper form
    used on Linux is a shell script that may live anywhere.
    """
    if override:
        return str(override)
    from_env = os.environ.get("RAPICK_CHIMERAX", "").strip()
    if from_env:
        return from_env
    import glob

    found = sorted(glob.glob(MACOS_CHIMERAX_GLOB))
    if found:
        return found[-1]
    raise SystemExit(
        "no ChimeraX found: pass --chimerax, set RAPICK_CHIMERAX, or install the "
        "macOS bundle. On Linux point either at "
        "results/figures/locres_maps/chimerax_headless.sh, which supplies OSMesa.")
