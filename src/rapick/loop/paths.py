"""Where everything lives: the loop's half of the repository's path contract.

`docs/CONFIGURATION.md` defines five environment variables and one repository-root
`.env`, and this module is the only place in `rapick.loop` that reads them. Nothing
here falls back to a path that happens to exist on somebody's server: a missing
variable raises `ConfigError` naming the variable, because a loop that silently wrote
its teacher labels somewhere else would produce a plausible-looking result built on
the wrong inputs.

    RAPICK_DATA         micrographs, annotations, pretrained weights (read-mostly)
    RAPICK_WORK         everything the pipeline produces, this loop included
    RAPICK_THIRD_PARTY  upstream checkouts            (default <repo>/third_party)
    RAPICK_ENVS         the per-tool virtual environments  (default <repo>)
    RAPICK_GPU          default GPU index, always overridable with --gpu
    <repo>/.env         CryoSPARC credentials, CRYOSPARC_WORKER, CRYOSPARC_PROJECT

The loop is a conductor: every round shells out to a tool that lives outside this
package -- the picker, the contamination filter, the 2D class selection. Each is
resolved through `tool_cmd()` below, from `RAPICK_TOOL_<NAME>`, so that a checkout laid
out differently is one variable rather than an edit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# src/rapick/loop/paths.py -> src/rapick/loop -> src/rapick -> src -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]

DOC = "docs/CONFIGURATION.md"


class ConfigError(RuntimeError):
    """A path or credential the caller has to supply is missing or unusable."""


def _require_env(name: str, what: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set ({what}); see {DOC}")
    return value


# --- the five roots ------------------------------------------------------

def data_root() -> Path:
    """$RAPICK_DATA: downloaded micrographs, annotations and pretrained weights."""
    return Path(_require_env("RAPICK_DATA", "downloaded inputs: micrographs, "
                                            "annotations, pretrained weights"))


def work_root() -> Path:
    """$RAPICK_WORK: everything the pipeline produces, this loop's state included."""
    return Path(_require_env("RAPICK_WORK", "where the pipeline writes its outputs"))


def third_party_root() -> Path:
    """$RAPICK_THIRD_PARTY, defaulting to the in-repo checkout dir."""
    return Path(os.environ.get("RAPICK_THIRD_PARTY") or (REPO_ROOT / "third_party"))


def envs_root() -> Path:
    """$RAPICK_ENVS, defaulting to the repository root (one .venv per env dir)."""
    return Path(os.environ.get("RAPICK_ENVS") or REPO_ROOT)


def gpu(override: Optional[str] = None) -> str:
    """The GPU index to pin. --gpu wins; otherwise $RAPICK_GPU, which must be set.

    Guessing 0 would be worse than stopping: on a shared host card 0 is usually the
    one somebody else is already computing on, and a fine-tune that OOMs there costs
    the whole round back to its `pick` step.
    """
    if override not in (None, ""):
        return str(override)
    return _require_env("RAPICK_GPU", "default GPU index; pass --gpu to override")


# --- the repository-root .env -------------------------------------------

def env_file() -> Path:
    """The repository-root `.env` (git-ignored; `.env.example` is the template)."""
    return REPO_ROOT / ".env"


def load_env() -> dict:
    """Parse `<repo>/.env`, then layer the process environment on top.

    Same KEY=VALUE rule and same precedence as `rapick.recon.config.load_env`, kept
    here so that the reporting tools (`status`, `round_metrics`) stay importable with
    nothing but the standard library.
    """
    values: dict[str, str] = {}
    path = env_file()
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def _require_cryosparc(key: str, what: str, override: Optional[str] = None) -> str:
    if override not in (None, ""):
        return str(override)
    value = load_env().get(key, "").strip()
    if not value:
        raise ConfigError(f"{key} is not set ({what}); put it in {env_file()} "
                          f"or export it -- see {DOC}")
    return value


def cryosparc_worker(override: Optional[str] = None) -> str:
    """The worker/lane name CryoSPARC reports for this site (`CRYOSPARC_WORKER`)."""
    return _require_cryosparc(
        "CRYOSPARC_WORKER", "the CryoSPARC worker lane every GPU job is pinned to; "
                            "`cryosparcm cli \"get_scheduler_targets()\"` lists it",
        override)


def cryosparc_project(override: Optional[str] = None) -> str:
    """The project uid the pipeline writes into (`CRYOSPARC_PROJECT`, e.g. P1)."""
    return _require_cryosparc(
        "CRYOSPARC_PROJECT", "the CryoSPARC project uid the pipeline writes into, "
                             "created once in the web interface",
        override)


# --- inputs under $RAPICK_DATA ------------------------------------------

def base_checkpoint() -> Path:
    """theta_0: the one checkpoint every round fine-tunes from (Eq. 1, Sec. S2)."""
    path = data_root() / "checkpoints" / "CryoTransformer_head_repaired.pth"
    if not path.is_file():
        raise ConfigError(f"theta_0 checkpoint missing: {path} "
                          f"(scripts/download.sh fetches it)")
    return path


def annotated_micrographs(empiar: str) -> Path:
    """The 300 annotated micrographs of one entry."""
    return data_root() / "cryoppp" / empiar / "micrographs"


def gt_star(empiar: str) -> Path:
    """CryoPPP's annotation for one entry, in the GT-aligned convention."""
    return (data_root() / "cryoppp" / empiar / "ground_truth" /
            f"empiar-{empiar}_particles_selected.star")


def fullset_micrographs(empiar: str) -> Path:
    """The whole deposition of one entry (997-1,873 micrographs)."""
    return data_root() / "cryoppp_fullset" / empiar / "micrographs"


# --- outputs under $RAPICK_WORK -----------------------------------------

def mask_dir(empiar: str) -> Path:
    """Stored triangular-blend contamination masks, one .npz per micrograph.

    Written once per entry by the contamination stage; they depend on the micrograph
    and not on the picks, so every round and every checkpoint reads the same files
    instead of re-running MicrographCleaner.
    """
    return work_root() / "masks" / empiar


def select2d_root() -> Path:
    """Where the iterative 2D selection keeps its per-cycle state and scores."""
    return work_root() / "select2d"


def manifest_dir(empiar: str, setting: str, condition: str) -> Path:
    """$RAPICK_WORK/empiar_<id>/<setting>/<condition>: one reconstruction arm."""
    return work_root() / f"empiar_{empiar}" / setting / condition


def picks_star(empiar: str, condition: str) -> Path:
    """$RAPICK_WORK/picks/<id>/<condition>.star: the picks one condition reconstructs.

    This is where the loop hands off. The dataset configs name these files, so writing
    one here is what makes a checkpoint's picks reconstructable by `rapick.recon`.
    """
    return work_root() / "picks" / empiar / f"{condition}.star"


# --- the tools this package drives --------------------------------------

@dataclass(frozen=True)
class Tool:
    """One external script the loop shells out to."""

    env_var: str                 # RAPICK_TOOL_<NAME>, always wins
    venv: Optional[str]          # which env dir under $RAPICK_ENVS/envs runs it
    root: str                    # "third_party" or "repo": what `default` is relative to
    default: str                 # where it normally sits
    what: str                    # what it is, for the error message
    module: Optional[str] = None      # run as `-m <module>`: it uses relative imports
    pythonpath: Optional[str] = None  # "src" or "eval": what it has to be able to import


# Two roots. `third_party` is the upstream picker checkout that `scripts/setup.sh`
# clones and copies src/rapick/picker/overlay/ over -- so `predict.py` and `finetune.py`
# are found there, not in this repository. `repo` is a sibling stage of this repository
# that this package drives but does not own.
TOOLS = {
    "predict": Tool(
        "RAPICK_TOOL_PREDICT", "cryotransformer", "third_party",
        "cryotransformer/predict.py",
        "the picker's inference entry point",
        pythonpath="eval"),
    "predict_fullset": Tool(
        "RAPICK_TOOL_PREDICT_FULLSET", "cryotransformer", "third_party",
        "cryotransformer/predict_fullset.py",
        "the picker's inference entry point for a whole deposition (predict.py "
        "without the diagnostic machinery)",
        pythonpath="eval"),
    "finetune": Tool(
        "RAPICK_TOOL_FINETUNE", "cryotransformer", "third_party",
        "cryotransformer/finetune.py",
        "the picker's fine-tuning entry point",
        pythonpath="eval"),
    "scorer": Tool(
        # Standard library only, so it runs under whichever interpreter runs the loop.
        "RAPICK_TOOL_SCORER", None, "repo",
        "src/rapick/eval/calc_common_2d_metrics.py",
        "the cross-picker 2D scorer: GT-aligned STAR in, macro P/R/F1 out"),
    "mask_filter": Tool(
        "RAPICK_TOOL_MASK_FILTER", "micrograph_cleaner", "repo",
        "src/rapick/cleaner/filter_star_from_masks.py",
        "the contamination filter that applies the stored masks to a STAR"),
    "select_2d": Tool(
        "RAPICK_TOOL_SELECT_2D", "cryosift", "repo",
        "src/rapick/select2d/iterate_class2d.py",
        "the iterative 2D class selection",
        module="rapick.select2d.iterate_class2d", pythonpath="src"),
}

# What each `pythonpath` value resolves to. The picker's own scripts import the STAR
# reader from src/rapick/eval and look for it on PYTHONPATH first -- their in-repository
# fallback stops resolving once the overlay has been copied over the upstream clone.
PYTHONPATHS = {"src": lambda: REPO_ROOT / "src",
               "eval": lambda: REPO_ROOT / "src" / "rapick" / "eval"}


def venv_python(name: str) -> Path:
    """$RAPICK_ENVS/envs/<name>/.venv/bin/python, checked."""
    path = envs_root() / "envs" / name / ".venv" / "bin" / "python"
    if not path.is_file():
        raise ConfigError(f"no interpreter for the {name!r} environment at {path}; "
                          f"build it with scripts/setup.sh, or set RAPICK_ENVS")
    return path


def tool_script(name: str) -> Path:
    """The script `name` resolves to, checked for existence."""
    spec = TOOLS[name]
    override = os.environ.get(spec.env_var, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise ConfigError(f"{spec.env_var}={override} is not a file "
                              f"({spec.what})")
        return path
    root = third_party_root() if spec.root == "third_party" else REPO_ROOT
    path = root / spec.default
    if not path.is_file():
        raise ConfigError(f"{spec.what} not found at {path}; run scripts/setup.sh, "
                          f"or set {spec.env_var} to where it lives")
    return path


def _tool_override(name: str) -> str:
    return os.environ.get(TOOLS[name].env_var, "").strip()


def tool_cmd(name: str) -> list:
    """[interpreter, script-or-module] for `name`, ready to have arguments appended.

    A tool whose `module` is set is invoked as `-m`: its own package uses relative
    imports, and running it by file path would fail on the first one. An explicit
    `RAPICK_TOOL_*` override is always run by path, so a tool pointed somewhere else
    has to be runnable that way.
    """
    import sys

    spec = TOOLS[name]
    interpreter = venv_python(spec.venv) if spec.venv else Path(sys.executable)
    if spec.module and not _tool_override(name):
        tool_script(name)                      # existence check, for a clear error
        return [interpreter, "-m", spec.module]
    return [interpreter, tool_script(name)]


def tool_cwd(name: str) -> Path:
    """Where to run `name` from.

    A script invoked by path imports its own checkout's modules by relative path, so
    the working directory is part of the call and not a detail. A module invocation
    resolves through PYTHONPATH instead and runs from the repository root.
    """
    if TOOLS[name].module and not _tool_override(name):
        return REPO_ROOT
    return tool_script(name).parent


def tool_env(name: str) -> dict:
    """Environment additions `name` needs, prepended to any inherited PYTHONPATH."""
    spec = TOOLS[name]
    if not spec.pythonpath:
        return {}
    prefix = str(PYTHONPATHS[spec.pythonpath]())
    existing = os.environ.get("PYTHONPATH", "")
    return {"PYTHONPATH": f"{prefix}{os.pathsep}{existing}" if existing else prefix}


def annotated_data_root() -> Path:
    """The picker's `--data_root` for the 300 annotated micrographs of each entry."""
    return data_root() / "cryoppp"


def fullset_data_root() -> Path:
    """The picker's `--data_root` for the full depositions."""
    return data_root() / "cryoppp_fullset"


def picker_images(root: Path, empiar: str) -> Path:
    """`<data root>/<id>/images`, the directory the picker reads micrographs from.

    The layout below the root is the picker's own and is not negotiable: it appends
    `<EMPIAR id>/images` to whatever root it is given. Both roots above already have
    the `<id>/` level, so each entry needs an `images` directory or symlink beside its
    `micrographs`.
    """
    return Path(root) / empiar / "images"


# --- the reconstruction package ------------------------------------------

def recon_python() -> Path:
    """The interpreter that has cryosparc-tools (the `recon` environment)."""
    return venv_python("recon")


def recon_env() -> dict:
    """Environment additions that make `rapick.*` importable from any interpreter."""
    src = str(REPO_ROOT / "src")
    existing = os.environ.get("PYTHONPATH", "")
    return {"PYTHONPATH": f"{src}{os.pathsep}{existing}" if existing else src}


def recon_profile() -> Path:
    """The CryoSPARC job-type / port contract every reconstruction job is built from."""
    path = Path(os.environ.get("RAPICK_RECON_PROFILE") or
                (REPO_ROOT / "configs" / "cryosparc_v47.yaml"))
    if not path.is_file():
        raise ConfigError(f"no CryoSPARC profile at {path}; set RAPICK_RECON_PROFILE")
    return path


def recon_config() -> Path:
    """configs/recon.yaml: the class_2D and reconstruction parameters, shared by every
    arm so that two arms can never differ in what the chain does to their particles.

    `RAPICK_RECON_CONFIG` overrides, so a site that lays configs/ out differently does
    not have to edit this package.
    """
    path = Path(os.environ.get("RAPICK_RECON_CONFIG") or
                (REPO_ROOT / "configs" / "recon.yaml"))
    if not path.is_file():
        raise ConfigError(f"no reconstruction config at {path} "
                          f"(set RAPICK_RECON_CONFIG to override)")
    return path
