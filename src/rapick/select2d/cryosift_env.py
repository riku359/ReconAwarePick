"""Upstream imports and the CryoSPARC connection shared by the CryoSift wrappers.

Upstream (`Sandbox/particle_processor` in sstagg/Magellon) is used unmodified.
Upstream's `extract_class_scores.get_class_labels` holds the weights path relative to
the current working directory (`MODEL_PATH = "class_labeling/final_model/
final_model_cont.pth"`), so this module calls `cryosparcpredict` directly instead,
which takes the weights path as an argument.

Every path comes from the environment variables in docs/CONFIGURATION.md. Nothing is
hardcoded to a machine, and a variable that cannot be resolved raises an error naming
it rather than falling back to a directory that happens to exist.
"""

import os
import sys
from pathlib import Path

# src/rapick/select2d/cryosift_env.py -> select2d -> rapick -> src -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]

# CryoSPARC credentials come from the repository-root .env, the single place this
# repository keeps secrets. `--env` swaps in another file (docs/CONFIGURATION.md).
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# Layout inside the upstream checkout. The pretrained CNN ships inside the clone,
# so there is no separate weights download.
UPSTREAM_SUBPATH = Path("magellon") / "Sandbox" / "particle_processor"
WEIGHTS_SUBPATH = Path("class_labeling") / "final_model" / "final_model_cont.pth"


def env_path(name, default=None):
    """Read a path-valued environment variable, naming it when it is missing."""
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is not None:
        return Path(default)
    raise SystemExit(
        f"environment variable {name} is not set; see docs/CONFIGURATION.md"
    )


def third_party_root():
    """`$RAPICK_THIRD_PARTY`, which defaults to `<repo>/third_party`."""
    return env_path("RAPICK_THIRD_PARTY", REPO_ROOT / "third_party")


def upstream_dir():
    """The pinned Magellon checkout that carries the CryoSift model."""
    return third_party_root() / UPSTREAM_SUBPATH


def weights_path():
    """The pretrained CNN, bundled inside the Magellon checkout."""
    return upstream_dir() / WEIGHTS_SUBPATH


def default_gpu():
    """`$RAPICK_GPU`, the GPU index the re-classification jobs default to."""
    return int(os.environ.get("RAPICK_GPU") or 0)


def import_upstream():
    """Return upstream's `cryosparcpredict` and star parser.

    Side effect: prepends the upstream checkout to `sys.path`.
    """
    weights = weights_path()
    if not weights.is_file():
        raise SystemExit(
            f"pretrained weights not found: {weights}\n"
            "Fetch the upstream checkout first; see the Setup section of "
            "src/rapick/select2d/README.md"
        )

    upstream = str(upstream_dir())
    if upstream not in sys.path:
        sys.path.insert(0, upstream)

    from class_labeling.cryosparc_labeler import cryosparcpredict
    from extract_class_scores import extract_scores_from_star

    return cryosparcpredict, extract_scores_from_star


def read_env(env_file=None):
    """Parse the repository-root `.env` into a dict of key -> value."""
    env_file = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if not env_file.is_file():
        raise SystemExit(
            f"{env_file} not found. Copy .env.example to the repository root as .env "
            "and fill it in; see docs/CONFIGURATION.md"
        )

    env = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Same unquoting as rapick.recon.config.load_env: a value written as
        # KEY="..." must not authenticate with the quote characters in it.
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def require(env, key, env_file=None):
    """Fetch a `.env` key, naming it in the error when it is missing or empty."""
    value = env.get(key)
    if not value:
        raise SystemExit(
            f"{key} is not set in {env_file or DEFAULT_ENV_FILE}; "
            "see docs/CONFIGURATION.md"
        )
    return value


def connect(env=None, env_file=None):
    """Open a CryoSPARC session from the credentials in `.env`."""
    from cryosparc.tools import CryoSPARC

    env = env if env is not None else read_env(env_file)
    cs = CryoSPARC(
        host=require(env, "CRYOSPARC_HOST", env_file),
        base_port=int(require(env, "CRYOSPARC_PORT", env_file)),
        license=require(env, "CRYOSPARC_LICENSE_ID", env_file),
        email=require(env, "CRYOSPARC_EMAIL", env_file),
        password=require(env, "CRYOSPARC_PASSWORD", env_file),
    )
    if not cs.test_connection():
        raise SystemExit("could not connect to CryoSPARC")
    return cs


def resolve_work_dir(explicit=None):
    """Where this stage writes: `$RAPICK_WORK/select2d`.

    Each run keys its own subdirectory off the CryoSPARC project uid plus the starter
    job uid, so runs never collide across datasets or experiments.
    """
    if explicit:
        return Path(explicit).expanduser()
    return env_path("RAPICK_WORK") / "select2d"


def resolve_job_dir(cs, project_uid, job_uid):
    """Locate a job directory on disk.

    Upstream's `CryosparcPredictor` opens the job's `.mrc` and `.cs` files directly, so
    this stage has to run on the CryoSPARC machine or on one that sees the same shared
    filesystem. The project directory is asked of the server, never hardcoded.
    """
    project_dir = Path(cs.cli.get_project(project_uid)["project_dir"])
    job_dir = project_dir / job_uid
    if not job_dir.is_dir():
        raise SystemExit(f"job directory not found: {job_dir}")
    return job_dir


def parse_model_star(star_path):
    """Read the scores out of the model.star that `cryosparcpredict` writes.

    Upstream's `extract_class_scores.extract_scores_from_star` picks the score up with
    the regex `\\d+@.*\\s(\\d+\\.\\d+)`, which cannot match a negative number. Scores do
    go negative: `unconvert_labels` returns `5 - 5*pred + weight`, so any model output
    above 1 gives a negative score. The line then fails to match and the class is
    silently recorded as 5.0, the worst possible score -- which means the BEST class is
    the one thrown away. Here the last token on the line is parsed as a float, so the
    sign survives.
    """
    scores = []
    for line in Path(star_path).read_text().splitlines():
        if "@" not in line:
            continue
        scores.append(float(line.split()[-1]))
    return scores
