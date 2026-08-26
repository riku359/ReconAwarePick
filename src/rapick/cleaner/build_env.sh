#!/usr/bin/env bash
# Build the MicrographCleaner venv from the committed uv lockfile
# (envs/micrograph_cleaner/). tf2 branch = TF 2.16 / Keras 3 on py3.10.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/rapick/cleaner
REPO="$(cd "$HERE/../../.." && pwd)"                    # repository root
ENV_PROJECT="$REPO/envs/micrograph_cleaner"            # pyproject.toml + uv.lock

# Where the venv itself lives. Point RAPICK_ENVS at a local SSD to keep it off a
# small code disk (docs/CONFIGURATION.md).
RAPICK_ENVS="${RAPICK_ENVS:-$REPO}"
ENV_DIR="${RAPICK_ENVS%/}/envs/micrograph_cleaner/.venv"

# Keep the uv cache on the same disk as the venv (uv's file locks hang on NFS).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$ENV_DIR")/.uv_cache}"
export UV_LINK_MODE=copy
mkdir -p "$(dirname "$ENV_DIR")" "$UV_CACHE_DIR"

# uv.lock refers to upstream as an editable source, so it cannot sync without the clone.
UPSTREAM="${RAPICK_THIRD_PARTY:-$REPO/third_party}/micrograph_cleaner_em"
[ -d "$UPSTREAM" ] || {
  echo "error: $UPSTREAM is missing. Clone the tf2 branch first:" >&2
  echo "  git clone -b tf2 https://github.com/rsanchezgarc/micrograph_cleaner_em.git $UPSTREAM" >&2
  exit 1
}

[ -f "$ENV_PROJECT/pyproject.toml" ] || {
  echo "error: $ENV_PROJECT/pyproject.toml is missing" >&2
  exit 1
}

( cd "$ENV_PROJECT" && UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --locked )

echo "BUILD_DONE  ($ENV_DIR)"
"$ENV_DIR/bin/python" -c "import tensorflow as tf, keras; print('tf', tf.__version__, 'keras', keras.__version__)"
