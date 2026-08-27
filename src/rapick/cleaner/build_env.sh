#!/usr/bin/env bash
# Build the MicrographCleaner venv from the committed uv lockfile
# (envs/micrograph_cleaner/). tf2 branch = TF 2.16 / Keras 3 on py3.10.

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"       # .../src/rapick/cleaner
REPO="$(cd "$HERE/../../.." && pwd)"        # repository root
ENV_PROJECT="$REPO/envs/micrograph_cleaner" # pyproject.toml + uv.lock

# Where the venv itself lives. Point RAPICK_ENVS at a local SSD to keep it off a
# small code disk (docs/CONFIGURATION.md). A trailing slash on it is dropped, so
# that the path below never ends up with two.
ENVS_ROOT="$(echo "${RAPICK_ENVS:-$REPO}" | sed 's|/*$||')"
ENV_DIR="$ENVS_ROOT/envs/micrograph_cleaner/.venv"
ENV_PARENT="$(dirname "$ENV_DIR")"

# Keep the uv cache on the same disk as the venv (uv's file locks hang on NFS).
if [ -z "${UV_CACHE_DIR:-}" ]; then
  UV_CACHE_DIR="$ENV_PARENT/.uv_cache"
fi
export UV_CACHE_DIR
export UV_LINK_MODE=copy
mkdir -p "$ENV_PARENT" "$UV_CACHE_DIR"

# uv.lock refers to upstream as an editable source, so it cannot sync without the clone.
UPSTREAM="${RAPICK_THIRD_PARTY:-$REPO/third_party}/micrograph_cleaner_em"
if [ ! -d "$UPSTREAM" ]; then
  echo "error: $UPSTREAM is missing. Clone the tf2 branch first:" >&2
  echo "  git clone -b tf2 https://github.com/rsanchezgarc/micrograph_cleaner_em.git $UPSTREAM" >&2
  exit 1
fi

if [ ! -f "$ENV_PROJECT/pyproject.toml" ]; then
  echo "error: $ENV_PROJECT/pyproject.toml is missing" >&2
  exit 1
fi

# In a subshell: uv sync reads the project in the current directory.
( cd "$ENV_PROJECT" && UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --locked )

echo "BUILD_DONE  ($ENV_DIR)"
"$ENV_DIR/bin/python" -c "import tensorflow as tf, keras; print('tf', tf.__version__, 'keras', keras.__version__)"
