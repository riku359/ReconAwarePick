#!/usr/bin/env bash
# Build the CryoSift virtual environment from the committed lockfile
# (envs/cryosift/requirements_exact.txt).
#
# Inference never touches the GPU: upstream's class_labeling/cryosparc_labeler.py
# hardcodes device='cpu'. requirements_exact.txt is upstream's own frozen list,
# vendored unchanged -- which is why CUDA wheels appear in it even though they are
# never used. It pins torch 2.6.0 + cryosparc-tools 4.7.0, matching a CryoSPARC v4.7
# server. It is not converted to a uv pyproject.toml + uv.lock because upstream's
# frozen list is already complete as a lockfile, and rewriting it would make the
# difference from upstream hard to follow.

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"       # src/rapick/select2d/scripts
STAGE_DIR="$(dirname "$HERE")"              # src/rapick/select2d
REPO="$(cd "$STAGE_DIR/../../.." && pwd)"   # repository root

REQ="$REPO/envs/cryosift/requirements_exact.txt"
if [ ! -f "$REQ" ]; then
  echo "error: lockfile not found: $REQ" >&2
  exit 1
fi

# RAPICK_ENVS says where the per-tool virtual environments are built; it defaults to
# the repository itself, which puts one .venv in each env directory. Point it at a
# local SSD when the code disk is small or sits on NFS (docs/CONFIGURATION.md).
ENV_DIR="${RAPICK_ENVS:-$REPO}/envs/cryosift/.venv"
LINK_DIR="$REPO/envs/cryosift/.venv"
ENV_PARENT="$(dirname "$ENV_DIR")"

# uv's file locks hang on NFS, so its cache must not land there. Read the filesystem
# type of the target and fall back to $TMPDIR when it is NFS. An explicit UV_CACHE_DIR
# always wins.
if [ -z "${UV_CACHE_DIR:-}" ]; then
  mkdir -p "$ENV_PARENT"
  # stat -f -c is GNU. Where the host's stat does not understand it we get
  # nothing back, and the disk is treated as local, which is the right default.
  FS_TYPE="$(stat -f -c %T "$ENV_PARENT" 2>/dev/null || true)"
  case "$FS_TYPE" in
    nfs*)
      UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-cryosift-$(id -un)"
      echo "note: $ENV_PARENT is NFS; moving the uv cache to $UV_CACHE_DIR" >&2
      ;;
    *)
      UV_CACHE_DIR="$(dirname "$ENV_PARENT")/.uv-cache"
      ;;
  esac
fi
export UV_CACHE_DIR
export UV_LINK_MODE=copy
mkdir -p "$ENV_PARENT" "$UV_CACHE_DIR"

# The pretrained weights ship inside the upstream clone, so there is nothing separate
# to download -- but without the clone there is nothing to run inference with.
UPSTREAM="${RAPICK_THIRD_PARTY:-$REPO/third_party}/magellon/Sandbox/particle_processor"
WEIGHTS="$UPSTREAM/class_labeling/final_model/final_model_cont.pth"
if [ ! -f "$WEIGHTS" ]; then
  echo "error: pretrained weights not found: $WEIGHTS" >&2
  echo "       fetch the upstream checkout first; see the Setup section of" >&2
  echo "       src/rapick/select2d/README.md" >&2
  exit 1
fi

# uv fetches a managed CPython where the host has no python3.12.
uv venv --python 3.12 "$ENV_DIR"

# pysqlite3 is dropped from upstream's list rather than removed from the committed
# copy of it, so that the file stays byte-identical to upstream and the deviation is
# visible here instead. It has no source wheel, so pip compiles it, and the compile
# needs sqlite3.h -- a system header most machines do not have. Nothing imports it:
# not upstream's particle_processor, not this stage, and Python has carried `sqlite3`
# in its standard library since long before the version this environment pins.
FILTERED="$(mktemp)"
trap 'rm -f "$FILTERED"' EXIT
grep -v '^pysqlite3==' "$REQ" > "$FILTERED"
if ! diff -q "$REQ" "$FILTERED" >/dev/null; then
  echo "note: skipping pysqlite3 (unused, and building it needs sqlite3.h)" >&2
fi
VIRTUAL_ENV="$ENV_DIR" uv pip install -r "$FILTERED"

# When RAPICK_ENVS points elsewhere, leave an absolute symlink at the in-repo path so
# every command in the README works unchanged. Moving the repository to another machine
# means re-running this script.
if [ "$ENV_DIR" != "$LINK_DIR" ]; then
  mkdir -p "$(dirname "$LINK_DIR")"
  ln -sfn "$ENV_DIR" "$LINK_DIR"
fi

echo "BUILD_DONE  ($LINK_DIR -> $ENV_DIR)"
"$ENV_DIR/bin/python" - <<'PY'
import torch, numpy, cv2, mrcfile, starfile, cryosparc
print("torch", torch.__version__, "| numpy", numpy.__version__,
      "| cv2", cv2.__version__, "| cryosparc-tools", cryosparc.__version__)
PY
