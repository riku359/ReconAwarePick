#!/usr/bin/env bash
# Fetch the pinned upstream code and build the per-stage environments.
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Fetch the pinned upstream code and build the per-stage environments.

Usage:
  bash scripts/setup.sh                 pipeline only
  bash scripts/setup.sh --baselines     also crYOLO, Topaz and CryoSegNet
  bash scripts/setup.sh --skip-envs     clone and overlay, build nothing

Every pin comes from repos.lock.yaml, which is the single source of truth.
HELP
}

# -e  stop at the first command that fails
# -u  treat reading an unset variable as an error
# -o pipefail  a pipeline fails when any command in it fails, not just the last
set -euo pipefail

# This script lives in scripts/, so the repository root is the directory above it.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="${RAPICK_THIRD_PARTY:-$REPO/third_party}"
WITH_BASELINES=0
SKIP_ENVS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --baselines) WITH_BASELINES=1; shift ;;
    --skip-envs) SKIP_ENVS=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

for tool in git curl; do
  if ! command -v "$tool" >/dev/null; then
    echo "error: $tool is required" >&2
    exit 1
  fi
done
if [ "$SKIP_ENVS" -eq 0 ] && ! command -v uv >/dev/null; then
  echo "error: uv is required to build the environments." >&2
  echo "       Install it from https://docs.astral.sh/uv/, or pass --skip-envs." >&2
  exit 1
fi

mkdir -p "$THIRD_PARTY"

# Read one field out of repos.lock.yaml. Keeping the pins in exactly one place
# means a stale copy here can never disagree with what the lock file says.
pin() {  # pin <section> <name> <field>
  RAPICK_REPO="$REPO" uv run --quiet --with pyyaml python3 -c '
import os, sys, yaml

section, name, field = sys.argv[1], sys.argv[2], sys.argv[3]
lock = yaml.safe_load(open(os.environ["RAPICK_REPO"] + "/repos.lock.yaml"))
print(lock[section][name].get(field, "") or "")
' "$1" "$2" "$3"
}

clone_pinned() {  # clone_pinned <section> <name>
  local section="$1"
  local name="$2"
  local url path commit branch subdir dest
  url=$(pin "$section" "$name" url)
  path=$(pin "$section" "$name" path)
  commit=$(pin "$section" "$name" commit)
  branch=$(pin "$section" "$name" branch)
  subdir=$(pin "$section" "$name" subdir)
  if [ -z "$path" ]; then
    path="$name"
  fi
  dest="$THIRD_PARTY/$path"

  if [ -d "$dest/.git" ]; then
    echo "  $name: already at $dest, leaving it alone"
    return
  fi

  # A repository that declares a subdir is one we need a corner of. Magellon is
  # 2.2 GB whole and 164 MB as the one directory CryoSift lives in, so take the
  # corner: blobless, no checkout, then a sparse checkout of that path.
  if [ -n "$subdir" ]; then
    echo "  $name: cloning $url (sparse: $subdir)"
    git clone --quiet --filter=blob:none --no-checkout "$url" "$dest"
    git -C "$dest" sparse-checkout set --no-cone "$subdir"
    if [ -n "$commit" ]; then
      git -C "$dest" checkout --quiet "$commit"
      echo "  $name: pinned at $commit ($(du -sh "$dest" | cut -f1))"
    else
      git -C "$dest" checkout --quiet "$branch"
      echo "  $name: pinned at $branch ($(du -sh "$dest" | cut -f1))"
    fi
    return
  fi

  echo "  $name: cloning $url"
  if [ -n "$branch" ]; then
    git clone --quiet --branch "$branch" "$url" "$dest"
  else
    git clone --quiet "$url" "$dest"
  fi
  if [ -n "$commit" ]; then
    git -C "$dest" checkout --quiet "$commit"
    echo "  $name: pinned at $commit"
  else
    echo "  $name: pinned at branch $branch ($(git -C "$dest" rev-parse --short HEAD))"
  fi
}

echo "==> Fetching upstream code into $THIRD_PARTY"
clone_pinned pipeline cryotransformer
clone_pinned pipeline micrograph_cleaner
clone_pinned pipeline cryosift

if [ "$WITH_BASELINES" -eq 1 ]; then
  echo "==> Fetching the comparison pickers (see docs/BASELINES.md)"
  clone_pinned baselines topaz
  clone_pinned baselines cryosegnet
  clone_pinned baselines cryolo
  echo "  note: crYOLO's repository is documentation only. The software installs"
  echo "        from PyPI under a non-commercial licence; see docs/BASELINES.md."
fi

# --- the CryoTransformer overlay -------------------------------------------
# predict.py, train.py and datasets/micrograph.py are modified copies of upstream;
# finetune.py, predict_fullset.py and head_repair/ are new. Upstream's flat imports
# (`from models import build_model`) only resolve from inside its own tree, which
# is why these are copied in rather than imported from src/.
echo "==> Applying the CryoTransformer overlay"
CT="$THIRD_PARTY/$(pin pipeline cryotransformer path)"
if [ ! -d "$CT" ]; then
  echo "error: $CT is missing; the clone above did not succeed" >&2
  exit 1
fi
OVERLAY="$REPO/src/rapick/picker/overlay"
cp -R "$OVERLAY/." "$CT/"
OVERLAY_FILES=$(find "$OVERLAY" -name '*.py' | wc -l | tr -d ' ')
echo "  copied $OVERLAY_FILES files over the clone"
echo "  the changes against upstream are readable in src/rapick/picker/patches/"

# --- environments ------------------------------------------------------------
if [ "$SKIP_ENVS" -eq 1 ]; then
  echo "==> Skipping environment builds (--skip-envs)"
  echo "Done."
  exit 0
fi

# uv's file locks hang on NFS, so the cache has to sit on a local disk.
export UV_LINK_MODE=copy
if [ -z "${UV_CACHE_DIR:-}" ]; then
  UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache-rapick"
fi
export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"

# Two stages do not use a uv project. CryoSift's lockfile is upstream's own frozen
# requirements list, kept unconverted so the difference from upstream stays legible,
# and MicrographCleaner installs its upstream checkout editable. Each has its own
# build script; delegate rather than reimplementing them here.
# A case rather than an associative array: macOS still ships bash 3.2, which has none.
own_builder() {  # own_builder <name> -> path, or empty
  case "$1" in
    cryosift)           echo "src/rapick/select2d/scripts/build_env.sh" ;;
    micrograph_cleaner) echo "src/rapick/cleaner/build_env.sh" ;;
    *)                  echo "" ;;
  esac
}

build_env() {  # build_env <name>
  local name="$1"
  local dir="$REPO/envs/$name"
  local own
  own="$(own_builder "$name")"

  if [ -n "$own" ]; then
    if [ ! -f "$REPO/$own" ]; then
      echo "  $name: $own is missing, skipping"
      return
    fi
    echo "  $name: building via $own"
    bash "$REPO/$own"
    return
  fi

  if [ ! -f "$dir/pyproject.toml" ]; then
    echo "  $name: no pyproject.toml and no build script, skipping"
    return
  fi
  if [ ! -f "$dir/uv.lock" ]; then
    echo "  $name: no uv.lock committed; run 'uv lock' in $dir first, skipping"
    return
  fi
  echo "  $name: building"
  # In a subshell, so that the cd does not follow us into the next environment.
  ( cd "$dir" && UV_PROJECT_ENVIRONMENT="$dir/.venv" uv sync --quiet --locked )
}

echo "==> Building environments"
for env_name in cryotransformer micrograph_cleaner cryosift recon; do
  build_env "$env_name"
done

if [ "$WITH_BASELINES" -eq 1 ]; then
  for env_name in topaz cryosegnet; do
    build_env "$env_name"
  done
  echo "  cryolo: needs conda, not uv. See docs/BASELINES.md."
fi

echo
echo "Done. Next:"
echo "  cp .env.example .env      and fill in your CryoSPARC credentials"
echo "  bash scripts/download.sh"
