#!/usr/bin/env bash
# Reconstruct one arm from its 2D classification and collect its metrics (Sec. S1).
# `--help` prints the whole story; usage() below is the one copy of it.

usage() {
  cat <<'HELP'
Reconstruct one arm and collect its metrics (Sec. S1).

  ab-initio x3 -> homogeneous refine x3 -> best of the three by GSFSC 0.143
                                        -> local resolution on the winner
                                        -> metrics.json

Everything above ab-initio belongs to scripts/2d_classification.sh, which has to
have run first: this stage never creates an import, an extraction or a class_2D of
its own. Two ways in, and they differ only in which particles reach ab-initio.

  The whole classified stack:

    bash scripts/reconstruct.sh --entry 10081 --name cryotransformer_mask

  What a 2D class selection kept:

    bash scripts/reconstruct.sh --entry 10081 --parent cryotransformer_mask
    bash scripts/reconstruct.sh --entry 10081 --parent fb_mask --name fb

  --name      what to record this arm as. Names $RAPICK_WORK/empiar_<entry>/
              <setting>/<name>/. With --parent it defaults to <parent>_select.
  --parent    the arm whose class_2D the selection sits on. Its final Select 2D
              Classes job is read out of the state.json scripts/select2d.sh wrote.
  --select2d  that job, named directly instead.
  --no-local-res   skip the local-resolution estimate on the winner.

THREE SEEDS, NOT ONE. The protocol (Sec. 4.2) reports the best of three by GSFSC
0.143, so a single-seed run reproduces something the paper does not report. If a
seed's ab-initio dies, advance the seed number rather than reporting a
best-of-two as a best-of-three.
HELP
}

source "$(dirname "$0")/_common.sh"

ENTRY=""
NAME=""
PARENT=""
SETTING="full"
SEEDS="0,1,2"
SELECT2D=""
NO_LOCAL_RES=""
DRY_RUN=""
GPUS="${RAPICK_GPU:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --entry)         ENTRY="$2"; shift 2 ;;
    --name)          NAME="$2"; shift 2 ;;
    --parent)        PARENT="$2"; shift 2 ;;
    --setting)       SETTING="$2"; shift 2 ;;
    --seeds)         SEEDS="$2"; shift 2 ;;
    --select2d)      SELECT2D="$2"; shift 2 ;;
    --no-local-res)  NO_LOCAL_RES="--no-local-res"; shift ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --dry-run)       DRY_RUN="--dry-run"; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
require_entry "$ENTRY"
require_setting "$SETTING"

# A selection reconstructs its parent's class_2D one step lower, so its own name is
# its parent's with the stage appended -- the same rule the STAR names follow.
if [ -z "$NAME" ] && [ -n "$PARENT" ]; then
  NAME="${PARENT}_select"
fi
if [ -n "$SELECT2D" ] && [ -z "$PARENT" ]; then
  echo "error: --select2d needs --parent: the arm whose class_2D that selection sits" >&2
  echo "       on is what the run is seeded from and checked against." >&2
  exit 2
fi
if [ -z "$NAME" ]; then
  echo "error: --name is required (or --parent, which implies <parent>_select)." >&2
  exit 2
fi

# --help must work with nothing configured, so the roots are demanded only once
# the arguments are known to be valid.
require_roots

PY="$(venv_python recon)"
DATACFG="$REPO/configs/datasets/empiar_$ENTRY.yaml"
RECONCFG="$REPO/configs/recon.yaml"

# Every call into the reconstruction CLI goes through here.
recon() {  # recon <subcommand> [flags...]
  PYTHONPATH="$REPO/src" "$PY" -m rapick.recon.cli "$@"
}

if [ -z "$PARENT" ]; then
  # --- the whole classified stack ------------------------------------------------
  MANIFEST="$(arm_dir "$ENTRY" "$SETTING" "$NAME")/manifest.json"
  if [ ! -f "$MANIFEST" ]; then
    echo "error: no manifest for '$NAME' at" >&2
    echo "         $MANIFEST" >&2
    echo "       Classify its picks first:" >&2
    echo "         bash scripts/2d_classification.sh --entry $ENTRY --star <picks>.star --name $NAME" >&2
    exit 1
  fi
  if [ -z "$(manifest_class2d "$MANIFEST")" ]; then
    echo "error: '$NAME' has a manifest but no class_2D job in it." >&2
    echo "       Run scripts/2d_classification.sh --entry $ENTRY --name $NAME first;" >&2
    echo "       this stage starts at ab-initio and creates nothing above it." >&2
    exit 1
  fi
  # The picks are named once, when they are classified. Reading them back out of the
  # manifest is what lets the completed import / extract / class_2D be recognised and
  # reused instead of rebuilt: their fingerprint is that STAR's sha256.
  STAR="$(manifest_input_star "$MANIFEST")"
  if [ -z "$STAR" ]; then
    echo "error: $MANIFEST records no input STAR." >&2
    exit 1
  fi

  banner "Preflight"
  recon check-setup --condition "$RECONCFG" --dataset "$DATACFG" \
      --setting "$SETTING" --source "$NAME" --star "$STAR"

  # `rapick-recon run` has no dry run: it either creates the chain or it does not.
  if [ -n "$DRY_RUN" ]; then
    echo
    echo "Dry run. The preflight passed; nothing was created. A real run would:"
    echo "  reuse the import, extraction and class_2D recorded in the manifest, then"
    echo "  ab-initio and refine on seeds $SEEDS, keep the best by GSFSC 0.143, and"
    echo "  estimate local resolution on it."
    echo "  Output: $(arm_dir "$ENTRY" "$SETTING" "$NAME")/"
    exit 0
  fi

  banner "Reconstructing $NAME on $ENTRY ($SETTING), seeds $SEEDS"
  # $NO_LOCAL_RES is deliberately left unquoted: it is either empty, and then adds
  # no argument at all, or the single word --no-local-res.
  recon run --condition "$RECONCFG" --dataset "$DATACFG" --setting "$SETTING" \
      --source "$NAME" --star "$STAR" --seeds "$SEEDS" --gpus "$GPUS" $NO_LOCAL_RES
else
  # --- what a 2D class selection kept ---------------------------------------------
  MANIFEST="$(arm_dir "$ENTRY" "$SETTING" "$PARENT")/manifest.json"
  if [ ! -f "$MANIFEST" ]; then
    echo "error: no manifest for the parent arm '$PARENT' at" >&2
    echo "         $MANIFEST" >&2
    echo "       Classify its picks first:" >&2
    echo "         bash scripts/2d_classification.sh --entry $ENTRY --star <picks>.star --name $PARENT" >&2
    exit 1
  fi

  # The selection's final Select 2D Classes job, at the 3.5 cutoff, is recorded in
  # the state.json that scripts/select2d.sh left behind.
  if [ -z "$SELECT2D" ]; then
    CLASS2D="$(manifest_class2d "$MANIFEST")"
    STATE="$(select2d_state_file "$CLASS2D")"
    if [ -f "$STATE" ]; then
      SELECT2D="$(select2d_at_cutoff "$STATE")"
    fi
  fi
  if [ -z "$SELECT2D" ]; then
    echo "error: no 2D class selection to reconstruct from." >&2
    echo "       Build it first:" >&2
    echo "         bash scripts/select2d.sh --entry $ENTRY --name $PARENT" >&2
    echo "       or name the job with --select2d J<n>." >&2
    exit 1
  fi

  banner "Reconstructing $NAME on $ENTRY from $SELECT2D, seeds $SEEDS"
  # This driver takes a single GPU, so hand it the first of the list.
  FIRST_GPU="$(echo "$GPUS" | cut -d, -f1)"
  # $DRY_RUN and $NO_LOCAL_RES are deliberately left unquoted: each is either empty,
  # and then adds no argument at all, or a single word.
  recon reconstruct-from-selection \
      --entry "$ENTRY" --select2d "$SELECT2D" \
      --condition "$NAME" --parent "$PARENT" --setting "$SETTING" \
      --seeds "$SEEDS" --gpu "$FIRST_GPU" $NO_LOCAL_RES $DRY_RUN
fi

if [ -n "$DRY_RUN" ]; then
  exit 0
fi

# The from-selection driver deliberately does not collect, so this is not optional:
# without it the arm ends up with a manifest and no metrics.json.
banner "Collecting metrics"
recon collect --condition "$RECONCFG" --dataset "$DATACFG" --setting "$SETTING" \
    --source "$NAME"

OUT="$(arm_dir "$ENTRY" "$SETTING" "$NAME")"
echo
echo "Metrics: $OUT/metrics.json"
if [ -f "$OUT/metrics.json" ]; then
  # A one-line summary of the file just written. Reading it is a convenience, so a
  # failure here must not fail the run.
  uv run --quiet python3 -c '
import json, sys

metrics = json.load(open(sys.argv[1]))
best = metrics.get("best") or {}
resolution = metrics.get("res_gsfsc_0143") or best.get("res_gsfsc_0143")
if resolution:
    print(f"  GSFSC 0.143: {resolution} A  (best of the seeds run)")
' "$OUT/metrics.json" || true
fi
