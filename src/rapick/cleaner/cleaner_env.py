"""Paths, model location and per-entry constants shared by the cleaner scripts.

The mask precompute, the two STAR filters and the analysis scripts all have to
agree on where the micrographs are, which model weight to load, and which box
size to preprocess an EMPIAR entry with, so those live here instead of being
repeated per script.

Every path comes from the environment variables in docs/CONFIGURATION.md.
Nothing is hardcoded to a machine, and a variable that cannot be resolved raises
an error naming it rather than falling back to a directory that happens to
exist.

The denoised-background helpers and the ptxas fix are shared with the
contamination-detection driver of the research repository, which is not part of
this release; they are kept here because the STAR filters and the mask
precompute need them.
"""
import contextlib
import glob
import os
import sys
from pathlib import Path

# src/rapick/cleaner/cleaner_env.py -> cleaner -> rapick -> src -> repository root
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

if str(HERE) not in sys.path:      # the scripts here import each other flat
    sys.path.insert(0, str(HERE))


def env_path(name, default=None):
    """Read a path-valued environment variable, naming it when it is missing."""
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is not None:
        return Path(default)
    raise SystemExit(
        "environment variable %s is not set; see docs/CONFIGURATION.md" % name
    )


def data_root():
    """`$RAPICK_DATA`: micrographs, annotations and pretrained weights."""
    return env_path("RAPICK_DATA")


def work_root():
    """`$RAPICK_WORK`: masks, filtered STAR and everything else a run produces."""
    return env_path("RAPICK_WORK")


def upstream_dir():
    """The pinned MicrographCleaner checkout (`tf2` branch), under `$RAPICK_THIRD_PARTY`,
    which defaults to `<repo>/third_party`. plot_mask_postproc_figures.py imports
    `fixJumpInBorders` from here."""
    return env_path("RAPICK_THIRD_PARTY", REPO_ROOT / "third_party") / "micrograph_cleaner_em"


# --- input layout under $RAPICK_DATA -----------------------------------------

def annotated_mic_dir(eid):
    """The 300 annotated micrographs of one entry."""
    return os.path.join(data_root(), "cryoppp", str(eid), "micrographs")


def fullset_root():
    """Parent of <id>/micrographs for the full deposition."""
    return os.path.join(data_root(), "cryoppp_fullset")


def ground_truth_star(eid):
    """The CryoPPP annotations of one entry."""
    return os.path.join(data_root(), "cryoppp", str(eid), "ground_truth",
                        "empiar-%s_particles_selected.star" % eid)


# --- output layout under $RAPICK_WORK ----------------------------------------

def mask_store_root():
    """Triangular-blend masks: `$RAPICK_WORK/masks`, one subdirectory per entry."""
    return os.path.join(work_root(), "masks")


def mask_dir_of(eid):
    """Triangular-blend masks of one entry: `$RAPICK_WORK/masks/<id>`."""
    return os.path.join(mask_store_root(), str(eid))


def official_mask_store_root():
    """Masks assembled by the released post-processing: `$RAPICK_WORK/masks_official`.

    No script in this release writes this store; it is the input the mask-method
    comparison reads for the released arm. See this directory's README.
    """
    return os.path.join(work_root(), "masks_official")


def picks_dir_of(eid):
    """Filtered STAR of one entry: `$RAPICK_WORK/picks/<id>`."""
    return os.path.join(work_root(), "picks", str(eid))


# --- pretrained model --------------------------------------------------------

# download_model.sh installs the weight here. Upstream's own default location is
# a directory, not a file, so every script passes an explicit .h5 path.
MODEL_BASENAME = "micrograph_cleaner_defaultModel.h5"


def default_model_path():
    """`$RAPICK_DATA/checkpoints/micrograph_cleaner_defaultModel.h5`."""
    return os.path.join(data_root(), "checkpoints", MODEL_BASENAME)


def resolve_model(override=None):
    """The model weight to load: `--model` when given, otherwise the default path."""
    return override or default_model_path()


# --- per-entry constants -----------------------------------------------------

# EMPIAR entries MicrographCleaner was trained on (= in-distribution). This split
# is MicrographCleaner's own and is unrelated to any picker's training set.
TRAIN_EMPIAR = {
    10005, 10028, 10033, 10049, 10061, 10075, 10077, 10081,
    10090, 10093, 10097, 10099, 10168, 10175, 10190, 10203,
}

# Nominal particle diameters (px, mrc scale), used as MaskPredictor's boxSize.
# Unknown entries get DEFAULT_BOX.
DIAMETERS = {10017: 108, 10028: 224, 10081: 154, 10093: 172,
             10345: 149, 10532: 174, 11056: 164}
DEFAULT_BOX = 180           # a sane default particle box (px) for unknown entries
BOX_MIN, BOX_MAX = 64, 512  # clamp, to avoid an extreme downsample


def dist_class_of(eid):
    """EMPIAR id -> 'in_distribution' | 'out_of_distribution'."""
    return "in_distribution" if int(eid) in TRAIN_EMPIAR else "out_of_distribution"


def box_size_of(eid, override=None):
    if override:
        return int(override)
    box = DIAMETERS.get(int(eid), DEFAULT_BOX)
    return int(min(BOX_MAX, max(BOX_MIN, box)))


# --- runtime fixes -----------------------------------------------------------

def _ensure_modern_ptxas():
    """Put the pip-bundled CUDA-12 ptxas first on PATH, so TF 2.16 (XLA) runs on Ada
    (RTX 4090, CC 8.9). Left alone, the old /usr/bin/ptxas (CUDA 11.5) is picked up
    and predict dies with 'XLA requires ptxas version 11.8 or higher'.
    """
    for base in (sys.prefix, os.path.dirname(os.path.dirname(sys.executable))):
        hits = glob.glob(os.path.join(
            base, "lib", "python*", "site-packages", "nvidia", "cuda_nvcc", "bin"))
        if hits:
            os.environ["PATH"] = hits[0] + os.pathsep + os.environ.get("PATH", "")
            return hits[0]
    return None


@contextlib.contextmanager
def suppress_stdout():
    """Silence the keras predict progress bar (verbose=1 is hardcoded upstream)."""
    saved = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = saved


# --- denoised backgrounds ----------------------------------------------------
# The background of an overlay is normally denoised per micrograph, which is slow.
# CryoSegNet's released denoised JPGs run the same processing chain in the same
# flipud orientation, so an existing JPG only has to be resized to serve as the
# background, and a micrograph without one falls back to denoise_flip_frame.
#   train_dataset/<id>/<split>/images/<stem>.jpg   (split = train/val ...)
#   test_dataset/<id>/images/<stem>.jpg

def default_denoised_root():
    """`$RAPICK_DATA/cryosegnet_dataset` when it exists, otherwise None."""
    root = os.path.join(data_root(), "cryosegnet_dataset")
    return root if os.path.isdir(root) else None


def build_denoised_index(denoised_root, eid):
    """Build {micrograph stem: jpg path} for one entry from <denoised_root>'s
    train/test. Called once per entry so later lookups are O(1). Empty dict when
    the root is not set."""
    if not denoised_root:
        return {}
    index = {}
    patterns = [                                   # test first, so test wins
        os.path.join(denoised_root, "test_dataset", str(eid), "images", "*.jpg"),
        os.path.join(denoised_root, "train_dataset", str(eid), "*", "images", "*.jpg"),
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            index.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return index


def load_denoised(index, mic_name):
    """Read a micrograph's denoised image (flipud, full-res uint8) from the index.
    None when it is not in the index or cannot be read, in which case the caller
    falls back to denoise_flip_frame."""
    p = index.get(os.path.splitext(mic_name)[0])
    if not p:
        return None
    import cv2
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)


# The background denoise must stay identical to CryoSegNet's, so it is defined
# once in overlay_panel and re-exported here rather than reimplemented.
from overlay_panel import denoise_flip_frame  # noqa: E402,F401
