"""CryoPPP ground-truth access shared by the head-repair scripts.

The functions here were extracted verbatim from the monorepo's
`analysis_diagnosis/phase_a_facts.py` (`DIAMETERS_ALL`, `TRAIN_IDS_EXPECTED_COUNT`,
`cached_load_star_points`, `load_star_points_quoted_fallback`, `expand_alias_keys`,
`resolve_train_stem_key`) and `analysis_diagnosis/verify_hypotheses.py`
(`gt_match_labels`). Those two files are fact-checking and hypothesis-testing scripts
that the paper does not use; only these helpers are needed, so they live here instead
of shipping both scripts whole.

STAR parsing itself is not reimplemented: `load_star_points` and `normalize_mic_name`
are imported from the repository's one 2D scorer, `calc_common_2d_metrics`, and
re-exported here so the head-repair scripts have a single import point.

Environment: RAPICK_DATA (the CryoPPP entries) and RAPICK_WORK (the parse cache).
See docs/CONFIGURATION.md.
"""
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _import_common_2d_metrics():
    """Import calc_common_2d_metrics -- the repository's single STAR reader.

    scripts/00_setup.sh copies this directory over the upstream clone at
    $RAPICK_THIRD_PARTY/cryotransformer/, so a path relative to __file__ resolves only
    while the file still sits in the repository. Try an already-importable module first
    (put <repo>/src/rapick/eval on PYTHONPATH), then the in-repository location.
    """
    try:
        import calc_common_2d_metrics as module
        return module
    except ImportError:
        pass
    in_repo_eval_dir = Path(__file__).resolve().parents[3] / "eval"
    if in_repo_eval_dir.is_dir():
        sys.path.insert(0, str(in_repo_eval_dir))
        import calc_common_2d_metrics as module
        return module
    raise SystemExit(
        "cannot import calc_common_2d_metrics. Put <repo>/src/rapick/eval on "
        "PYTHONPATH before running this script. See src/rapick/picker/README.md.")


_ccm = _import_common_2d_metrics()
load_star_points = _ccm.load_star_points
normalize_mic_name = _ccm.normalize_mic_name
DIAMETERS = _ccm.DIAMETERS


def _require_env(name, what):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set; it must point at {what}. See docs/CONFIGURATION.md.")
    return Path(os.path.expanduser(value))


def cryoppp_root():
    """$RAPICK_DATA/cryoppp -- the CryoPPP entries (micrographs + annotations)."""
    return _require_env("RAPICK_DATA", "the downloaded input data") / "cryoppp"


def gt_cache_dir():
    """$RAPICK_WORK/head_repair/gt_cache -- pickled GT star parses (see below)."""
    return _require_env("RAPICK_WORK", "the pipeline's output tree") / "head_repair" / "gt_cache"


def gt_star_path(eid):
    """The CryoPPP ground-truth selected.star of one entry."""
    return cryoppp_root() / str(eid) / "ground_truth" / f"empiar-{eid}_particles_selected.star"


# EMPIAR-10077's GT star wraps every data_particles row in an extra, non-standard
# outer '"'..'"' with doubled '""' as the literal-quote escape (CSV-style, not real
# STAR/CIF quoting) -- because this deposition's own micrograph filenames contain
# spaces ("sb1_210512 pos 1038 1-2_1.mrc"), which the naive whitespace tokenizer in
# calc_common_2d_metrics.read_star_rows() cannot handle: it returns 0 rows for this
# file. Recover _rlnMicrographName/_rlnCoordinateX/_rlnCoordinateY directly with a
# regex instead of touching that shared, otherwise-correct parser used by the core
# 2D evaluation pipeline (10077 is not one of its CORE_IDS today).
_QUOTED_ROW_RE = re.compile(r'""(?P<mic>[^"]*?\.mrc)""\s+(?P<x>-?[\d.]+)\s+(?P<y>-?[\d.]+)')


def load_star_points_quoted_fallback(path: Path):
    points = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            m = _QUOTED_ROW_RE.search(line)
            if not m:
                continue
            mic = normalize_mic_name(m.group("mic"))
            points[mic].append((float(m.group("x")), float(m.group("y"))))
    return dict(points)


# The 22 training entries named in CryoTransformer's README (SN1-22), and the expected
# number of training micrographs each contributes.
TRAIN_IDS_EXPECTED_COUNT = {
    11183: 250, 11057: 250, 11051: 250, 10852: 270, 10816: 250, 10760: 250,
    10737: 250, 10671: 250, 10590: 250, 10526: 180, 10444: 250, 10406: 200,
    10387: 250, 10291: 250, 10289: 250, 10240: 250, 10184: 250, 10096: 250,
    10077: 250, 10075: 250, 10059: 250, 10005: 22,
}

# Particle diameters from the CryoPPP README's "Particle Diameter (px)" column.
DIAMETERS_ALL = {
    10389: 313, 10081: 154, 10289: 162, 11057: 186, 10444: 217, 10576: 265,
    10816: 359, 10526: 482, 11051: 214, 10760: 106, 11183: 159, 10671: 133,
    10291: 130, 10669: 730, 10077: 216, 10061: 471, 10028: 224, 10096: 84,
    10737: 179, 10387: 213, 10532: 174, 10240: 156, 10005: 142, 10017: 108,
    10075: 233, 10184: 118, 10059: 132, 10406: 212, 10590: 158, 10093: 172,
    10345: 149, 11056: 164, 10852: 123, 10947: 240,
}


def cached_load_star_points(star_path: Path):
    """Pickle-cache the result of load_star_points() under an (mtime, size) key.

    The 22 entries' GT stars come to nearly 800 MB in total and the per-line regex parse
    is slow, so this pays the cost once for scripts that read them repeatedly.
    """
    cache_dir = gt_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    st = star_path.stat()
    cache_key = f"{star_path.name}.{st.st_mtime_ns}.{st.st_size}.pkl"
    cache_path = cache_dir / cache_key
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    points = load_star_points(str(star_path))
    if not points:
        # standard whitespace tokenizer found no rows -- likely the 10077-style
        # quoted format, not a genuinely empty file. Confirm the file has content
        # before assuming that, so a truly-empty star still yields {} quietly.
        if star_path.stat().st_size > 500:
            points = load_star_points_quoted_fallback(star_path)
    with open(cache_path, "wb") as fh:
        # protocol 4 (not HIGHEST_PROTOCOL=5, py3.8+ only): this cache is also read
        # from the CryoTransformer environment, which is python 3.7.
        pickle.dump(points, fh, protocol=4)
    return points


def expand_alias_keys(d: dict) -> dict:
    """Add alias keys that absorb leftover suffix differences on the GT side.

    normalize_mic_name strips only a trailing ".mrc" from _rlnMicrographName, so the
    normalized strings do not always agree between the two sides: the original filename
    may have carried a double extension ("..._Enn.frames.mrc" -> "..._Enn.frames",
    EMPIAR-10737), or the train split's jpg may have dropped a motion-correction suffix
    that survives on the GT side ("..._patch_aligned_doseweighted", EMPIAR-10852). The
    rule is shared across value types (coordinate lists as well as ID sets).
    """
    out = dict(d)
    for k, v in d.items():
        if k.endswith(".frames"):
            out.setdefault(k[: -len(".frames")], v)
        suffix = "_patch_aligned_doseweighted"
        if k.endswith(suffix):
            out.setdefault(k[: -len(suffix)], v)
    return out


def resolve_train_stem_key(d: dict, stem: str):
    """Look up a raw train-split stem in d, whose keys are normalize_mic_name'd GT names.

    A train stem carries no CryoSPARC import hash, so the raw stem is tried first: a
    genuine name that begins with a date, such as "20210903_106_...", would otherwise be
    mistaken for a hash and stripped by normalize_mic_name (EMPIAR-11183 / 10852).
    normalize_mic_name(stem) is then tried as insurance for a train stem that really did
    carry a hash.
    """
    if stem in d:
        return d[stem]
    return d.get(normalize_mic_name(stem))


def gt_match_labels(cx_px, cy_px, gt_xy, radius):
    """True for each raw query centre with at least one GT particle within `radius`.

    This is not the one-to-one greedy matching the 2D scorer performs -- the only
    question here is whether this raw query's predicted position sits on a real
    particle, so many-to-one is fine.
    """
    if len(gt_xy) == 0:
        return np.zeros(len(cx_px), dtype=bool)
    gt = np.asarray(gt_xy)  # (n_gt, 2)
    d2 = (cx_px[:, None] - gt[None, :, 0]) ** 2 + (cy_px[:, None] - gt[None, :, 1]) ** 2
    return d2.min(axis=1) <= radius ** 2
