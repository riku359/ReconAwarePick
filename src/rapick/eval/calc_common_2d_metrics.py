#!/usr/bin/env python3
"""calc_common_2d_metrics.py -- one particle-wise 2D scorer for every picker (STAR input).

Published pickers ship mutually incompatible evaluation code: some score particle
centres (Topaz, crYOLO), some score pixels (CryoTransformer, CryoSegNet). This script
scores **every picker with the same particle-wise logic**. Its only input is a
GT-aligned STAR
(`_rlnMicrographName / _rlnCoordinateX / _rlnCoordinateY[, _rlnAutopickFigureOfMerit]`);
native output formats (.txt / .cbox / .box) are never touched. Convert those first with
convert_star_to_gt.py.

------------------------------------------------------------------------------
Matching (Topaz's "maximum assignment radius")
------------------------------------------------------------------------------
Topaz is the one publication that states its centre-distance match criterion
explicitly: a prediction matches when its centre lies within **one particle radius
(= 0.5 x diameter)** of an annotated centre (Topaz paper, and CryoTransformer's
supplement "Maximum allowed radius for matching prediction to labeled target"). This
script adopts it:

  * per micrograph, predicted centres and annotated centres are paired **one-to-one**
    at Euclidean distance <= R, with R = diameter / 2 (radius_frac default 0.5).
  * pairs are formed **greedily in order of increasing distance**, independently of any
    score. CryoSegNet emits no confidence and therefore has no score column, so a
    score-ordered greedy match cannot be shared across the four pickers. Ordering by
    distance keeps score out of the matching entirely, so all pickers go through an
    identical procedure.
  * TP = a matched (prediction, annotation) pair; FP = an unmatched prediction;
    FN = an unmatched annotation. One annotation absorbs at most one prediction, so a
    duplicate prediction on the same particle counts as FP.

Metrics: precision / recall / F1, both macro (mean over micrographs, the headline
number) and micro (TP/FP/FN pooled). Score-dependent secondary metrics (AP, best-F1)
are deliberately not reported: the operating point is whatever each picker's STAR
already contains.

------------------------------------------------------------------------------
Coordinates and evaluated micrographs
------------------------------------------------------------------------------
  * GT-aligned STAR and the annotations share a top-left origin, so **no Y flip is
    applied**. To catch a native STAR passed in by mistake, the match count with the
    predictions flipped as H-y is reported alongside and warned about
    (--check-orientation, on by default; only when H can be read from an mrc header).
  * The evaluated set is **every annotated micrograph** -- a fair denominator. If a
    picker returned nothing on a micrograph, all of that micrograph's annotations
    become FN. Predictions on micrographs with no annotation are not scored, because
    there is nothing to score them against.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
  # one entry, one picker
  calc_common_2d_metrics.py --id 10081 \
      --pred $RAPICK_WORK/picks/10081/cryolo.star [--gt GT.star] [--diam 154] [--json]

  # batch: every picker x every entry, resolved from $RAPICK_WORK/picks/
  calc_common_2d_metrics.py --batch [--markdown] [--out-json PATH]

--pred accepts a file, a directory of per-micrograph STAR files, or a glob.
--gt defaults to
$RAPICK_DATA/cryoppp/<ID>/ground_truth/empiar-<ID>_particles_selected.star.

Environment: RAPICK_DATA (annotations, micrographs) and, for --batch only,
RAPICK_WORK (the picks tree). Both are described in docs/CONFIGURATION.md. Neither
has a default: a missing variable is an error naming it.
"""
import argparse
import glob
import json
import os
import re
import struct
import sys


# ---------------------------------------------------------------------------
# Paths (single source of truth, resolved from the environment -- never guessed)
# ---------------------------------------------------------------------------
def _require_env(name, what):
    """Value of environment variable `name`, or a hard error naming it."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set; it must point at {what}. "
            f"See docs/CONFIGURATION.md.")
    return os.path.expanduser(value)


def cryoppp_root():
    """$RAPICK_DATA/cryoppp -- the CryoPPP entries (micrographs + annotations)."""
    return os.path.join(_require_env("RAPICK_DATA", "the downloaded input data"),
                        "cryoppp")


def picks_root():
    """$RAPICK_WORK/picks -- GT-aligned picks, one file or directory per condition."""
    return os.path.join(_require_env("RAPICK_WORK", "the pipeline's output tree"),
                        "picks")


# Nominal particle diameter (px) per EMPIAR entry, from the CryoPPP README's
# "Particle Diameter (px)" column. Match radius R = diameter * radius_frac.
DIAMETERS = {10017: 108, 10028: 224, 10081: 154, 10093: 172,
             10345: 149, 10532: 174, 11056: 164}

# The four EMPIAR entries this study uses.
CORE_IDS = [10081, 10093, 10345, 10532]

PICKERS = ["cryolo", "topaz", "cryotransformer", "cryosegnet"]

# Data leakage: EMPIAR entries contained in a picker's own training data
# (= in-distribution). crYOLO's general model was trained on 10017/10028/10081, and
# the Topaz publication reports 10028. CryoTransformer and CryoSegNet have none.
# The released Topaz general model's training data is undocumented, so an overlap
# cannot be ruled out for any entry -- see the README before reading the
# "avg (leak-free)" row as the paper's greying rule.
LEAK = {"cryolo": {10017, 10028, 10081}, "topaz": {10028},
        "cryotransformer": set(), "cryosegnet": set()}


def picker_pred_path(picker, eid):
    """GT-aligned picks for (picker, entry): a single STAR, else a per-micrograph dir."""
    root = os.path.join(picks_root(), str(eid))
    single = os.path.join(root, f"{picker}.star")
    if os.path.exists(single):
        return single
    return os.path.join(root, picker)          # directory of per-micrograph STAR files


def gt_path_for(eid):
    """Path of the CryoPPP ground-truth selected.star."""
    return os.path.join(cryoppp_root(), str(eid), "ground_truth",
                        f"empiar-{eid}_particles_selected.star")


# ---------------------------------------------------------------------------
# Reading STAR
# ---------------------------------------------------------------------------
def normalize_mic_name(raw):
    """_rlnMicrographName -> comparison key: basename minus a leading <digits>_ and .mrc.

    The annotations carry a CryoSPARC import prefix of random digits, as in
    '>J1/imported/000...371_stack_..._DW.mrc', while a picker's GT-aligned STAR has
    plain 'stack_..._DW.mrc'. Applying the same normalization to both yields one key.
    """
    mic = os.path.basename(raw)
    mic = re.sub(r"^\d+_", "", mic)
    if mic.endswith(".mrc"):
        mic = mic[:-4]
    return mic


def read_star_rows(path):
    """Read the STAR loop_ and return (dict name->column, list of data-row tokens).

    Loops without coordinates (data_optics and friends) are ignored; the loop holding
    _rlnCoordinateX is the one used (same rule as convert_star_to_gt.read_star). The
    columns are bound the moment the _rlnCoordinateX header is seen, binding cur_cols
    and cur_rows by reference. That way a **prediction STAR with zero data rows** (a
    micrograph where the picker picked nothing) still reads correctly, as coordinate
    columns with rows=[]. Snapshotting only once a data row is reached would lose the
    columns of an empty STAR.
    """
    cols, rows = {}, []
    cur_cols, cur_rows, in_loop = {}, [], False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s == "loop_":
                cur_cols, cur_rows, in_loop = {}, [], True
                continue
            if s.startswith("_"):
                m = re.search(r"#(\d+)", s)
                idx = int(m.group(1)) - 1 if m else len(cur_cols)
                name = s.split()[0]
                cur_cols[name] = idx
                if name == "_rlnCoordinateX":
                    cols, rows = cur_cols, cur_rows   # adopt this loop; rows fills in below
                continue
            if s.startswith("data_") or s.startswith("#"):
                in_loop = False
                continue
            if in_loop:
                cur_rows.append(s.split())
    return cols, rows


def load_star_points(path):
    """One STAR -> {mic_key: [(x, y), ...]}; the score column is unused and dropped."""
    cols, rows = read_star_rows(path)
    if "_rlnCoordinateX" not in cols or "_rlnCoordinateY" not in cols:
        raise SystemExit(f"no coordinate columns found in: {path}")
    ix, iy = cols["_rlnCoordinateX"], cols["_rlnCoordinateY"]
    imic = cols.get("_rlnMicrographName")
    # A per-micrograph STAR without _rlnMicrographName takes its file stem as the name.
    default_mic = normalize_mic_name(os.path.basename(path))
    points = {}
    for t in rows:
        if len(t) <= max(ix, iy):
            continue
        try:
            x, y = float(t[ix]), float(t[iy])
        except ValueError:
            continue
        mic = normalize_mic_name(t[imic]) if imic is not None and len(t) > imic else default_mic
        points.setdefault(mic, []).append((x, y))
    return points


def iter_star_files(pred_path):
    """Expand --pred into a list of STAR files (directory / glob / single file)."""
    if os.path.isdir(pred_path):
        return sorted(glob.glob(os.path.join(pred_path, "*.star")))
    if any(c in pred_path for c in "*?["):
        return sorted(glob.glob(pred_path))
    if not os.path.exists(pred_path):
        raise SystemExit(f"prediction STAR not found: {pred_path}")
    return [pred_path]


def load_pred(pred_path):
    """Merge any number of STAR files into {mic_key: [(x, y), ...]}."""
    files = iter_star_files(pred_path)
    if not files:
        raise SystemExit(f"no STAR found under: {pred_path}")
    merged = {}
    for fp in files:
        for mic, pts in load_star_points(fp).items():
            merged.setdefault(mic, []).extend(pts)
    return merged


def mrc_height(eid):
    """ny (= H) from the first CryoPPP .mrc header, for the orientation check.

    None when the micrographs are not available.
    """
    try:
        root = cryoppp_root()
    except SystemExit:
        return None
    files = sorted(glob.glob(os.path.join(root, str(eid), "micrographs", "*.mrc")))
    if not files:
        return None
    try:
        with open(files[0], "rb") as fh:
            _nx, ny = struct.unpack("<2i", fh.read(8))
        return ny
    except (OSError, struct.error):
        return None


# ---------------------------------------------------------------------------
# Distance-ascending greedy matching (score-independent)
# ---------------------------------------------------------------------------
def build_grid(gt_points, radius):
    """Bucket annotated centres into radius-sized cells: {(cx, cy): [gt_index, ...]}."""
    grid = {}
    for i, (gx, gy) in enumerate(gt_points):
        grid.setdefault((int(gx // radius), int(gy // radius)), []).append(i)
    return grid


def match_pairs(gt_points, pred_points, radius):
    """One-to-one (pred_idx, gt_idx) pairs within `radius`, formed in distance order.

    For each prediction the unused annotations in the neighbouring cells are candidates;
    every (distance, pred, gt) edge is then processed in ascending distance and accepted
    when both ends are still free. Score plays no part.

    The pairs themselves are returned, not just their count, because callers also need
    to know *which* annotations were recovered. Use count_true_positives when only the
    TP count matters.
    """
    if not gt_points or not pred_points:
        return []
    radius_sq = radius * radius
    grid = build_grid(gt_points, radius)
    edges = []
    for pi, (px, py) in enumerate(pred_points):
        cx, cy = int(px // radius), int(py // radius)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for gi in grid.get((cx + dx, cy + dy), ()):
                    gx, gy = gt_points[gi]
                    dist_sq = (px - gx) ** 2 + (py - gy) ** 2
                    if dist_sq <= radius_sq:
                        edges.append((dist_sq, pi, gi))
    edges.sort()
    pred_used = [False] * len(pred_points)
    gt_used = [False] * len(gt_points)
    pairs = []
    for _dist_sq, pi, gi in edges:
        if pred_used[pi] or gt_used[gi]:
            continue
        pred_used[pi] = True
        gt_used[gi] = True
        pairs.append((pi, gi))
    return pairs


def count_true_positives(gt_points, pred_points, radius):
    """Number of one-to-one distance-ordered matches within `radius`."""
    return len(match_pairs(gt_points, pred_points, radius))


def flip_y(points, height):
    return [(x, height - y) for (x, y) in points]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def prf(tp, fp, fn):
    """(precision, recall, f1); a zero denominator gives 0.0."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_id(gt, pred, radius):
    """Score one EMPIAR entry, with every annotated micrograph in the denominator.

    Returns a dict: macro/micro P/R/F1, TP/FP/FN, n_mics, n_gt, n_pred_eval,
    covered_mics (annotated micrographs on which the picker returned at least one pick).
    """
    precisions, recalls, f1s = [], [], []
    total_tp = total_fp = total_fn = total_pred = 0
    covered = 0
    for mic, gt_points in gt.items():
        pred_points = pred.get(mic, [])
        if pred_points:
            covered += 1
        total_pred += len(pred_points)
        tp = count_true_positives(gt_points, pred_points, radius)
        fp = len(pred_points) - tp
        fn = len(gt_points) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p, r, f = prf(tp, fp, fn)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    n = len(gt) or 1
    macro_p, macro_r, macro_f1 = sum(precisions) / n, sum(recalls) / n, sum(f1s) / n
    micro_p, micro_r, micro_f1 = prf(total_tp, total_fp, total_fn)
    return {
        "n_mics": len(gt),
        "n_gt": sum(len(v) for v in gt.values()),
        "n_pred_eval": total_pred,
        "covered_mics": covered,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "macro_P": macro_p, "macro_R": macro_r, "macro_F1": macro_f1,
        "micro_P": micro_p, "micro_R": micro_r, "micro_F1": micro_f1,
        "mean_picks_per_mic": total_pred / n,
    }


def orientation_note(gt, pred, radius, height, sample_mics=12):
    """Report the match count with the predictions Y-flipped, to catch a native STAR.

    With GT-aligned input, noflip should be far above flip. A much larger flip count is
    warned about. Rematching every micrograph both ways is expensive, so only the first
    few are sampled -- a frame flip is unmistakable within a handful of micrographs.
    Returns None when the height is unknown (the check cannot be made).
    """
    if height is None:
        return None
    mics = list(gt)[:sample_mics]
    noflip = sum(count_true_positives(gt[m], pred.get(m, []), radius) for m in mics)
    flip = sum(count_true_positives(gt[m], flip_y(pred.get(m, []), height), radius) for m in mics)
    return {"matches_noflip": noflip, "matches_flip": flip, "sample_mics": len(mics),
            "warn_flipped": flip > noflip * 1.2 and flip > noflip + 20}


# ---------------------------------------------------------------------------
# single evaluation / batch
# ---------------------------------------------------------------------------
def run_single(eid, pred_path, gt_star, diam, radius_frac, check_orientation):
    """Score one (picker, entry) pair and return the result dict."""
    if diam is None:
        if eid not in DIAMETERS:
            raise SystemExit(f"no diameter registered for EMPIAR {eid}; pass --diam.")
        diam = DIAMETERS[eid]
    radius = diam * radius_frac
    gt = load_gt_points(gt_star or gt_path_for(eid))
    pred = load_pred(pred_path)
    # Scoring is restricted to annotated micrographs: a prediction on an unannotated
    # micrograph cannot be scored.
    result = evaluate_id(gt, pred, radius)
    result.update(id=eid, diam=diam, radius=round(radius, 1))
    if check_orientation:
        result["orientation"] = orientation_note(gt, pred, radius, mrc_height(eid))
    return result


def load_gt_points(gt_star):
    if not os.path.exists(gt_star):
        raise SystemExit(f"ground-truth STAR not found: {gt_star}")
    return load_star_points(gt_star)


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_batch(ids, pickers, radius_frac, check_orientation):
    """Run picker x entry and return {picker: {id: result, ...}, ...}."""
    table = {}
    for picker in pickers:
        table[picker] = {}
        for eid in ids:
            pred_path = picker_pred_path(picker, eid)
            if not (os.path.isdir(pred_path) or os.path.exists(pred_path)):
                table[picker][eid] = {"error": f"no STAR: {pred_path}"}
                continue
            try:
                table[picker][eid] = run_single(
                    eid, pred_path, None, DIAMETERS.get(eid), radius_frac, check_orientation)
            except SystemExit as e:
                table[picker][eid] = {"error": str(e)}
    return table


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


def print_single(result):
    o = result.get("orientation")
    orient = ""
    if o:
        flag = "  !FLIPPED?" if o["warn_flipped"] else ""
        orient = f" | orient noflip={o['matches_noflip']} flip={o['matches_flip']}{flag}"
    print(f"[{result['id']}] R={result['radius']}px mics={result['n_mics']} "
          f"GT={result['n_gt']} picks={result['n_pred_eval']} "
          f"covered={result['covered_mics']}/{result['n_mics']}{orient}")
    print(f"  macro P={fmt(result['macro_P'])} R={fmt(result['macro_R'])} F1={fmt(result['macro_F1'])}"
          f"   micro P={fmt(result['micro_P'])} R={fmt(result['micro_R'])} F1={fmt(result['micro_F1'])}"
          f"   TP={result['TP']} FP={result['FP']} FN={result['FN']}")


def print_batch_markdown(table, ids):
    """Batch results as a markdown table: macro P/R/F1 plus mean and leak-free mean."""
    for picker in table:
        print(f"\n### {picker}\n")
        print("| EMPIAR | leak | mics | GT | picks | P | R | F1 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        f1s_all, f1s_clean = [], []
        p_all, r_all = [], []
        p_clean, r_clean = [], []
        for eid in ids:
            res = table[picker][eid]
            leak = "*" if eid in LEAK.get(picker, set()) else ""
            if "error" in res:
                print(f"| {eid} | {leak} | - | - | - | - | - | (no STAR) |")
                continue
            print(f"| {eid} | {leak} | {res['n_mics']} | {res['n_gt']} | {res['n_pred_eval']} "
                  f"| {fmt(res['macro_P'])} | {fmt(res['macro_R'])} | {fmt(res['macro_F1'])} |")
            f1s_all.append(res["macro_F1"])
            p_all.append(res["macro_P"])
            r_all.append(res["macro_R"])
            if eid not in LEAK.get(picker, set()):
                f1s_clean.append(res["macro_F1"])
                p_clean.append(res["macro_P"])
                r_clean.append(res["macro_R"])
        print(f"| **avg (all)** |  |  |  |  | {fmt(average(p_all))} | {fmt(average(r_all))} "
              f"| **{fmt(average(f1s_all))}** |")
        if len(f1s_clean) != len(f1s_all):
            print(f"| **avg (leak-free)** |  |  |  |  | {fmt(average(p_clean))} "
                  f"| {fmt(average(r_clean))} | **{fmt(average(f1s_clean))}** |")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="store_true",
                    help="score every picker x every entry from $RAPICK_WORK/picks/")
    ap.add_argument("--id", type=int, help="EMPIAR ID for a single evaluation")
    ap.add_argument("--pred", help="prediction STAR for a single evaluation (file/dir/glob)")
    ap.add_argument("--gt", help="ground-truth STAR (default: resolved under $RAPICK_DATA)")
    ap.add_argument("--diam", type=float, default=None,
                    help="particle diameter (px); default comes from the table")
    ap.add_argument("--radius-frac", type=float, default=0.5,
                    help="match radius = diam * frac")
    ap.add_argument("--ids", type=int, nargs="+", default=None, help="override the batch IDs")
    ap.add_argument("--pickers", nargs="+", default=None, help="override the batch pickers")
    ap.add_argument("--no-check-orientation", action="store_true",
                    help="skip the Y-flip sanity check")
    ap.add_argument("--markdown", action="store_true",
                    help="print the batch result as a markdown table")
    ap.add_argument("--out-json", default=None, help="save the batch result dict as JSON")
    ap.add_argument("--json", action="store_true",
                    help="also print the single-evaluation result as JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    check_orientation = not args.no_check_orientation

    if args.batch:
        ids = args.ids or CORE_IDS
        pickers = args.pickers or PICKERS
        table = run_batch(ids, pickers, args.radius_frac, check_orientation)
        if args.markdown:
            print_batch_markdown(table, ids)
        else:
            for picker in pickers:
                print(f"\n===== {picker} =====")
                for eid in ids:
                    res = table[picker][eid]
                    if "error" in res:
                        print(f"[{eid}] {res['error']}")
                    else:
                        print_single(res)
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(table, f, indent=2)
            print(f"\nsaved: {args.out_json}", file=sys.stderr)
        return

    if not (args.id and args.pred):
        raise SystemExit("a single evaluation needs --id and --pred (or use --batch).")
    result = run_single(args.id, args.pred, args.gt, args.diam,
                        args.radius_frac, check_orientation)
    print_single(result)
    if args.json:
        print("JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
