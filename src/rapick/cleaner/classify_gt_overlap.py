#!/usr/bin/env python3
"""classify_gt_overlap.py -- for the micrographs that carry ground truth, classify the
overlays into success(n==0)/failure(n>0) by symlink, where n is the number of annotated
particles that landed on the anomaly mask, and compare the released arm against the
triangular one. It also counts the area the mask covered (= removed) and plots its
distribution.

Definition of n: the number of ground-truth centres that fall inside mask>=0.5 (the
ground-truth containment count and filter_star_by_contamination.keep_flags are the same
implementation = the centre-pixel test on flipud(mask)).
  official  n = the manifest's n_gt_in_contam (from the full-res predictMask mask)
  triangle  n = the triangular store meta's n_gt_in_tri_contam (from the model-scale triangular mask)
The two are in the same frame, at the same threshold 0.5, against the same ground truth,
so they compare directly (only the resolution differs, full-res vs model scale).

Definition of area: the fraction of pixels with mask>=0.5 = the fraction of the frame that
was removed. The official mask is full-res and the triangular one is at model scale, so
they are compared as fractions rather than pixel counts. A micrograph with no mask npz is
treated as clean (max_mask<0.5), i.e. area 0. Counting is expensive, so it is cached in
<work>/mask_area_frac.csv. An npz missing from the cache is counted and added every run,
so adding masks never leaves any out. --recount-area rebuilds the whole cache.

The micrographs considered are those with n_gt>0 in the triangular store (the ground truth
parses correctly for every entry). The manifest records EMPIAR-11183 as having 0 ground
truth (its micrograph-name normalisation mistakes the date prefix for a CryoSPARC uid), so
for that entry alone the official n is recounted from the full-res mask and the correct
ground truth.

Link targets: <vis>/success/{name}.jpg, <vis>/failure/{n}_{name}.jpg (pointing at the real
files under positive/negative).
Output: <out>/mce_gt_overlap_hist.png (+ statistics json)
Run in an environment with numpy and matplotlib. No GPU needed.

The official mask store, the manifest and both overlay galleries come from the
contamination-detection driver of the research repository, which is not part of this
release. Use --no-copy to skip the overlay classification and produce only the figures.
"""
import argparse, csv, glob, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleaner_env as env
import filter_star_by_contamination as fsc

BLUE, ORANGE = "#2a78d6", "#eb6834"     # dataviz categorical slot 1/2 (CVD-safe)
DEEP_THR = 0.5
GT_UID = re.compile(r"^\d{12,}_")       # the uid prefix on CryoPPP ground truth (12+ digits only)


def stem(mic):
    return os.path.splitext(mic)[0]


# ---------------------------------------------------------------- records

def manifest_rows(manifest_path):
    with open(manifest_path) as f:
        return list(csv.DictReader(f))


def triangle_records(mask_root, vis_root):
    """(eid, mic) -> record, from the triangular store's meta. The ground-truth parse is
    correct for every entry."""
    recs = {}
    for f in glob.glob(os.path.join(mask_root, "*", "*_tri.npz")):
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        kind = "positive" if int(m.get("has_anomaly", 1)) != 0 else "negative"
        src = os.path.join(vis_root, kind, m["dist_class"],
                           m["empiar_id"], stem(m["micrograph"]) + "_overlay.jpg")
        recs[(m["empiar_id"], m["micrograph"])] = {
            "eid": m["empiar_id"], "mic": m["micrograph"],
            "n_gt": int(m.get("n_gt", 0)), "n": int(m.get("n_gt_in_tri_contam", 0)), "src": src}
    return recs


def recount_default_n(eid, rows_of_eid, official_root):
    """Recount the official n from the full-res mask and the correct ground truth (for an
    entry whose manifest ground truth is broken)."""
    _, gt_rows = fsc.parse_star(env.ground_truth_star(eid))
    gt = {}
    for (_, mic, x, y) in gt_rows:
        gt.setdefault(GT_UID.sub("", mic), []).append((x, y))

    n_by_mic = {}
    for r in rows_of_eid:
        picks = gt.get(r["micrograph"], [])
        if not picks:
            continue
        if str(r["has_anomaly"]).strip() == "0":      # a clean micrograph carries no mask npz
            n_by_mic[r["micrograph"]] = 0
            continue
        npz = os.path.join(official_root, eid, stem(r["micrograph"]) + "_mask.npz")
        mask = np.load(npz)["mask"].astype(np.float32)
        n_by_mic[r["micrograph"]] = sum(1 for keep in fsc.keep_flags(mask, picks, DEEP_THR) if not keep)
    return n_by_mic


def default_records(gt_keys, manifest_path, official_root, vis_root):
    """Build the official-side records for gt_keys (the ground-truth-bearing micrographs the
    triangular store names)."""
    rows = manifest_rows(manifest_path)
    by_key = {(r["empiar_id"], r["micrograph"]): r for r in rows}
    fixed = recount_default_n("11183", [r for r in rows if r["empiar_id"] == "11183"], official_root)

    recs = {}
    for key in gt_keys:
        eid, mic = key
        r = by_key.get(key)
        if r is None:
            continue
        n = fixed[mic] if eid == "11183" else int(float(r["n_gt_in_contam"] or 0))
        recs[key] = {"eid": eid, "mic": mic, "n_gt": int(float(r["n_gt"] or 0)), "n": n,
                     "src": os.path.join(vis_root, r["overlay_path"])}
    return recs


# ---------------------------------------------------------------- mask area

def _area_frac(job):
    """The 'fraction of pixels with mask>=0.5' for one mask npz. Runs in a worker, so it is
    a top-level function."""
    path, key = job
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    mask = z[key]
    return (meta["empiar_id"], meta["micrograph"], float((mask >= DEEP_THR).mean()),
            int(mask.shape[0]), int(mask.shape[1]))


AREA_FIELDS = ["method", "empiar_id", "micrograph", "area_frac", "mask_h", "mask_w"]


def mask_npz_jobs(official_root, mask_root):
    """Every mask npz on disk -> (method, path, the key inside the npz)."""
    return ([("default", p, "mask")
             for p in glob.glob(os.path.join(official_root, "*", "*_mask.npz"))] +
            [("triangle", p, "tri")
             for p in glob.glob(os.path.join(mask_root, "*", "*_tri.npz"))])


def npz_cache_key(job):
    """(method, eid, micrograph name without extension) from an npz path.
    <...>/<eid>/<stem>_mask.npz"""
    method, path, _ = job
    return (method, os.path.basename(os.path.dirname(path)),
            os.path.basename(path).rsplit("_", 1)[0])


def read_area_cache(area_cache):
    if not os.path.exists(area_cache):
        return {}
    with open(area_cache) as f:
        return {(r["method"], r["empiar_id"], r["micrograph"]): r for r in csv.DictReader(f)}


def write_area_cache(area_cache, rows):
    tmp = area_cache + ".part"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, AREA_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    os.replace(tmp, area_cache)
    print("[area] -> %s (%d rows)" % (area_cache, len(rows)))


def load_area_cache(area_cache, official_root, mask_root, workers, recount):
    """Read the area cache and count only the npz that have not been counted yet.

    area_array treats a micrograph missing from the cache as area 0 (= a clean micrograph
    with no mask), so an uncounted one is indistinguishable from zero contamination.
    Counting the difference prevents that confusion. Counting everything takes tens of
    minutes; counting only the difference costs time proportional to what was added.
    """
    rows = {} if recount else read_area_cache(area_cache)
    counted = {(method, eid, stem(mic)) for (method, eid, mic) in rows}
    todo = [job for job in mask_npz_jobs(official_root, mask_root) if npz_cache_key(job) not in counted]

    if todo:
        print("[area] counting %d uncached mask npz with %d workers ..." % (len(todo), workers))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_area_frac, [(p, k) for _, p, k in todo], chunksize=8)
            for done, ((method, _, _), (eid, mic, frac, h, wd)) in enumerate(zip(todo, results), 1):
                rows[(method, eid, mic)] = dict(zip(AREA_FIELDS,
                                                    [method, eid, mic, "%.8g" % frac, h, wd]))
                if done % 500 == 0:
                    print("[area] %d/%d" % (done, len(todo)), flush=True)
        write_area_cache(area_cache, rows)

    return {key: float(r["area_frac"]) for key, r in rows.items()}


def area_array(area, method, keys):
    """The area fractions in the order of keys. A micrograph with no npz is clean
    (max_mask<0.5), so 0."""
    return np.array([area.get((method, eid, mic), 0.0) for eid, mic in keys])


# ---------------------------------------------------------------- classify & stats

def classify_link(recs, vis_root, label):
    """Link success/failure to the overlays with relative symlinks.

    Real copies would duplicate the positive/negative trees (official 4.3G + triangle 4.1G)
    and leave stale duplicates behind every time the overlays are redrawn. A symlink looks
    the same to the classification and a redraw is reflected straight away.
    """
    succ, fail = os.path.join(vis_root, "success"), os.path.join(vis_root, "failure")
    os.makedirs(succ, exist_ok=True)
    os.makedirs(fail, exist_ok=True)
    seen, missing, coll = set(), 0, 0
    for r in recs:
        name, n = stem(r["mic"]), r["n"]
        dst = os.path.join(succ, name + ".jpg") if n == 0 else os.path.join(fail, "%d_%s.jpg" % (n, name))
        if os.path.basename(dst) in seen:        # basename collision across entries -> insert the id
            coll += 1
            dst = (os.path.join(succ, "%s_%s.jpg" % (r["eid"], name)) if n == 0
                   else os.path.join(fail, "%d_%s_%s.jpg" % (n, r["eid"], name)))
        seen.add(os.path.basename(dst))
        if not os.path.exists(r["src"]):
            missing += 1
            continue
        if os.path.lexists(dst):                 # replace an old real copy / stale link
            os.remove(dst)
        os.symlink(os.path.relpath(r["src"], os.path.dirname(dst)), dst)
    print("[%s] linked overlays: missing_src=%d collisions=%d" % (label, missing, coll))


def summ(ns, ngt, areas):
    f = ns[ns > 0]
    a = areas[areas > 0]
    return dict(mics=int(len(ns)), success=int((ns == 0).sum()), failure=int((ns > 0).sum()),
                fail_rate=float((ns > 0).mean()), total_gt=int(ngt.sum()), total_overlap=int(ns.sum()),
                overlap_frac=float(ns.sum() / max(1, ngt.sum())),
                mean_fail=float(f.mean()) if len(f) else 0.0,
                median_fail=float(np.median(f)) if len(f) else 0.0, max_n=int(ns.max()) if len(ns) else 0,
                masked_mics=int((areas > 0).sum()),
                mean_area=float(areas.mean()), median_area_masked=float(np.median(a)) if len(a) else 0.0,
                mean_area_masked=float(a.mean()) if len(a) else 0.0, max_area=float(areas.max()),
                area_ge10pct=int((areas >= 0.10).sum()), area_ge30pct=int((areas >= 0.30).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-copy", action="store_true", help="skip the overlay classification symlinks and only build the figures")
    ap.add_argument("--recount-area", action="store_true", help="rebuild the area cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mask-root", default=None,
                    help="triangular mask store (default: $RAPICK_WORK/masks)")
    ap.add_argument("--official-root", default=None,
                    help="store of released-arm masks (default: $RAPICK_WORK/masks_official)")
    ap.add_argument("--manifest", default=None,
                    help="per-micrograph manifest CSV of the released arm (default: $RAPICK_WORK/manifest.csv)")
    ap.add_argument("--official-vis-root", default=None,
                    help="overlay gallery of the released arm (default: $RAPICK_WORK/overlays/official)")
    ap.add_argument("--triangular-vis-root", default=None,
                    help="overlay gallery of the triangular arm (default: $RAPICK_WORK/overlays/triangular)")
    ap.add_argument("--area-cache", default=None,
                    help="mask-area cache CSV (default: $RAPICK_WORK/mask_area_frac.csv)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: $RAPICK_WORK/figures/gt_overlap)")
    args = ap.parse_args()

    mask_root = args.mask_root or env.mask_store_root()
    official_root = args.official_root or env.official_mask_store_root()
    manifest = args.manifest or os.path.join(env.work_root(), "manifest.csv")
    official_vis = args.official_vis_root or os.path.join(env.work_root(), "overlays", "official")
    triangular_vis = args.triangular_vis_root or os.path.join(env.work_root(), "overlays", "triangular")
    area_cache = args.area_cache or os.path.join(env.work_root(), "mask_area_frac.csv")
    out = args.out or os.path.join(env.work_root(), "figures", "gt_overlap")

    os.makedirs(out, exist_ok=True)
    tri = triangle_records(mask_root, triangular_vis)
    keys = sorted(k for k, r in tri.items() if r["n_gt"] > 0)
    dflt = default_records(keys, manifest, official_root, official_vis)
    keys = [k for k in keys if k in dflt]                 # compare only the micrographs both methods cover
    print("[set] GT-bearing micrographs compared: %d" % len(keys))

    d_ns = np.array([dflt[k]["n"] for k in keys])
    t_ns = np.array([tri[k]["n"] for k in keys])
    n_gt = np.array([tri[k]["n_gt"] for k in keys])

    area = load_area_cache(area_cache, official_root, mask_root, args.workers, args.recount_area)
    d_area = area_array(area, "default", keys)
    t_area = area_array(area, "triangle", keys)

    if not args.no_copy:
        classify_link([dflt[k] for k in keys], official_vis, "default")
        classify_link([tri[k] for k in keys], triangular_vis, "triangle")

    ds, ts = summ(d_ns, n_gt, d_area), summ(t_ns, n_gt, t_area)

    # ---- figure: (A) over-removed GT particles, (B) the n distribution over failures,
    #      (C) the distribution of the removed area ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 4.6))
    # A: the total number of ground-truth particles lost. Counted per micrograph, triangle
    # looks worse because it has more failure micrographs, but what affects the 3D
    # resolution is the total number of good particles thrown away, so counting particles
    # reverses the order.
    x = np.arange(2)
    lost = [ds["total_overlap"], ts["total_overlap"]]
    ax1.bar(x, lost, 0.5, color=[BLUE, ORANGE])
    ax1.set_xticks(x)
    ax1.set_xticklabels(["default (official)", "triangle"])
    ax1.set_ylabel("GT particles removed by mask")
    ax1.set_title("A. over-removed GT particles (of %s GT in %d mics)" % (f"{ds['total_gt']:,}", len(keys)))
    ax1.set_ylim(0, max(lost) * 1.22)
    for xi, s in zip(x, (ds, ts)):
        ax1.text(xi, s["total_overlap"], "%s\n%.3f%% of GT" % (f"{s['total_overlap']:,}",
                                                              s["overlap_frac"] * 100),
                 ha="center", va="bottom", fontsize=9, color="#0b0b0b", linespacing=1.4)

    # B: failure n distribution (n>0)
    df, tf = d_ns[d_ns > 0], t_ns[t_ns > 0]
    mx = int(max(df.max() if len(df) else 1, tf.max() if len(tf) else 1))
    bins = np.linspace(1, mx, 31)
    ax2.hist(df, bins=bins, color=BLUE, alpha=0.55, label="default (official)")
    ax2.hist(tf, bins=bins, color=ORANGE, alpha=0.55, label="triangle")
    ax2.set_xlabel("n = GT particles overlapping mask (per mic)")
    ax2.set_ylabel("failure micrographs")
    ax2.set_title("B. distribution of n over failures (n>0, log y)")
    ax2.set_yscale("log")
    ax2.legend(frameon=False)

    # C: the masked-area distribution. The area spans 6 orders of magnitude, so it is drawn
    # on a log x axis. Micrographs below the lower end of 1e-4% (a few tens of pixels at
    # full resolution) are pushed into the first bin. np.histogram silently discards
    # anything out of range, so the number pushed in is noted on the figure. Micrographs
    # with zero area do not fit on a log axis either, so they are noted the same way.
    AREA_MIN_PCT = 1e-4
    da, ta = d_area[d_area > 0] * 100, t_area[t_area > 0] * 100
    n_clipped = int((da < AREA_MIN_PCT).sum() + (ta < AREA_MIN_PCT).sum())
    abins = np.logspace(np.log10(AREA_MIN_PCT), 2, 61)
    ax3.hist(np.clip(da, AREA_MIN_PCT, None), bins=abins, color=BLUE, alpha=0.55,
             label="default (official)")
    ax3.hist(np.clip(ta, AREA_MIN_PCT, None), bins=abins, color=ORANGE, alpha=0.55,
             label="triangle")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_ylim(0.7, ax3.get_ylim()[1] * 8)        # leave height so the top-right note does not overlap the peak
    ax3.set_xlabel("masked area per mic (% of frame, mask $\\geq$ 0.5)")
    ax3.set_ylabel("micrographs")
    ax3.set_title("C. removed area (log x, log y)")
    ax3.legend(frameon=False, loc="upper left")
    ax3.text(0.98, 0.97,
             "default / triangle\n"
             "median of masked mics: %.2f%% / %.2f%%\n"
             "mean over all %d mics: %.2f%% / %.2f%%\n"
             "$\\geq$30%% of frame: %d / %d mics\n"
             "zero-area mics: %d / %d   (%d clipped into first bin)"
             % (ds["median_area_masked"] * 100, ts["median_area_masked"] * 100, len(keys),
                ds["mean_area"] * 100, ts["mean_area"] * 100,
                ds["area_ge30pct"], ts["area_ge30pct"],
                len(keys) - ds["masked_mics"], len(keys) - ts["masked_mics"], n_clipped),
             transform=ax3.transAxes, ha="right", va="top", fontsize=8, color="#5b5b57",
             linespacing=1.5)

    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    png = os.path.join(out, "mce_gt_overlap_hist.png")
    fig.savefig(png, dpi=140, facecolor="white")
    print("[fig] ->", png)

    # Emit A on its own for a slide: three panels side by side get too small on one slide.
    fig_a, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.bar(x, lost, 0.5, color=[BLUE, ORANGE])
    ax.set_xticks(x)
    ax.set_xticklabels(["default (official)", "triangle"])
    ax.set_ylabel("GT particles removed by mask")
    ax.set_title("over-removed GT particles (%s GT, %d mics)" % (f"{ds['total_gt']:,}", len(keys)))
    ax.set_ylim(0, max(lost) * 1.22)
    for xi, s in zip(x, (ds, ts)):
        ax.text(xi, s["total_overlap"], "%s\n%.3f%% of GT" % (f"{s['total_overlap']:,}",
                                                             s["overlap_frac"] * 100),
                ha="center", va="bottom", fontsize=9, color="#0b0b0b", linespacing=1.4)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig_a.tight_layout()
    png_a = os.path.join(out, "mce_gt_overlap_total.png")
    fig_a.savefig(png_a, dpi=200, facecolor="white")
    print("[fig] ->", png_a)

    paired = {"tri_lt_def": int((t_ns < d_ns).sum()), "tri_gt_def": int((t_ns > d_ns).sum()),
              "eq": int((t_ns == d_ns).sum()),
              "def_only_fail": int(((d_ns > 0) & (t_ns == 0)).sum()),
              "tri_only_fail": int(((t_ns > 0) & (d_ns == 0)).sum()),
              "area_tri_lt_def": int((t_area < d_area).sum()),
              "area_tri_gt_def": int((t_area > d_area).sum())}
    with open(os.path.join(out, "mce_gt_overlap_stats.json"), "w") as jf:
        json.dump({"default": ds, "triangle": ts, "paired": paired}, jf, indent=2)
    print("DEFAULT ", ds)
    print("TRIANGLE", ts)
    print("PAIRED  ", paired)


if __name__ == "__main__":
    main()
