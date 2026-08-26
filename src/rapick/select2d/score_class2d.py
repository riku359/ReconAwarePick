#!/usr/bin/env python
"""Score the class averages of an existing CryoSPARC class_2D job with CryoSift's CNN.

The job is never re-run. CryoSift's `CryosparcPredictor` reads the job directory
directly -- `J<N>_<iter>_class_averages.mrc` / `.cs` and `J<N>_<iter>_particles.cs` --
and feeds the estimated resolution, the class distribution, the pixel size and three
mass deviations (from the mean, the median and the mode) to the network alongside the
image. Scores are continuous, from 1.0 (best) to 5.0 (worst).

Usage:
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.score_class2d --job J15
    PYTHONPATH=src envs/cryosift/.venv/bin/python \\
        -m rapick.select2d.score_class2d --job J15 --cutoff 3.5 --montage
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from .cryosift_env import (
    DEFAULT_ENV_FILE,
    connect,
    import_upstream,
    parse_model_star,
    read_env,
    require,
    resolve_job_dir,
    resolve_work_dir,
    weights_path,
)

DEFAULT_CUTOFF = 3.5  # upstream settings.ini default for abinitio_cutoff


def latest_iteration_files(job_dir, job_uid):
    """Return the (class_averages.cs, particles.cs) of the last iteration.

    particles.cs is derived from the averages filename rather than globbed because the
    same directory holds a `*_passthrough_particles.cs` that carries no per-particle
    class index, and a glob would pick that one up.
    """
    averages_cs = sorted(Path(job_dir).glob(f"{job_uid}_*_class_averages.cs"))
    if not averages_cs:
        raise SystemExit(f"no class_averages.cs in {job_dir} (the job may be unfinished)")
    averages_cs = averages_cs[-1]
    particles_cs = Path(str(averages_cs).replace("_class_averages.cs", "_particles.cs"))
    return averages_cs, particles_cs


def read_class_metadata(job_dir, job_uid):
    """Read per-class resolution, pixel size and particle count from the job's .cs."""
    averages_cs, particles_cs = latest_iteration_files(job_dir, job_uid)

    classes = np.load(averages_cs)
    particles = np.load(particles_cs, mmap_mode="r")
    counts = np.bincount(
        np.asarray(particles["alignments2D/class"]), minlength=len(classes)
    )

    return {
        "n_classes": len(classes),
        "est_res_A": np.asarray(classes["blob/res_A"]),
        "psize_A": np.asarray(classes["blob/psize_A"]),
        "n_particles": counts,
        "averages_mrc": Path(str(averages_cs).replace(".cs", ".mrc")),
        "n_particles_total": len(particles),
    }


def write_scores_csv(csv_path, scores, meta, cutoff):
    total = meta["n_particles"].sum()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class_idx", "cryosift_score", "keep", "n_particles",
             "class_frac", "est_res_A", "psize_A"]
        )
        for idx, score in enumerate(scores):
            writer.writerow([
                idx,
                f"{score:.3f}",
                int(score < cutoff),
                int(meta["n_particles"][idx]),
                f"{meta['n_particles'][idx] / total:.6f}",
                f"{meta['est_res_A'][idx]:.4f}",
                f"{meta['psize_A'][idx]:.4f}",
            ])


def draw_montage(png_path, averages_mrc, scores, meta, cutoff):
    """Class averages in ascending score order, colour-coded by the cutoff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mrcfile

    with mrcfile.open(averages_mrc, permissive=True) as mrc:
        images = np.array(mrc.data)

    order = np.argsort(scores)
    n_cols = 10
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.72 * n_rows))
    for ax, idx in zip(axes.ravel(), order):
        is_kept = scores[idx] < cutoff
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(
            f"#{idx}  {scores[idx]:.2f}\n{meta['n_particles'][idx]:,}p",
            fontsize=7,
            color="tab:green" if is_kept else "tab:red",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("tab:green" if is_kept else "tab:red")
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")

    # Text drawn into the figure stays ASCII: matplotlib's default font carries no CJK
    # glyphs and renders them as tofu boxes.
    n_kept = int((np.asarray(scores) < cutoff).sum())
    fig.suptitle(
        f"{png_path.stem}  |  sorted by CryoSift score  |  "
        f"green = keep (score < {cutoff}): {n_kept}/{len(order)} classes",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def score_job(job_dir, out_dir, job_uid=None, cutoff=DEFAULT_CUTOFF, montage_name=None):
    """Score a job directory's class averages, write scores.csv, return scores + metadata.

    The single entry point shared by this CLI and the loop in iterate_class2d.py.
    `job_dir` is the CryoSPARC job directory, `out_dir` is where the artifacts go.
    """
    cryosparcpredict, upstream_parse = import_upstream()

    job_dir = Path(job_dir)
    out_dir = Path(out_dir)
    job_uid = job_uid or job_dir.name
    upstream_out = out_dir / "upstream"
    upstream_out.mkdir(parents=True, exist_ok=True)

    model_star = cryosparcpredict(
        input_dir=str(job_dir),
        output_dir=str(upstream_out),
        weights_path=str(weights_path()),
    )
    scores = parse_model_star(model_star)

    # Cross-check against upstream's parser. The two diverge exactly when upstream
    # drops a negative score to 5.0, and feeding that into the selection would discard
    # the best class.
    _, upstream_scores = upstream_parse(model_star)
    mismatch = [
        (i, a, b) for i, (a, b) in enumerate(zip(scores, upstream_scores))
        if abs(a - b) > 1e-6
    ]
    if mismatch:
        print(f"WARN: {len(mismatch)} classes disagree with the upstream parser "
              "(negative scores it fails to match)")
        for i, ours, theirs in mismatch[:10]:
            print(f"      class {i}: {ours:.3f} (correct) vs {theirs:.3f} (upstream)")

    meta = read_class_metadata(job_dir, job_uid)
    if len(scores) != meta["n_classes"]:
        raise SystemExit(
            f"{len(scores)} scores does not match {meta['n_classes']} classes"
        )

    csv_path = out_dir / "scores.csv"
    write_scores_csv(csv_path, scores, meta, cutoff)
    if montage_name:
        draw_montage(out_dir / montage_name, meta["averages_mrc"], scores, meta, cutoff)

    return np.asarray(scores), meta, csv_path


def read_scores_csv(csv_path):
    """Read {class_idx: score} back out of a scores.csv written by score_job."""
    with open(csv_path, newline="") as f:
        return {int(row["class_idx"]): float(row["cryosift_score"])
                for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="class_2D job uid, e.g. J15")
    parser.add_argument("--project", help="CryoSPARC project uid (default: CRYOSPARC_PROJECT)")
    parser.add_argument("--job-dir", help="job directory, given directly (no CryoSPARC connection)")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF,
                        help=f"keep threshold: score < cutoff is kept (default {DEFAULT_CUTOFF})")
    parser.add_argument("--out-root", help="output parent (default $RAPICK_WORK/select2d)")
    parser.add_argument("--env", default=None,
                        help=f"CryoSPARC credentials .env (default {DEFAULT_ENV_FILE})")
    parser.add_argument("--montage", action="store_true", help="also write the montage PNG")
    args = parser.parse_args()

    env = read_env(args.env)
    project = args.project or require(env, "CRYOSPARC_PROJECT", args.env)
    if args.job_dir:
        job_dir = Path(args.job_dir)
    else:
        job_dir = resolve_job_dir(connect(env, args.env), project, args.job)

    out_dir = resolve_work_dir(args.out_root) / f"{project}_{args.job}"

    print(f"job dir : {job_dir}")
    print(f"weights : {weights_path()}")
    print(f"out dir : {out_dir}")

    montage_name = f"{project}_{args.job}_montage.png" if args.montage else None
    scores_arr, meta, csv_path = score_job(job_dir, out_dir, job_uid=args.job,
                                           cutoff=args.cutoff, montage_name=montage_name)

    is_kept = scores_arr < args.cutoff
    kept_particles = int(meta["n_particles"][is_kept].sum())
    print()
    print(f"classes         : {meta['n_classes']}")
    print(f"score min/med/max: {scores_arr.min():.3f} / "
          f"{np.median(scores_arr):.3f} / {scores_arr.max():.3f}")
    print(f"cutoff {args.cutoff}     : keep {int(is_kept.sum())} classes / "
          f"{kept_particles:,} particles "
          f"({kept_particles / meta['n_particles_total']:.1%} of "
          f"{meta['n_particles_total']:,})")
    print(f"csv             : {csv_path}")


if __name__ == "__main__":
    main()
