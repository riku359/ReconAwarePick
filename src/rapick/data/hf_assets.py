#!/usr/bin/env python3
"""Upload and download the artifacts this project hosts on Hugging Face.

Everything else the pipeline needs is either public upstream (CryoTransformer,
MicrographCleaner, CryoSift, all fetched by scripts/setup.sh) or downloadable from
EMPIAR and CryoPPP. These are the exceptions: artifacts this project produced, with no
upstream source, expensive enough to regenerate that they are worth publishing.

  checkpoint   theta_0, the head-repaired CryoTransformer checkpoint (~870 MB) that
               every condition in the paper starts from, and that each round of the
               loop restarts from. Sec. S2 describes the repair;
               src/rapick/picker/README.md describes how to redo it.
  cleaner-data The full-set picks that survive contamination masking, for all four
               entries. A few MB of STAR text each, but reproducing one needs a
               MicrographCleaner inference pass over the entire micrograph set.
  masks        The precomputed triangular-blend contamination masks for all four
               entries at full-set scale. The annotated 300 micrographs are a strict
               subset of the full deposition and share their filenames, so this one
               store serves both scales. Downloading them means never having to build
               MicrographCleaner's TensorFlow environment: applying a cached mask to a
               STAR file needs only numpy and opencv. About 6,070 small .npz files, so
               whole per-entry folders move at once rather than file by file.

  loop         The round-1 fine-tuned checkpoints, one per entry and per arm. `fb` is
               the paper's method, so those are what the Ours row picks with; without
               them that row can only be reproduced by re-running the loop, at about
               two hours per round per entry. `fb_gt` is the perfect-teacher upper
               bound of Table 7's lower row.

  picks        The four pickers' candidates over the whole deposition, in the
               GT-aligned format every stage downstream reads. Table 2 and Table S2
               are the only places they are needed, and having them means not
               installing crYOLO, Topaz or CryoSegNet, none of which is easy to
               build and one of which is not redistributable.

The remote repo layout (the HF-side paths below) is independent of local layout and
does not change when a site moves things around locally: --data-root / --experiments-root
only say where these commands read from / write to on THIS machine.

Needs `pip install huggingface_hub` and `hf auth login` (older huggingface_hub versions:
`huggingface-cli login`; or an HUGGING_FACE_HUB_TOKEN env var) first; neither is a
project dependency of any of this repo's per-tool venvs, so run this with whatever
interpreter has huggingface_hub.

Usage:
    # from the machine that has the local files (upload)
    python src/rapick/data/hf_assets.py upload-checkpoint --repo <user>/recon-aware-pick-weights \
        --data-root "$RAPICK_DATA"
    python src/rapick/data/hf_assets.py upload-cleaner-data --repo <user>/recon-aware-pick-data \
        --experiments-root "$RAPICK_WORK"
    python src/rapick/data/hf_assets.py upload-masks --repo <user>/recon-aware-pick-data \
        --experiments-root "$RAPICK_WORK"

    # on the new site (download)
    python src/rapick/data/hf_assets.py download \
        --repo-weights <user>/recon-aware-pick-weights \
        --repo-data <user>/recon-aware-pick-data \
        --data-root "$RAPICK_DATA" \
        --experiments-root "$RAPICK_WORK" --with-masks
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Remote (HF Hub) layout only -- fixed once published, independent of local paths.
CHECKPOINT_REL = Path("cryotransformer/eos_coef=0.1(default)")
CHECKPOINT_FILES = ["CryoTransformer_head_repaired.pth", "CryoTransformer_head_repaired.json"]

# The fine-tuned checkpoints one round of the loop delivers, per entry and per arm.
# `fb` is the paper's method (Table 2's Ours row, Table 4's fb row); `fb_gt` is the
# perfect-teacher upper bound of Table 7's lower row. Round 1 is what the paper
# reports, so that is what is published.
LOOP_ARMS = ("fb", "fb_gt")
LOOP_REL = "weights/loop/{arm}/round1/empiar_{eid}.pth"

# The four pickers' candidates over the whole deposition, in the GT-aligned format
# every downstream stage reads. With these, Table 2 and Table S2 can be reproduced
# without installing crYOLO, Topaz or CryoSegNet, none of which is easy to build and
# one of which is not redistributable.
PICKERS = ("cryotransformer", "cryolo", "topaz", "cryosegnet")
PICKS_REL = "picks/full/{eid}/{picker}.star"

# The 4 IDs experiments 1-4 run on, and the 3 files filter_star_triangular.py writes
# per ID (decisions_tri.jsonl is a resume checkpoint, not needed downstream -- skipped).
CLEANER_IDS = ["10081", "10093", "10345", "10532"]
CLEANER_STAR = "cryotransformer_clean_tri.star"      # the filter's own output name
CLEANER_FILES = [CLEANER_STAR, "summary_tri.json", "filter_stats_tri.csv"]

# What the filter's output is published as locally: the picker's name plus the stage it
# has been through, which is how every driver and every dataset config addresses it.
MASKED_PICKS = "cryotransformer_mask.star"

# All 4 IDs and which half of MicrographCleaner's own in/out-of-distribution split
# their masks sit under. MicrographCleaner's training set includes 10081 and 10093
# but not 10345 or 10532, which is why the paper attributes the mask's failure on
# 10532 to domain shift rather than to the pipeline.
MASK_IDS = {"10081": "in_distribution", "10093": "in_distribution",
            "10345": "out_of_distribution", "10532": "out_of_distribution"}
MASK_SUBDIR = "triangle_mask_overlay/anomaly_mask_npy"   # remote only; local is masks/<id>/


def _hub():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("error: huggingface_hub not installed -- pip install huggingface_hub")
    return HfApi()


def cmd_upload_checkpoint(args):
    api = _hub()
    root = Path(args.data_root).expanduser() / "checkpoints"
    missing = [f for f in CHECKPOINT_FILES if not (root / f).is_file()]
    if missing:
        sys.exit(f"error: missing under {root}: {missing}")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for f in CHECKPOINT_FILES:
        src = root / f
        dest = f"weights/{CHECKPOINT_REL}/{f}"
        print(f"[upload] {src} -> {args.repo}:{dest} ({src.stat().st_size / 1e6:.0f} MB)")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                        repo_id=args.repo, repo_type="model")
    print(f"[done] https://huggingface.co/{args.repo}")


def cmd_upload_loop_checkpoints(args):
    """Publish the round-1 checkpoints of one loop arm, one file per entry.

    They are what the `fb` condition picks with, so without them the paper's headline
    row can only be reproduced by re-running the loop, which is about two hours per
    round per entry.
    """
    api = _hub()
    root = Path(args.models_root).expanduser()
    ids = args.ids.split(",") if args.ids else list(CLEANER_IDS)

    found = []
    for eid in ids:
        src = Path(str(root).replace("{eid}", eid))
        if not src.is_file():
            sys.exit(f"error: no checkpoint for {eid} at {src}")
        found.append((eid, src))

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for eid, src in found:
        dest = LOOP_REL.format(arm=args.arm, eid=eid)
        print(f"[upload] {src} -> {args.repo}:{dest} ({src.stat().st_size / 1e6:.0f} MB)")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                        repo_id=args.repo, repo_type="model")
    print(f"[done] https://huggingface.co/{args.repo}")


def cmd_upload_picks(args):
    """Publish the pickers' full-set candidates, one STAR per entry and picker."""
    api = _hub()
    root = Path(args.picks_root).expanduser()
    ids = args.ids.split(",") if args.ids else list(CLEANER_IDS)
    pickers = args.pickers.split(",") if args.pickers else list(PICKERS)

    found = []
    for eid in ids:
        for picker in pickers:
            src = root / eid / f"{picker}.star"
            if not src.is_file():
                print(f"[skip] {eid}/{picker}: not at {src}")
                continue
            found.append((eid, picker, src))
    if not found:
        sys.exit(f"error: no STAR files under {root}")

    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for eid, picker, src in found:
        dest = PICKS_REL.format(eid=eid, picker=picker)
        print(f"[upload] {src} -> {args.repo}:{dest} ({src.stat().st_size / 1e6:.0f} MB)")
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                        repo_id=args.repo, repo_type="dataset")
    print(f"[done] https://huggingface.co/datasets/{args.repo}")


def cmd_upload_cleaner_data(args):
    api = _hub()
    root = Path(args.experiments_root).expanduser()
    ids = args.ids.split(",") if args.ids else CLEANER_IDS
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for eid in ids:
        id_dir = root / "picks" / eid
        local_of = {f: (MASKED_PICKS if f == CLEANER_STAR else f) for f in CLEANER_FILES}
        missing = [f for f in CLEANER_FILES if not (id_dir / local_of[f]).is_file()]
        if missing:
            print(f"[skip] {eid}: missing {[local_of[f] for f in missing]} under {id_dir}")
            continue
        for f in CLEANER_FILES:
            src = id_dir / local_of[f]
            dest = f"fullset_filter/{eid}/cryotransformer/{f}"
            print(f"[upload] {src} -> {args.repo}:{dest} ({src.stat().st_size / 1e3:.0f} KB)")
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                            repo_id=args.repo, repo_type="dataset")
    print(f"[done] https://huggingface.co/datasets/{args.repo}")


def cmd_upload_masks(args):
    api = _hub()
    root = Path(args.experiments_root).expanduser() / "masks"
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for eid, dist in MASK_IDS.items():
        mask_dir = root / eid
        n = len(list(mask_dir.glob("*.npz"))) if mask_dir.is_dir() else 0
        if not n:
            print(f"[skip] {eid}: no .npz under {mask_dir}")
            continue
        dest = f"{MASK_SUBDIR}/{dist}/{eid}"
        print(f"[upload] {mask_dir} ({n} files) -> {args.repo}:{dest}/")
        api.upload_folder(folder_path=str(mask_dir), path_in_repo=dest,
                          repo_id=args.repo, repo_type="dataset",
                          allow_patterns=["*.npz"])
    print(f"[done] https://huggingface.co/datasets/{args.repo}")


def cmd_download(args):
    from huggingface_hub import hf_hub_download

    if args.repo_weights:
        dest_root = Path(args.data_root).expanduser() / "checkpoints"
        dest_root.mkdir(parents=True, exist_ok=True)
        for f in CHECKPOINT_FILES:
            dest = dest_root / f
            got = hf_hub_download(args.repo_weights, f"weights/{CHECKPOINT_REL}/{f}",
                                  repo_type="model")
            dest.write_bytes(Path(got).read_bytes())
            print(f"[got] {dest}")

        # The loop's round-1 checkpoints land next to theta_0, under a name that says
        # which arm and which entry, so the picker can be pointed at one directly.
        arm = getattr(args, "with_loop_checkpoints", None)
        if arm:
            ids = args.ids.split(",") if args.ids else CLEANER_IDS
            for eid in ids:
                dest = dest_root / f"loop_{arm}_round1_empiar_{eid}.pth"
                got = hf_hub_download(args.repo_weights,
                                      LOOP_REL.format(arm=arm, eid=eid), repo_type="model")
                dest.write_bytes(Path(got).read_bytes())
                print(f"[got] {dest}")

    if args.repo_data:
        experiments_root = Path(args.experiments_root).expanduser()
        ids = args.ids.split(",") if args.ids else CLEANER_IDS

        if getattr(args, "with_picks", False):
            for eid in ids:
                picks_dir = experiments_root / "picks" / eid
                picks_dir.mkdir(parents=True, exist_ok=True)
                for picker in PICKERS:
                    got = hf_hub_download(args.repo_data,
                                          PICKS_REL.format(eid=eid, picker=picker),
                                          repo_type="dataset")
                    (picks_dir / f"{picker}.star").write_bytes(Path(got).read_bytes())
                print(f"[got] {picks_dir}/{{{','.join(PICKERS)}}}.star")
        # The masked picks go where scripts/contamination_removal.sh would have written
        # them, under the name that says which stages they have been through. Landing
        # them anywhere else is the same as not fetching them: nothing downstream looks
        # for a STAR outside picks/<id>/.
        for eid in ids:
            picks_dir = experiments_root / "picks" / eid
            picks_dir.mkdir(parents=True, exist_ok=True)
            for f in CLEANER_FILES:
                got = hf_hub_download(args.repo_data, f"fullset_filter/{eid}/cryotransformer/{f}",
                                      repo_type="dataset")
                # The STAR is published remotely under the filter's own output name and
                # lands here under the name the pipeline reads; the other two are
                # diagnostics and keep theirs.
                local = MASKED_PICKS if f == CLEANER_STAR else f
                (picks_dir / local).write_bytes(Path(got).read_bytes())
            print(f"[got] {picks_dir}/{MASKED_PICKS} and its two diagnostics files")

    if args.with_masks:
        from huggingface_hub import snapshot_download
        if not args.repo_data:
            sys.exit("--with-masks needs --repo-data")
        experiments_root = Path(args.experiments_root).expanduser()
        experiments_root.mkdir(parents=True, exist_ok=True)
        # Remote layout is MASK_SUBDIR/<dist>/<id>/*.npz, where <dist> records which half
        # of MicrographCleaner's own in/out-of-distribution split an entry falls in.
        # snapshot_download mirrors that under a staging dir; the <dist> level is then
        # dropped, because every stage that reads a mask reads $RAPICK_WORK/masks/<id>
        # (rapick.loop.paths.mask_dir, scripts/contamination_removal.sh --masks) and a
        # store one level deeper is a store nothing finds.
        staging = experiments_root / "_hf_masks_staging"
        snapshot_download(args.repo_data, repo_type="dataset",
                          allow_patterns=[f"{MASK_SUBDIR}/*"], local_dir=str(staging))
        masks_root = experiments_root / "masks"
        for eid, dist in MASK_IDS.items():
            src = staging / MASK_SUBDIR / dist / eid
            if not src.is_dir():
                continue
            dst = masks_root / eid
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"[got] masks under {dst}")
        shutil.rmtree(staging, ignore_errors=True)

    print("[done] the files landed under --data-root and --experiments-root above; "
          "point RAPICK_DATA and RAPICK_WORK at them (docs/CONFIGURATION.md).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    up_ckpt = sub.add_parser("upload-checkpoint", help="upload the head-repaired checkpoint")
    up_ckpt.add_argument("--repo", required=True, help="e.g. <user>/recon-aware-pick-weights")
    up_ckpt.add_argument("--data-root", required=True,
                         help="$RAPICK_DATA on this machine "
                              "(checkpoint files live under <data-root>/checkpoints/)")
    up_ckpt.add_argument("--private", action="store_true")
    up_ckpt.set_defaults(func=cmd_upload_checkpoint)

    up_loop = sub.add_parser("upload-loop-checkpoints",
                             help="upload one loop arm's round-1 checkpoints, one per entry")
    up_loop.add_argument("--repo", required=True, help="e.g. <user>/recon-aware-pick-weights")
    up_loop.add_argument("--arm", required=True, choices=list(LOOP_ARMS),
                         help="fb is the paper's method; fb_gt is Table 7's upper bound")
    up_loop.add_argument("--models-root", required=True,
                         help="path to one checkpoint with {eid} standing in for the entry, "
                              "e.g. /data/loop/empiar_{eid}/models/model_1.pth")
    up_loop.add_argument("--ids", help=f"comma-separated, default {','.join(CLEANER_IDS)}")
    up_loop.add_argument("--private", action="store_true")
    up_loop.set_defaults(func=cmd_upload_loop_checkpoints)

    up_picks = sub.add_parser("upload-picks",
                              help="upload the pickers' full-set candidates, one STAR each")
    up_picks.add_argument("--repo", required=True, help="e.g. <user>/recon-aware-pick-data")
    up_picks.add_argument("--picks-root", required=True,
                          help="directory holding <entry>/<picker>.star")
    up_picks.add_argument("--ids", help=f"comma-separated, default {','.join(CLEANER_IDS)}")
    up_picks.add_argument("--pickers", help=f"comma-separated, default {','.join(PICKERS)}")
    up_picks.add_argument("--private", action="store_true")
    up_picks.set_defaults(func=cmd_upload_picks)

    up_data = sub.add_parser("upload-cleaner-data", help="upload the 4 IDs' cleaner stars")
    up_data.add_argument("--repo", required=True, help="e.g. <user>/recon-aware-pick-data")
    up_data.add_argument("--experiments-root", required=True,
                         help="$RAPICK_WORK on this machine")
    up_data.add_argument("--ids", help=f"comma-separated, default {','.join(CLEANER_IDS)}")
    up_data.add_argument("--private", action="store_true")
    up_data.set_defaults(func=cmd_upload_cleaner_data)

    up_masks = sub.add_parser("upload-masks",
                              help="upload all 4 entries' full-set triangular masks "
                                   "(the loop's per-round filter, and the full-set masked picks)")
    up_masks.add_argument("--repo", required=True, help="e.g. <user>/recon-aware-pick-data")
    up_masks.add_argument("--experiments-root", required=True,
                          help="$RAPICK_WORK on this machine")
    up_masks.add_argument("--private", action="store_true")
    up_masks.set_defaults(func=cmd_upload_masks)

    dl = sub.add_parser("download", help="download onto the new site")
    dl.add_argument("--repo-weights", help="checkpoint repo, e.g. <user>/recon-aware-pick-weights")
    dl.add_argument("--repo-data", help="cleaner-data repo, e.g. <user>/recon-aware-pick-data")
    dl.add_argument("--data-root", help="the directory to place inputs in; normally $RAPICK_DATA")
    dl.add_argument("--experiments-root", help="the directory to place run artifacts in; normally $RAPICK_WORK")
    dl.add_argument("--ids", help=f"comma-separated, default {','.join(CLEANER_IDS)}")
    dl.add_argument("--with-masks", action="store_true",
                    help="also fetch the contamination masks from --repo-data")
    dl.add_argument("--with-picks", action="store_true",
                    help="also fetch the four pickers' full-set candidates, which is what "
                         "Table 2 and Table S2 need and what avoids installing them")
    dl.add_argument("--with-loop-checkpoints", metavar="ARM", nargs="?", const="fb",
                    choices=list(LOOP_ARMS),
                    help="also fetch that arm's round-1 checkpoints (default arm: fb, "
                         "which is what the paper's Ours row picks with)")
    dl.set_defaults(func=cmd_download)

    args = ap.parse_args()
    if args.cmd == "download" and not (args.repo_weights or args.repo_data):
        ap.error("download needs at least one of --repo-weights / --repo-data")
    if args.cmd == "download" and args.repo_weights and not args.data_root:
        ap.error("--repo-weights needs --data-root")
    if args.cmd == "download" and args.repo_data and not args.experiments_root:
        ap.error("--repo-data needs --experiments-root")
    args.func(args)


if __name__ == "__main__":
    main()
