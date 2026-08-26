"""Map each CryoTransformer training micrograph to the EMPIAR entry it came from.

CryoTransformer's training split (train_val_test_data/train/*.jpg) carries only the
depositor's original filenames, with no record of which of the 22 CryoPPP entries each
micrograph belongs to. The head repair needs that grouping, so it is recovered here by
matching every training stem against the local CryoPPP ground-truth stars'
_rlnMicrographName. The naming convention alone ("stack_NNNN_<suffix>") is not enough:
several entries can collide on it, so the match is on the full normalized name.

Read-only: the extracted training data and the CryoPPP ground truth are never written.
The only outputs are the mapping CSV and the parse cache under $RAPICK_WORK.

Output: <out-csv>, columns stem,empiar_id, consumed by aggregate_hs_by_id.py.

Environment: RAPICK_DATA, RAPICK_WORK. See docs/CONFIGURATION.md.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryoppp_gt import (  # noqa: E402
    TRAIN_IDS_EXPECTED_COUNT, cached_load_star_points, cryoppp_root,
    expand_alias_keys, resolve_train_stem_key,
)


def build_gt_name_to_id():
    """{normalize_mic_name -> entry id} from the local CryoPPP GT (selected + excluded)."""
    name_to_ids = defaultdict(set)
    ids_seen = []
    for eid in sorted(TRAIN_IDS_EXPECTED_COUNT):
        gt_dir = cryoppp_root() / str(eid) / "ground_truth"
        stars = sorted(gt_dir.glob("*_selected.star")) + sorted(gt_dir.glob("*_excluded.star"))
        if not stars:
            print(f"WARNING: no GT star for {eid} under {gt_dir}", file=sys.stderr)
            continue
        ids_seen.append(eid)
        for star in stars:
            print(f"  parsing {star} ...", file=sys.stderr)
            points = cached_load_star_points(star)
            for mic_key in expand_alias_keys(points):
                name_to_ids[mic_key].add(eid)
    collisions = {k: v for k, v in name_to_ids.items() if len(v) > 1}
    if collisions:
        print(f"WARNING: {len(collisions)} micrograph name(s) collide across IDs: "
              f"{list(collisions.items())[:5]}", file=sys.stderr)
    return {k: next(iter(v)) for k, v in name_to_ids.items()}, ids_seen


def list_train_split_stems(train_dir: Path):
    return [p.stem for p in train_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, type=Path,
                     help="train_val_test_data/train/ (already extracted from the tar.gz)")
    ap.add_argument("--out-csv", required=True, type=Path,
                     help="stem,empiar_id for every matched training image; consumed by "
                          "aggregate_hs_by_id.py to group --dump_hs output by entry")
    args = ap.parse_args()

    gt_name_to_id, ids_seen = build_gt_name_to_id()
    print(f"loaded GT for {len(ids_seen)}/{len(TRAIN_IDS_EXPECTED_COUNT)} train IDs")

    train_stems = list_train_split_stems(args.train_dir)
    print(f"{len(train_stems)} train-split image files found on disk")

    id_counts = Counter()
    unmatched = []
    stem_to_id = {}
    for stem in train_stems:
        # train_val_test_data's jpg stems are the depositor's ORIGINAL filenames,
        # not CryoSPARC-imported ones -- unlike the GT star's _rlnMicrographName,
        # they never carry a CryoSPARC import hash prefix to strip. Some of them
        # legitimately start with a digit run (e.g. a date "20210903_106_..."),
        # which normalize_mic_name's leading "^\d+_" strip would misinterpret as
        # that hash and cut off (EMPIAR 11183, 10852). Try the raw stem first;
        # fall back to normalize_mic_name only for the (so far unseen) case where
        # a train stem genuinely does carry such a prefix.
        eid = resolve_train_stem_key(gt_name_to_id, stem)
        if eid is None:
            unmatched.append(stem)
        else:
            id_counts[eid] += 1
            stem_to_id[stem] = eid

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w_csv = csv.writer(fh)
        w_csv.writerow(["stem", "empiar_id"])
        for stem, eid in sorted(stem_to_id.items()):
            w_csv.writerow([stem, eid])
    print(f"wrote {args.out_csv} ({len(stem_to_id)} rows)")

    print(f"\nmatched {len(train_stems) - len(unmatched)}/{len(train_stems)}, "
          f"unmatched: {len(unmatched)}")
    if unmatched:
        print(f"  first unmatched examples: {unmatched[:10]}")

    print("\nper-ID train counts (observed vs the counts CryoTransformer's README lists):")
    for eid, expected in sorted(TRAIN_IDS_EXPECTED_COUNT.items()):
        observed = id_counts.get(eid, 0)
        flag = "" if observed == expected else "  <-- MISMATCH"
        print(f"  {eid}: expected={expected:4d} observed={observed:4d}{flag}")


if __name__ == "__main__":
    main()
