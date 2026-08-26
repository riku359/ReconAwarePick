"""Group predict.py --dump_hs's per-micrograph npz into one file per EMPIAR entry.

--dump_hs writes one npz per micrograph, which keeps the picking process's memory flat.
The leave-one-ID-out CV wants to read and write a whole entry at a time, so this script
regroups them using the stem -> empiar_id table written by build_train_stem_mapping.py.
Read-only with respect to the dumps: the per-micrograph npz files are not deleted.

Output: <out-dir>/<empiar_id>.npz
    hs_last       (n_mic, 600, 256) fp16
    pred_logits   (n_mic, 600, 2)   fp16
    pred_boxes_px (n_mic, 600, 4)   float32  cx,cy,w,h in original micrograph pixels
    mic_ids       (n_mic,)          <U... (stem)
    orig_w, orig_h (n_mic,)         int32
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-hs-dir", required=True, type=Path)
    ap.add_argument("--mapping-csv", required=True, type=Path,
                     help="stem,empiar_id from build_train_stem_mapping.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem_to_id = {}
    with open(args.mapping_csv) as fh:
        for row in csv.DictReader(fh):
            stem_to_id[row["stem"]] = int(row["empiar_id"])

    id_to_stems = defaultdict(list)
    for stem, eid in stem_to_id.items():
        id_to_stems[eid].append(stem)

    n_missing_total = 0
    for eid in sorted(id_to_stems):
        stems = sorted(id_to_stems[eid])
        hs_list, logits_list, boxes_list, mic_ids, ow_list, oh_list = [], [], [], [], [], []
        missing = []
        for stem in stems:
            f = args.dump_hs_dir / f"{stem}.npz"
            if not f.exists():
                missing.append(stem)
                continue
            d = np.load(f, allow_pickle=True)
            hs_list.append(d["hs_last"])
            logits_list.append(d["pred_logits"])
            boxes_list.append(d["pred_boxes_px"])
            mic_ids.append(str(d["mic_id"]))
            ow_list.append(int(d["orig_w"]))
            oh_list.append(int(d["orig_h"]))
        if missing:
            n_missing_total += len(missing)
            print(f"WARNING: {eid}: {len(missing)}/{len(stems)} dump files missing "
                  f"(first: {missing[:3]})", file=sys.stderr)
        if not hs_list:
            print(f"SKIP {eid}: no dump files found", file=sys.stderr)
            continue
        out_path = args.out_dir / f"{eid}.npz"
        np.savez(
            out_path,
            hs_last=np.stack(hs_list).astype(np.float16),
            pred_logits=np.stack(logits_list).astype(np.float16),
            pred_boxes_px=np.stack(boxes_list).astype(np.float32),
            mic_ids=np.array(mic_ids),
            orig_w=np.array(ow_list, dtype=np.int32),
            orig_h=np.array(oh_list, dtype=np.int32),
        )
        print(f"{eid}: {len(hs_list)} micrographs -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")

    if n_missing_total:
        print(f"\nTOTAL missing dump files across all IDs: {n_missing_total}", file=sys.stderr)


if __name__ == "__main__":
    main()
