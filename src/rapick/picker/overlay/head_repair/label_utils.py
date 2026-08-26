"""Labelling utilities shared by the head repair of Sec. S2 (leave-one-ID-out CV).

Many-to-one labels: a query is positive when the centre of its predicted box lies within
R = diameter / 2 (per entry) of any GT particle. Hungarian one-to-one assignment is not
used: with the decoder frozen, the hidden states of several queries on the same particle
are nearly identical, so making only one winner positive would hand the linear head
contradictory labels.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryoppp_gt import (  # noqa: E402
    DIAMETERS_ALL, TRAIN_IDS_EXPECTED_COUNT, cached_load_star_points,
    expand_alias_keys, gt_match_labels, gt_star_path, resolve_train_stem_key,
)

TRAIN_IDS = sorted(TRAIN_IDS_EXPECTED_COUNT)


def load_id_arrays(hs_dumps_dir: Path, eid: int):
    """Return (hs_last, pred_boxes_px, labels, mic_ids, radius) for one entry.

    hs_last: (n_mic, 600, 256) float32 (upcast from the fp16 dump)
    labels:  (n_mic, 600) bool, the many-to-one GT-proximity test
    """
    d = np.load(hs_dumps_dir / f"{eid}.npz", allow_pickle=True)
    hs_last = d["hs_last"].astype(np.float32)
    pred_boxes_px = d["pred_boxes_px"]
    mic_ids = d["mic_ids"]
    radius = DIAMETERS_ALL[eid] / 2.0

    gt_points = expand_alias_keys(cached_load_star_points(gt_star_path(eid)))
    n_mic = hs_last.shape[0]
    labels = np.zeros((n_mic, hs_last.shape[1]), dtype=bool)
    n_no_gt = 0
    for i in range(n_mic):
        gt_xy = resolve_train_stem_key(gt_points, str(mic_ids[i])) or []
        if not gt_xy:
            n_no_gt += 1
        cx = pred_boxes_px[i, :, 0]
        cy = pred_boxes_px[i, :, 1]
        labels[i] = gt_match_labels(cx, cy, gt_xy, radius)
    return dict(hs_last=hs_last, pred_boxes_px=pred_boxes_px, labels=labels,
                mic_ids=mic_ids, radius=radius, n_no_gt=n_no_gt)


def load_all_ids(hs_dumps_dir: Path, ids=None):
    ids = ids or TRAIN_IDS
    return {eid: load_id_arrays(hs_dumps_dir, eid) for eid in ids}


def flatten(id_data: dict, ids):
    """Flatten and concatenate several entries' (hs_last, labels) to (N, 256) / (N,)."""
    xs, ys, id_of_row = [], [], []
    for eid in ids:
        d = id_data[eid]
        n_mic, n_q, c = d["hs_last"].shape
        xs.append(d["hs_last"].reshape(-1, c))
        ys.append(d["labels"].reshape(-1))
        id_of_row.append(np.full(n_mic * n_q, eid, dtype=np.int32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(id_of_row)
