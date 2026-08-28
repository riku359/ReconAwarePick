"""Probe the frozen decoder's features to separate a dead head from weak features.

A logistic-regression probe is fitted on the final decoder layer's hidden state hs
(600, 256) against "is this query near a GT particle", which tells apart two
explanations for the released picker's behaviour: that class_embed alone is dead, or
that the features themselves carry little separating information.

class_embed is a single nn.Linear(256, 2), so logistic regression has the same function
class as class_embed itself. The probe's AUC is therefore an estimate of the ceiling a
head-only retrain can reach.

Reads the npz files written by predict.py with --debug_dump (which includes hs).
Read-only.

Usage:
    python linear_probe.py --dump-dir <predict.py --debug_dump output> --out-dir <out>

Environment: RAPICK_DATA (the CryoPPP annotations). See docs/CONFIGURATION.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cryoppp_gt import (  # noqa: E402
    DIAMETERS, gt_match_labels, gt_star_path, load_star_points, normalize_mic_name,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--empiar", type=int, default=10081)
    ap.add_argument("--test-frac", type=float, default=0.2,
                     help="held-out micrograph fraction (split by micrograph, not by query)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    radius = DIAMETERS[args.empiar] / 2.0
    gt_points = load_star_points(str(gt_star_path(args.empiar)))

    files = sorted(args.dump_dir.glob("*.npz"))
    print(f"{len(files)} dumped micrographs found")

    rng = np.random.default_rng(args.seed)
    mic_keys = []
    per_mic = {}
    n_no_gt = 0
    for f in files:
        mic_key = normalize_mic_name(f.stem)
        gt_xy = gt_points.get(mic_key, [])
        if not gt_xy:
            n_no_gt += 1
            continue
        d = np.load(f)
        if "hs" not in d.files:
            raise SystemExit(f"{f} has no 'hs' field - re-run predict.py with the hs-capturing --debug_dump")
        hs = d["hs"]                      # (600, 256)
        pred_boxes = d["pred_boxes"]       # (600, 4) normalized cxcywh
        ow, oh = int(d["orig_w"]), int(d["orig_h"])
        cx = pred_boxes[:, 0] * ow
        cy = pred_boxes[:, 1] * oh
        label = gt_match_labels(cx, cy, gt_xy, radius)  # (600,) bool
        per_mic[mic_key] = (hs, label)
        mic_keys.append(mic_key)

    print(f"{len(mic_keys)} micrographs with GT used, {n_no_gt} dumped micrographs had no GT entry (skipped)")

    mic_keys = np.array(mic_keys)
    rng.shuffle(mic_keys)
    n_test = max(1, int(round(len(mic_keys) * args.test_frac)))
    test_keys = list(mic_keys[:n_test].tolist())
    train_keys = list(mic_keys[n_test:].tolist())

    def stack(keys):
        xs, ys = [], []
        for k in keys:
            hs, label = per_mic[k]
            xs.append(hs)
            ys.append(label)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = stack(train_keys)
    X_test, y_test = stack(test_keys)
    print(f"train: {X_train.shape[0]} queries ({y_train.mean():.4f} positive) from {len(train_keys)} micrographs")
    print(f"test:  {X_test.shape[0]} queries ({y_test.mean():.4f} positive) from {len(test_keys)} micrographs")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_train_s, y_train)

    train_pred = clf.decision_function(X_train_s)
    test_pred = clf.decision_function(X_test_s)
    auc_train = roc_auc_score(y_train, train_pred)
    auc_test = roc_auc_score(y_test, test_pred)

    # Control: how much of auc_test is explained by a fixed per-query-slot prior
    # (query slot k's historical positive rate, estimated from train micrographs only),
    # with no per-image content at all? query_embed is a single (600, hidden_dim) table
    # shared across every image, so each decoder query slot can specialize toward
    # "boxes near slot k tend to land on a particle" independent of what is actually
    # in a given micrograph. If auc_slot_only is close to auc_test, the hs probe above
    # is mostly reading out this fixed positional prior rather than per-image content;
    # if it is much lower, hs carries real per-image discriminative signal beyond that.
    train_label_matrix = np.stack([per_mic[k][1] for k in train_keys], axis=0)  # (n_train_mic, 600)
    slot_positive_rate = train_label_matrix.mean(axis=0)  # (600,)
    test_label_matrix = np.stack([per_mic[k][1] for k in test_keys], axis=0)    # (n_test_mic, 600)
    slot_score_test = np.tile(slot_positive_rate, (test_label_matrix.shape[0], 1)).reshape(-1)
    auc_slot_only = roc_auc_score(y_test, slot_score_test)

    result = dict(
        empiar=args.empiar,
        radius_px=radius,
        n_micrographs_used=len(mic_keys),
        n_micrographs_no_gt_skipped=n_no_gt,
        n_train_micrographs=len(train_keys),
        n_test_micrographs=len(test_keys),
        n_train_queries=int(X_train.shape[0]),
        n_test_queries=int(X_test.shape[0]),
        train_positive_frac=float(y_train.mean()),
        test_positive_frac=float(y_test.mean()),
        auc_train=float(auc_train),
        auc_test=float(auc_test),
        auc_slot_only=float(auc_slot_only),
        hidden_dim=int(X_train.shape[1]),
        note="probe = LogisticRegression on standardized hs (class_embed input), "
             "same function class as class_embed itself (nn.Linear). "
             "auc_test is the held-out-micrograph estimate; auc_train is included "
             "only to check for gross overfitting, not as a result. "
             "auc_slot_only is a control: score = query slot's historical positive "
             "rate estimated from train micrographs only, with zero per-image content "
             "(query_embed is a single table shared across all images, so a slot can "
             "have a fixed image-independent tendency to land on a particle). The gap "
             "auc_test - auc_slot_only is the part of the probe's AUC attributable to "
             "actual per-image content in hs, rather than fixed positional specialization.",
    )
    with open(args.out_dir / "linear_probe_result.json", "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
