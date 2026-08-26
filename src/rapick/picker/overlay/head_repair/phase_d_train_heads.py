"""Offline evaluation for retraining class_embed: LOIO-CV, head architecture, loss form.

The decoder is frozen and hs was dumped beforehand (predict.py --dump_hs), so training
here is nothing but a small head over 256-dimensional input (linear, or a 2-3 layer MLP)
on fp32 CPU tensors. No GPU is used, so a shared GPU stays free for other jobs.

What is trained is always **a single binary logit**. CryoTransformer's own two-class
softmax with eos_coef weighting is mathematically equivalent to weighted binary cross
entropy once one of the two outputs of a num_classes=2 softmax cross entropy is fixed at
0 -- the very equivalence that phase_e_writeback.py's embedding into a two-class layer
(weight[1]=0, bias[1]=0) relies on. The 'softmax' loss is therefore implemented as
weighted BCE, and the focal loss is defined over the same single output.

Headline metric: query-level precision at recall >= 0.98. AUC is reported alongside for
diagnosis only.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# On a many-core host, torch's default intra-op thread pool tries to use every core for
# every matmul. For these tiny heads (256-dim in, <=256 hidden) the per-op
# thread-spawn/sync overhead dominates the actual (microscopic) compute, so more threads
# made training slower, not faster (observed on a 256-core host: one 25-epoch fold pegged
# ~85 cores and still hadn't finished after 3 minutes). The real parallelism lever is
# running many folds as separate OS processes (see --held-out-ids), each single- or
# dual-threaded.
torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_utils import TRAIN_IDS, load_all_ids, flatten  # noqa: E402


def build_model(arch: str) -> nn.Module:
    if arch == "linear":
        return nn.Linear(256, 1)
    if arch == "mlp2":
        return nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    if arch == "mlp3":
        return nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 1))
    raise ValueError(arch)


def weighted_bce(logits, y, eos_coef):
    """Weighted BCE, numerically identical to two-class softmax cross entropy
    (target=0:particle / 1:background, weight=[1, eos_coef]) once one of class_embed's
    two outputs is fixed at 0."""
    weight = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, eos_coef))
    return F.binary_cross_entropy_with_logits(logits, y, weight=weight)


def focal_loss(logits, y, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1 - p)
    alpha_t = torch.where(y > 0.5, torch.full_like(y, alpha), torch.full_like(y, 1 - alpha))
    loss = -alpha_t * (1 - pt).clamp_min(0).pow(gamma) * torch.log(pt.clamp_min(1e-8))
    return loss.mean()


def train_head(X, y, arch, loss_kind, eos_coef=0.1, alpha=0.25, gamma=2.0,
               epochs=25, lr=2e-3, batch_size=32768, seed=0):
    torch.manual_seed(seed)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-6
    Xs = ((X - mu) / sigma).astype(np.float32)
    Xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y.astype(np.float32))
    model = build_model(arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = Xt.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx]).squeeze(-1)
            if loss_kind == "softmax":
                loss = weighted_bce(logits, yt[idx], eos_coef)
            elif loss_kind == "focal":
                loss = focal_loss(logits, yt[idx], alpha, gamma)
            else:
                raise ValueError(loss_kind)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, mu.astype(np.float32), sigma.astype(np.float32)


@torch.no_grad()
def score(model, mu, sigma, X):
    Xs = (X - mu) / sigma
    Xt = torch.from_numpy(Xs.astype(np.float32))
    return torch.sigmoid(model(Xt).squeeze(-1)).numpy()


def precision_at_recall(y_true, scores_arr, target_recall=0.98):
    order = np.argsort(-scores_arr)
    y_sorted = y_true[order]
    s_sorted = scores_arr[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = float(y_true.sum())
    if n_pos == 0:
        return None
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return dict(reachable=False, precision=None, recall=None, threshold=None)
    best = idx[np.argmax(precision[idx])]
    return dict(reachable=True, precision=float(precision[best]),
                recall=float(recall[best]), threshold=float(s_sorted[best]))


def auc_score(y_true, scores_arr):
    # Mann-Whitney U, i.e. AUC = P(positive score > negative score), without a scipy dep
    order = np.argsort(scores_arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores_arr) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_ranks_pos = ranks[y_true.astype(bool)].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def run_loio_cv(id_data, arch, loss_kind, eos_coef, alpha, gamma, seed,
                 log_prefix="", held_out_ids=None, epochs=15, batch_size=65536):
    rows = []
    for held_out in (held_out_ids or TRAIN_IDS):
        train_ids = [e for e in TRAIN_IDS if e != held_out]
        t0 = time.time()
        X_train, y_train, _ = flatten(id_data, train_ids)
        model, mu, sigma = train_head(X_train, y_train, arch, loss_kind, eos_coef, alpha, gamma,
                                       epochs=epochs, seed=seed, batch_size=batch_size)
        X_test, y_test, _ = flatten(id_data, [held_out])
        s_test = score(model, mu, sigma, X_test)
        auc = auc_score(y_test, s_test)
        par = precision_at_recall(y_test, s_test, 0.98)
        dt = time.time() - t0
        row = dict(held_out_id=held_out, arch=arch, loss=loss_kind, eos_coef=eos_coef,
                   alpha=alpha, gamma=gamma, auc=auc, n_test=len(y_test),
                   n_pos_test=int(y_test.sum()), seconds=round(dt, 1), **{f"pr98_{k}": v for k, v in par.items()})
        rows.append(row)
        print(f"{log_prefix}[{held_out}] auc={auc:.4f} "
              f"pr@r98={par.get('precision')} thr={par.get('threshold')} ({dt:.1f}s)")
    return rows


def run_oracle(id_data, arch, loss_kind, eos_coef, alpha, gamma, epochs, seed,
                test_frac=0.2, ids=None):
    rows = []
    rng = np.random.default_rng(seed)
    for eid in (ids or TRAIN_IDS):
        d = id_data[eid]
        n_mic = d["hs_last"].shape[0]
        idx = rng.permutation(n_mic)
        n_test = max(1, int(round(n_mic * test_frac)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        X_train = d["hs_last"][train_idx].reshape(-1, 256)
        y_train = d["labels"][train_idx].reshape(-1)
        X_test = d["hs_last"][test_idx].reshape(-1, 256)
        y_test = d["labels"][test_idx].reshape(-1)
        model, mu, sigma = train_head(X_train, y_train, arch, loss_kind, eos_coef, alpha, gamma,
                                       epochs=epochs, seed=seed)
        s_test = score(model, mu, sigma, X_test)
        auc = auc_score(y_test, s_test)
        par = precision_at_recall(y_test, s_test, 0.98)
        rows.append(dict(empiar_id=eid, arch=arch, loss=loss_kind, eos_coef=eos_coef,
                          n_train_mic=len(train_idx), n_test_mic=len(test_idx),
                          auc=auc, **{f"pr98_{k}": v for k, v in par.items()}))
        print(f"[oracle {eid}] auc={auc:.4f} pr@r98={par.get('precision')}")
    return rows


def write_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs-dumps-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", required=True,
                     choices=["eos_sweep", "arch_compare", "focal", "oracle"])
    ap.add_argument("--arch", default="linear", choices=["linear", "mlp2", "mlp3"])
    ap.add_argument("--eos-coef", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held-out-ids", type=int, nargs="+", default=None,
                     help="restrict LOIO-CV to these held-out IDs (default: all 22). "
                          "Used to fan this script out across IDs as separate OS "
                          "processes, since 22+ sequential fold trainings in one "
                          "process badly underuse a many-core host, and torch's own "
                          "intra-op threading does not help tiny 256-dim heads (see "
                          "torch.set_num_threads above).")
    ap.add_argument("--out-suffix", default="",
                     help="appended to the output CSV filename, e.g. '_id10081', so "
                          "parallel --held-out-ids invocations don't clobber each other")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tkw = dict(epochs=args.epochs, batch_size=args.batch_size)

    print("loading all 22 IDs' hs + labels ...")
    t0 = time.time()
    id_data = load_all_ids(args.hs_dumps_dir)
    print(f"loaded in {time.time()-t0:.1f}s")

    if args.mode == "eos_sweep":
        all_rows = []
        for eos in [0.1, 0.3, 0.5, 1.0]:
            print(f"\n=== eos_coef={eos} (arch=linear, loss=softmax) ===")
            rows = run_loio_cv(id_data, "linear", "softmax", eos, None, None, args.seed,
                                log_prefix=f"eos={eos} ", held_out_ids=args.held_out_ids, **tkw)
            all_rows += rows
        write_csv(all_rows, args.out_dir / f"d3_eos_sweep_loio{args.out_suffix}.csv")

    elif args.mode == "arch_compare":
        all_rows = []
        for arch in ["mlp2", "mlp3"]:
            print(f"\n=== arch={arch} (loss=softmax, eos_coef={args.eos_coef}) ===")
            rows = run_loio_cv(id_data, arch, "softmax", args.eos_coef, None, None, args.seed,
                                log_prefix=f"{arch} ", held_out_ids=args.held_out_ids, **tkw)
            all_rows += rows
        write_csv(all_rows, args.out_dir / f"d2_arch_compare_loio{args.out_suffix}.csv")

    elif args.mode == "focal":
        print(f"\n=== arch={args.arch}, loss=focal (alpha=0.25, gamma=2.0) ===")
        rows = run_loio_cv(id_data, args.arch, "focal", None, 0.25, 2.0, args.seed,
                            log_prefix="focal ", held_out_ids=args.held_out_ids, **tkw)
        write_csv(rows, args.out_dir / f"d3_focal_loio{args.out_suffix}.csv")

    elif args.mode == "oracle":
        print(f"\n=== oracle: arch={args.arch}, loss=softmax, eos_coef={args.eos_coef} ===")
        rows = run_oracle(id_data, args.arch, "softmax", args.eos_coef, None, None,
                           args.epochs, args.seed, ids=args.held_out_ids)
        write_csv(rows, args.out_dir / f"d5_oracle{args.out_suffix}.csv")


if __name__ == "__main__":
    main()
