"""Train the final head on all 22 entries and write theta_0: the repaired checkpoint.

With the configuration chosen by phase_d_train_heads.py, this trains the final head on
the full 22-entry data, overwrites class_embed in the released checkpoint, and writes a
new .pth.

- Standardization (the mu/sigma used during training) is folded into the head's weights,
  so inference needs no separate normalization step (w' = w/sigma, b' = b - w'.mu).
- Linear case: to embed the head into the two-class softmax, weight[0]=w', bias[0]=b',
  weight[1]=0, bias[1]=0. Then softmax(-1)[..., 0] == sigmoid(w'.h + b'), so predict.py's
  existing read-out (taking index 0 via [..., :-1]) works unchanged.
- MLP case: the same trick (two outputs, the second fixed at zero) is applied to the last
  Linear layer only, and models/detr.py's class_embed definition itself has to be
  switched to the matching shape. This script only saves a state_dict with the new key
  layout into the new checkpoint.
- backbone / encoder / decoder / bbox_embed / query_embed keep the released
  checkpoint's values, untouched.

The configuration is decided from phase_d_train_heads.py's LOIO-CV results and passed in
by the caller as --arch / --loss / --eos-coef. The offline training logic used here is
exactly phase_d_train_heads.train_head, not a copy of it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_utils import TRAIN_IDS, load_all_ids, flatten  # noqa: E402
from phase_d_train_heads import train_head, score, auc_score, precision_at_recall  # noqa: E402


def fold_standardization(model: nn.Module, mu: np.ndarray, sigma: np.ndarray):
    """Fold standardization into the first nn.Linear's weight/bias so the head eats raw
    hs directly, with no normalization step on the inference side."""
    first_linear = model if isinstance(model, nn.Linear) else model[0]
    assert isinstance(first_linear, nn.Linear)
    sigma_t = torch.from_numpy(sigma).float()
    mu_t = torch.from_numpy(mu).float()
    with torch.no_grad():
        W = first_linear.weight.data          # (out, 256)
        b = first_linear.bias.data             # (out,)
        W_new = W / sigma_t.unsqueeze(0)
        b_new = b - (W_new * mu_t.unsqueeze(0)).sum(dim=1)
        first_linear.weight.data = W_new
        first_linear.bias.data = b_new


def build_two_class_linear(model_1out: nn.Linear) -> nn.Linear:
    """Embed the 1-logit nn.Linear(256, 1) into an nn.Linear(256, 2).
    index 0 = particle (the trained logit), index 1 = no-object (always 0)."""
    new_layer = nn.Linear(256, 2)
    with torch.no_grad():
        new_layer.weight.data[0] = model_1out.weight.data[0]
        new_layer.bias.data[0] = model_1out.bias.data[0]
        new_layer.weight.data[1] = 0.0
        new_layer.bias.data[1] = 0.0
    return new_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs-dumps-dir", required=True, type=Path)
    ap.add_argument("--arch", required=True, choices=["linear", "mlp2", "mlp3"])
    ap.add_argument("--loss", required=True, choices=["softmax", "focal"])
    ap.add_argument("--eos-coef", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-in", type=Path,
                     default=Path("pretrained_model/CryoTransformer_pretrained_model.pth"))
    ap.add_argument("--checkpoint-out", type=Path, required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                     help="fraction of ALL 22-ID mics held out (mic-level) to report a "
                          "final sanity AUC/precision@recall98 for the deployed head; "
                          "the deployed weights are then refit on 100%% of the data")
    args = ap.parse_args()

    print("loading all 22 IDs' hs + labels ...")
    id_data = load_all_ids(args.hs_dumps_dir)

    # sanity metric on a held-out mic-level split (not used to pick anything -- the
    # config was already chosen from Phase D's LOIO-CV; this is just a final check
    # that the fully-refit deployed head is not obviously broken).
    rng = np.random.default_rng(args.seed)
    train_ids_shuffled = list(TRAIN_IDS)
    holdout_mic_idx = {}
    for eid in TRAIN_IDS:
        n_mic = id_data[eid]["hs_last"].shape[0]
        idx = rng.permutation(n_mic)
        n_hold = max(1, int(round(n_mic * args.holdout_frac)))
        holdout_mic_idx[eid] = (idx[:n_hold], idx[n_hold:])

    def flatten_subset(mic_selector):
        xs, ys = [], []
        for eid in TRAIN_IDS:
            sel = mic_selector(eid)
            if len(sel) == 0:
                continue
            xs.append(id_data[eid]["hs_last"][sel].reshape(-1, 256))
            ys.append(id_data[eid]["labels"][sel].reshape(-1))
        return np.concatenate(xs), np.concatenate(ys)

    X_sanity_test, y_sanity_test = flatten_subset(lambda e: holdout_mic_idx[e][0])
    X_sanity_train, y_sanity_train = flatten_subset(lambda e: holdout_mic_idx[e][1])
    sanity_model, mu_s, sigma_s = train_head(
        X_sanity_train, y_sanity_train, args.arch, args.loss,
        args.eos_coef, args.alpha, args.gamma, epochs=args.epochs, seed=args.seed)
    s_test = score(sanity_model, mu_s, sigma_s, X_sanity_test)
    sanity_auc = auc_score(y_sanity_test, s_test)
    sanity_par = precision_at_recall(y_sanity_test, s_test, 0.98)
    print(f"sanity (held-out {args.holdout_frac:.0%} mics, 90/10 split, NOT the deployed "
          f"weights): auc={sanity_auc:.4f} pr@r98={sanity_par}")

    # deployed weights: refit on ALL 22 IDs' full data.
    print("refitting on 100% of the 22-ID data for deployment ...")
    X_all, y_all, _ = flatten(id_data, TRAIN_IDS)
    model, mu, sigma = train_head(X_all, y_all, args.arch, args.loss,
                                   args.eos_coef, args.alpha, args.gamma,
                                   epochs=args.epochs, seed=args.seed)
    fold_standardization(model, mu, sigma)

    if args.arch == "linear":
        new_class_embed = build_two_class_linear(model)
        new_state = new_class_embed.state_dict()
        new_state = {f"class_embed.{k}": v for k, v in new_state.items()}
    else:
        # mlp2/mlp3: fold the same 1-logit -> 2-logit trick into the LAST Linear layer
        # of the trained Sequential; models/detr.py's class_embed definition must be
        # switched to the matching MLP shape. The paper uses --arch linear, for which no
        # such change to models/detr.py is needed.
        last_linear = model[-1]
        assert isinstance(last_linear, nn.Linear) and last_linear.out_features == 1
        new_last = nn.Linear(last_linear.in_features, 2)
        with torch.no_grad():
            new_last.weight.data[0] = last_linear.weight.data[0]
            new_last.bias.data[0] = last_linear.bias.data[0]
            new_last.weight.data[1] = 0.0
            new_last.bias.data[1] = 0.0
        model[-1] = new_last
        new_state = {f"class_embed.{k}": v for k, v in model.state_dict().items()}

    ckpt = torch.load(args.checkpoint_in, map_location="cpu")
    old_keys = [k for k in ckpt["model"] if k.startswith("class_embed.")]
    for k in old_keys:
        del ckpt["model"][k]
    ckpt["model"].update(new_state)
    ckpt["stage1_head_repair"] = dict(
        arch=args.arch, loss=args.loss, eos_coef=args.eos_coef,
        alpha=args.alpha, gamma=args.gamma, epochs=args.epochs, seed=args.seed,
        source_checkpoint=str(args.checkpoint_in),
        sanity_auc=sanity_auc, sanity_precision_at_recall98=sanity_par,
        note="class_embed only; backbone/encoder/decoder/bbox_embed/query_embed "
             "unchanged from source_checkpoint. index0=particle, index1=no-object "
             "(fixed at logit 0). Standardization folded into the weights.",
    )
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.checkpoint_out)
    print(f"wrote {args.checkpoint_out}")
    with open(args.checkpoint_out.with_suffix(".json"), "w") as fh:
        json.dump(ckpt["stage1_head_repair"], fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
