"""Quantile query selection, with explicit upstream reproduction semantics."""
import numpy as np
import torch


def select_queries(logits, num_queries, quantile, mode):
    """Return raw query indices and associated sigmoid ranking scores (batch size 1).

    Legacy deliberately applies a rank mask to raw query positions. idxfix gathers
    the ranked query IDs and keeps the first occurrence when both class logits rank.
    """
    values, flat = torch.topk(logits.sigmoid().reshape(logits.shape[0], -1),
                             num_queries, dim=1)
    keep = values[0] > np.quantile(values, quantile)
    if mode == 'legacy':
        return np.flatnonzero(keep.cpu().numpy()), values[0, keep]
    if mode != 'legacy_idxfix':
        raise ValueError('unsupported quantile selection: %s' % mode)
    n = int(keep.sum().item())
    indices = (flat[0, :n] // logits.shape[-1]).cpu().numpy()
    _, first = np.unique(indices, return_index=True)
    order = np.sort(first)
    return indices[order], values[0, :n][order]
