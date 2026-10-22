"""Run in the picker environment: python -m unittest discover -s tests -p test_query_selection.py."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/rapick/picker/overlay'))
from query_selection import select_queries


class Selection(unittest.TestCase):
    def test_rank_positions_differ_from_query_ids(self):
        logits = torch.tensor([[[0., -1.], [8., -2.], [6., -3.], [9., 7.]]])
        legacy, scores = select_queries(logits, 4, .25, 'legacy')
        fixed, fixed_scores = select_queries(logits, 4, .25, 'legacy_idxfix')
        np.testing.assert_array_equal(legacy, [0, 1, 2])
        np.testing.assert_array_equal(fixed, [3, 1])
        np.testing.assert_allclose(scores.numpy(), torch.sigmoid(torch.tensor([9., 8., 7.])).numpy())
        np.testing.assert_allclose(fixed_scores.numpy(), torch.sigmoid(torch.tensor([9., 8.])).numpy())

    def test_equal_scores_select_nothing_with_strict_quantile(self):
        for mode in ['legacy', 'legacy_idxfix']:
            indices, scores = select_queries(torch.zeros((1, 4, 2)), 4, .25, mode)
            self.assertEqual(len(indices), 0)
            self.assertEqual(len(scores), 0)
