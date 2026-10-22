"""CPU training invariants; requires setup's pinned picker checkout/environment."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from PIL import Image
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'src/rapick/picker/overlay'
sys.path.insert(0, str(ROOT / 'third_party/cryotransformer'))
sys.path.insert(0, str(OVERLAY / 'head_repair'))
spec = importlib.util.spec_from_file_location('rapick_finetune', OVERLAY / 'finetune.py')
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)
from phase_d_train_heads import train_head, score
from phase_e_writeback import fold_standardization, build_two_class_linear


class TrainingInvariants(unittest.TestCase):
    def test_split_is_disjoint_nonempty_and_reproducible(self):
        samples = list(range(5))
        train, val = ft.split_train_val(samples, .2, 7)
        self.assertEqual((train, val), ft.split_train_val(samples, .2, 7))
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 1)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(samples))

    def test_particle_zero_label_and_clipped_boxes(self):
        target = ft.build_target([(2, 3), (50, 50)], 100, 100, 10, 1)
        np.testing.assert_allclose(target['boxes'].numpy(), [[0, 0, 7, 8], [45, 45, 55, 55]])
        np.testing.assert_array_equal(target['labels'].numpy(), [0, 0])

    def test_head_writeback_preserves_probabilities(self):
        rng = np.random.RandomState(1)
        features = (rng.normal(size=(64, 256)) * 3 + 8).astype(np.float32)
        labels = np.arange(64) % 2 == 0
        model, mu, sigma = train_head(features, labels, 'linear', 'softmax', epochs=2)
        expected = score(model, mu, sigma, features)
        self.assertTrue(np.isfinite(expected).all())
        fold_standardization(model, mu, sigma)
        deployed = build_two_class_linear(model)
        actual = deployed(torch.from_numpy(features)).softmax(-1)[:, 0].detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(torch.count_nonzero(deployed.weight[1]).item(), 0)


class RasterInput(unittest.TestCase):
    def test_png_and_rgb_inputs_have_three_grayscale_channels(self):
        spec = importlib.util.spec_from_file_location('rapick_predict', OVERLAY / 'predict.py')
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        p = Path(tempfile.mkdtemp(prefix='rapick-raster-'))
        pixels = np.array([[0, 128], [255, 37]], dtype=np.uint8)
        for mode in ['L', 'RGB']:
            source = Image.fromarray(pixels).convert(mode)
            path = p / (mode + '.png')
            source.save(path)
            result = np.asarray(predictor.load_raster_rgb(path))
            self.assertEqual(result.shape, (2, 2, 3))
            for channel in range(3):
                np.testing.assert_array_equal(result[:, :, channel], pixels)
