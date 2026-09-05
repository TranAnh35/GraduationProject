"""
Tensor shape verification test for 5x5 PECT-JEPA.
"""

import unittest
import torch

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5


class TestShape5x5(unittest.TestCase):

    def setUp(self):
        self.config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=128,
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            min_masked=10,
            max_masked=15,
        )
        self.model = PECT_JEPA_5x5(self.config)

    def test_forward_shapes(self):
        B = 4
        x = torch.randn(B, 5, 5, 128)
        out = self.model(x)

        self.assertIn("loss", out)
        self.assertIn("loss_pred", out)
        self.assertIn("loss_var", out)
        self.assertIn("loss_cov", out)

        self.assertEqual(out["loss"].ndim, 0)

        N_tgt = out["target_indices"].shape[1]
        N_ctx = out["context_indices"].shape[1]

        self.assertEqual(N_tgt + N_ctx, 25)
        self.assertEqual(out["H_pred"].shape, (B, N_tgt, 128))
        self.assertEqual(out["H_tgt"].shape, (B, N_tgt, 128))
        self.assertEqual(out["H_ctx"].shape, (B, N_ctx, 128))

    def test_center_feature_extraction(self):
        B = 4
        x = torch.randn(B, 5, 5, 128)
        z_center = self.model.extract_center_feature(x)
        self.assertEqual(z_center.shape, (B, 128))

    def test_all_features_extraction(self):
        B = 4
        x = torch.randn(B, 5, 5, 128)
        z_all = self.model.extract_all_features(x)
        self.assertEqual(z_all.shape, (B, 25, 128))


if __name__ == "__main__":
    unittest.main()
