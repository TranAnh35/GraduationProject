"""
Unit Tests for PECT-JEPA Relative Perturbation Target Architecture (EXP-33).
Tests:
1. Relative perturbation target formulation: Delta H_tgt = H_tgt - mean(H_ctx)
2. Loss computation, gradient backward stability across Context Encoder and Predictor
3. compute_center_anomaly_score inference functionality
4. Full backward compatibility when relative_perturbation_target = False
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath("."))

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5


class TestRelativePerturbationTarget(unittest.TestCase):

    def test_relative_perturbation_target_formulation(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=32,
            encoder_depth=2,
            encoder_heads=2,
            predictor_depth=2,
            predictor_heads=2,
            in_channels=128,
            spatial_topology="concentric_star",
            relative_perturbation_target=True,
        )
        model = PECT_JEPA_5x5(cfg)
        model.eval()

        B = 4
        x = torch.randn(B, 5, 5, 128)

        with torch.no_grad():
            out = model(x)

        self.assertIn("loss", out)
        self.assertIn("H_pred", out)
        self.assertIn("H_tgt", out)
        self.assertIn("H_target_for_loss", out)
        self.assertIn("H_base", out)
        self.assertIsNotNone(out["H_base"])

        h_tgt = out["H_tgt"]
        h_base = out["H_base"]
        h_tgt_expected = h_tgt - h_base
        self.assertTrue(torch.allclose(out["H_target_for_loss"], h_tgt_expected, atol=1e-5))

    def test_relative_perturbation_backward_pass(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=32,
            encoder_depth=2,
            encoder_heads=2,
            predictor_depth=2,
            predictor_heads=2,
            in_channels=128,
            spatial_topology="concentric_star",
            relative_perturbation_target=True,
        )
        model = PECT_JEPA_5x5(cfg)
        model.train()

        B = 4
        x = torch.randn(B, 5, 5, 128)

        out = model(x)
        loss = out["loss"]
        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        # Check gradients on context encoder and predictor
        has_enc_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.context_encoder.parameters())
        has_pred_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.predictor.parameters())
        self.assertTrue(has_enc_grad, "Context encoder received no valid gradients")
        self.assertTrue(has_pred_grad, "Predictor received no valid gradients")

    def test_compute_center_anomaly_score(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=32,
            encoder_depth=2,
            encoder_heads=2,
            predictor_depth=2,
            predictor_heads=2,
            in_channels=128,
            spatial_topology="concentric_star",
            relative_perturbation_target=True,
        )
        model = PECT_JEPA_5x5(cfg)
        model.eval()

        B = 6
        x = torch.randn(B, 5, 5, 128)

        scores = model.compute_center_anomaly_score(x)
        self.assertEqual(scores.shape, (B,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue((scores >= 0.0).all(), "Squared L2 anomaly scores must be non-negative")

    def test_backward_compatibility_when_disabled(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=32,
            encoder_depth=2,
            encoder_heads=2,
            predictor_depth=2,
            predictor_heads=2,
            in_channels=128,
            spatial_topology="concentric_star",
            relative_perturbation_target=False,
            scale_separated_prediction=False,
        )
        model = PECT_JEPA_5x5(cfg)
        model.eval()

        B = 4
        x = torch.randn(B, 5, 5, 128)

        with torch.no_grad():
            out = model(x)

        self.assertIsNone(out["H_base"])
        self.assertTrue(torch.allclose(out["H_target_for_loss"], out["H_tgt"], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
