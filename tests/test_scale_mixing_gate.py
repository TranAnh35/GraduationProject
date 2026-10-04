#!/usr/bin/env python3
"""
Unit tests for EXP-26 Learnable Scale Mixing Gate
Verifies:
1. ScaleMixingGate shapes, bounds in (0, 1), and gradient flow.
2. PECT_JEPA_5x5 forward pass with learnable_scale_mixing=True.
3. Backward pass and gradient flow through scale_gate parameters.
4. Feature extraction with learnable_scale_mixing=True.
5. Backwards compatibility when learnable_scale_mixing=False.
"""

import unittest
import torch
import torch.nn as nn
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5, ScaleMixingGate

class TestScaleMixingGate(unittest.TestCase):
    def test_scale_mixing_gate_shapes_and_bounds(self):
        B, D = 4, 64
        gate = ScaleMixingGate(embed_dim=D)
        
        h_base = torch.randn(B, 1, D)
        delta_h = torch.randn(B, 8, D)
        
        g = gate(h_base, delta_h)
        self.assertEqual(g.shape, (B, 8, D))
        self.assertTrue(torch.all(g >= 0.0) and torch.all(g <= 1.0))
        
        # Test 2D inputs
        h_2d = torch.randn(B, D)
        d_2d = torch.randn(B, D)
        g_2d = gate(h_2d, d_2d)
        self.assertEqual(g_2d.shape, (B, D))
        self.assertTrue(torch.all(g_2d >= 0.0) and torch.all(g_2d <= 1.0))

    def test_jepa_forward_backward_with_scale_mixing(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=64,
            in_channels=128,
            grid_size=5,
            tokenizer_type="continuous_field",
            predictor_type="neural_field_subspace",
            scale_separated_prediction=True,
            keep_absolute_center_feature=True,
            learnable_scale_mixing=True,
        )
        model = PECT_JEPA_5x5(cfg)
        self.assertIsNotNone(model.scale_gate)
        
        x = torch.randn(2, 5, 5, 128)
        out = model(x)
        self.assertIn("loss", out)
        self.assertIn("gate", out)
        self.assertIsNotNone(out["gate"])
        
        # Check backward pass
        loss = out["loss"]
        loss.backward()
        
        gate_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.scale_gate.parameters())
        self.assertTrue(gate_has_grad, "ScaleMixingGate parameters did not receive gradients!")

    def test_extract_features_with_scale_mixing(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=64,
            in_channels=128,
            grid_size=5,
            tokenizer_type="continuous_field",
            predictor_type="neural_field_subspace",
            scale_separated_prediction=True,
            keep_absolute_center_feature=True,
            learnable_scale_mixing=True,
        )
        model = PECT_JEPA_5x5(cfg).eval()
        
        x = torch.randn(3, 5, 5, 128)
        with torch.no_grad():
            z = model.extract_features(x)
        self.assertEqual(z.shape, (3, 128))
        self.assertFalse(torch.isnan(z).any())

    def test_backwards_compatibility_disabled(self):
        cfg = Spatiotemporal5x5Config(
            embed_dim=64,
            in_channels=128,
            grid_size=5,
            tokenizer_type="continuous_field",
            predictor_type="neural_field_subspace",
            scale_separated_prediction=True,
            keep_absolute_center_feature=True,
            learnable_scale_mixing=False,
        )
        model = PECT_JEPA_5x5(cfg).eval()
        self.assertIsNone(model.scale_gate)
        
        x = torch.randn(2, 5, 5, 128)
        out = model(x)
        self.assertIsNone(out["gate"])
        
        with torch.no_grad():
            z = model.extract_features(x)
        self.assertEqual(z.shape, (2, 128))

if __name__ == "__main__":
    unittest.main()
