"""
Shape and Information Flow Tests (Section 22.1).
Verifies tensor dimensions across all pipeline stages for both T'=64 and T'=128.
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config


class TestShapes(unittest.TestCase):
    def test_forward_shape_t64(self):
        """Test full forward pass with T'=64."""
        config = get_default_config()
        config.temporal_encoder.t_prime = 64
        config.temporal_encoder.raw_samples = 500
        config.clip.temporal_length = 16
        config.clip.stride = 8
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        # Synthetic input: batch_size=2, spatial=64x64, temporal=500
        B, H, W, L = 2, 64, 64, 500
        x = torch.randn(B, H, W, L)

        out = model(x)

        loss = out["loss"]
        H_pred = out["H_pred"]
        H_target = out["H_target"]
        H_context = out["H_context"]
        grid_shape = out["grid_shape"]

        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertTrue(loss.ndim == 0, "Loss should be a scalar")
        self.assertEqual(H_pred.shape, H_target.shape, "H_pred and H_target shape mismatch")
        self.assertEqual(H_pred.shape[0], B, "Batch size mismatch")
        self.assertEqual(H_pred.shape[-1], config.tokenizer.embed_dim, "Embed dim mismatch")
        print(f"PASS: Shape Test T'=64 | Loss: {loss.item():.4f} | Grid: {grid_shape} | H_pred: {H_pred.shape}")

    def test_forward_shape_t128(self):
        """Test full forward pass with T'=128."""
        config = get_default_config()
        config.temporal_encoder.t_prime = 128
        config.temporal_encoder.raw_samples = 500
        config.clip.temporal_length = 16
        config.clip.stride = 8
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.encoder.attention_type = "factorized"
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        B, H, W, L = 2, 64, 64, 500
        x = torch.randn(B, H, W, L)

        out = model(x)

        loss = out["loss"]
        H_pred = out["H_pred"]
        H_target = out["H_target"]
        grid_shape = out["grid_shape"]

        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertEqual(H_pred.shape, H_target.shape)
        print(f"PASS: Shape Test T'=128 | Loss: {loss.item():.4f} | Grid: {grid_shape} | H_pred: {H_pred.shape}")

    def test_forward_shape_ps4(self):
        """Test forward pass with Ps=4 and large grid (H_t=75 > 64) with Factorized Attention."""
        config = get_default_config()
        config.temporal_encoder.t_prime = 64
        config.temporal_encoder.raw_samples = 500
        config.tokenizer.spatial_patch = 4  # Ps=4 -> H_t=75 for 300x300
        config.tokenizer.embed_dim = 64
        config.encoder.attention_type = "factorized"
        config.encoder.depth = 2
        config.predictor.depth = 1
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        # Input: 1 sample with 300x300 -> H_t=75
        B, H, W, L = 1, 300, 300, 500
        x = torch.randn(B, H, W, L)

        with torch.no_grad():
            out = model(x)
            feats = model.extract_features(x, pool_temporal=False)

        self.assertEqual(out["grid_shape"][:2], (75, 75), "Spatial grid should be (75, 75)")
        self.assertEqual(feats.shape, (1, 75, 75, 7, 64), "Full latent shape mismatch")
        print(f"PASS: Dynamic Ps=4 Test (Grid {out['grid_shape']}) | Feats: {feats.shape}")


if __name__ == "__main__":
    unittest.main()
