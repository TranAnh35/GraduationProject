"""
Shape and Information Flow Tests for PECT-JEPA v0.2 (implement.md, Section 4.1).
Verifies:
- Input clip: [1, 300, 300, 16] -> Token grid: [1, 37, 37, 16, 128]
- H_context: [1, N_ctx, 128]
- H_pred: [1, N_tgt, 128]
- H_target: [1, N_tgt, 128]
- Scalar JEPA loss
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config


class TestShapes(unittest.TestCase):
    def test_forward_shape_300x300x16(self):
        """Test full forward pass with input clip [1, 300, 300, 16]."""
        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.clip.temporal_length = 16
        config.mask.spatial_block_h = 8
        config.mask.spatial_block_w = 8
        config.mask.num_masked_frames = 8
        config.encoder.depth = 4
        config.predictor.depth = 2
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        # Input clip: [1, 300, 300, 16]
        B, H, W, T_c = 1, 300, 300, 16
        clip = torch.randn(B, H, W, T_c)

        out = model(clip)

        loss = out["loss"]
        H_pred = out["H_pred"]
        H_target = out["H_target"]
        H_context = out["H_context"]
        grid_shape = out["grid_shape"]
        ctx_idx = out["context_indices"]
        tgt_idx = out["target_indices"]

        # Expected dimensions: H_t = 300//8 = 37, W_t = 300//8 = 37, T_c = 16
        self.assertEqual(grid_shape, (37, 37, 16), f"Grid shape mismatch: {grid_shape}")
        
        N_tgt = 8 * 8 * 8  # K=8 frames * B_h=8 * B_w=8 = 512
        total_tokens = 37 * 37 * 16  # 21,904
        N_ctx = total_tokens - N_tgt  # 21,392

        self.assertEqual(H_context.shape, (1, N_ctx, 128), f"H_context shape mismatch: {H_context.shape}")
        self.assertEqual(H_pred.shape, (1, N_tgt, 128), f"H_pred shape mismatch: {H_pred.shape}")
        self.assertEqual(H_target.shape, (1, N_tgt, 128), f"H_target shape mismatch: {H_target.shape}")

        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertTrue(loss.ndim == 0, "Loss should be a scalar")
        print(f"PASS: test_shape.py | Input: {clip.shape} -> Grid: {grid_shape} | H_ctx: {H_context.shape}, H_pred: {H_pred.shape}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    unittest.main()
