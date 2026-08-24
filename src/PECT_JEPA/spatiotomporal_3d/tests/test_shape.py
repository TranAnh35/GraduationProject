"""
Shape and Information Flow Tests for PECT-JEPA v0.3 (implement.md, Section 5.1).
Verifies:
- Standard crop clip: [1, 64, 64, 16] -> Token grid: [1, 8, 8, 16, 128] = 1,024 tokens
- Resolution-agnostic support: [1, 300, 300, 16] -> Token grid: [1, 37, 37, 16, 128] = 21,904 tokens
- H_context: [1, N_ctx, 128]
- H_pred: [1, N_tgt, 128]
- H_target: [1, N_tgt, 128]
- Finite scalar JEPA loss
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config


class TestShapes(unittest.TestCase):
    def test_forward_shape_crop_64x64x16(self):
        """Test forward pass on standard training crop [1, 64, 64, 16]."""
        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.clip.temporal_length = 16
        config.mask.spatial_block_h = 4
        config.mask.spatial_block_w = 4
        config.mask.num_masked_frames = 8
        config.encoder.depth = 4
        config.predictor.depth = 2
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        # Standard training crop: [1, 64, 64, 16]
        B, H, W, T_c = 1, 64, 64, 16
        clip = torch.randn(B, H, W, T_c)

        out = model(clip)

        loss = out["loss"]
        H_pred = out["H_pred"]
        H_target = out["H_target"]
        H_context = out["H_context"]
        grid_shape = out["grid_shape"]
        ctx_idx = out["context_indices"]
        tgt_idx = out["target_indices"]

        # Expected dimensions: H_t = 64//8 = 8, W_t = 64//8 = 8, T_c = 16
        self.assertEqual(grid_shape, (8, 8, 16), f"Grid shape mismatch: {grid_shape}")
        
        N_tgt = 8 * 4 * 4  # K=8 frames * B_h=4 * B_w=4 = 128 target tokens
        total_tokens = 8 * 8 * 16  # 1,024 total tokens
        N_ctx = total_tokens - N_tgt  # 896 context tokens

        self.assertEqual(H_context.shape, (1, N_ctx, 128), f"H_context shape mismatch: {H_context.shape}")
        self.assertEqual(H_pred.shape, (1, N_tgt, 128), f"H_pred shape mismatch: {H_pred.shape}")
        self.assertEqual(H_target.shape, (1, N_tgt, 128), f"H_target shape mismatch: {H_target.shape}")

        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertTrue(loss.ndim == 0, "Loss should be a scalar")
        print(f"PASS: test_shape.py (Crop 64x64) | Input: {clip.shape} -> Grid: {grid_shape} | H_ctx: {H_context.shape}, H_pred: {H_pred.shape}, Loss: {loss.item():.4f}")

    def test_forward_shape_300x300x16(self):
        """Test resolution-agnostic forward pass with input clip [1, 300, 300, 16]."""
        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.clip.temporal_length = 16
        config.mask.spatial_block_h = 4
        config.mask.spatial_block_w = 4
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

        self.assertEqual(grid_shape, (37, 37, 16), f"Grid shape mismatch: {grid_shape}")
        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        print(f"PASS: test_shape.py (Full 300x300) | Input: {clip.shape} -> Grid: {grid_shape} | H_ctx: {H_context.shape}, H_pred: {H_pred.shape}")


if __name__ == "__main__":
    unittest.main()
