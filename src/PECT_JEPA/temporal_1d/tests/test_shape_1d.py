"""
Shape & Tensor Consistency Tests for 1D Temporal PECT-JEPA (implement.md, Section 5.1).
Verifies:
- Input: [2, 500] -> Token sequence: [2, 16, 128]
- H_context: [2, 5, 128]
- H_pred: [2, 11, 128]
- H_target: [2, 11, 128]
- Finite scalar loss
"""

import torch
import unittest
from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import get_default_config_1d


class TestShape1D(unittest.TestCase):
    def test_forward_shape_1d(self):
        config = get_default_config_1d()
        config.time_samples = 500
        config.patch_length = 32
        config.stride = 31
        config.embed_dim = 128
        config.mask_strategy = "late_decay"
        config.num_visible_early = 5
        config.encoder_depth = 4
        config.predictor_depth = 2
        config.device = "cpu"

        model = PECT_JEPA_1D(config)
        model.eval()

        # Batch of 2 waveforms
        B, T = 2, 500
        x = torch.randn(B, T)

        out = model(x)

        loss = out["loss"]
        H_pred = out["H_pred"]
        H_target = out["H_target"]
        H_context = out["H_context"]
        ctx_idx = out["context_indices"]
        tgt_idx = out["target_indices"]

        # Assertions
        self.assertEqual(ctx_idx.shape, (2, 5), f"Context indices shape mismatch: {ctx_idx.shape}")
        self.assertEqual(tgt_idx.shape, (2, 11), f"Target indices shape mismatch: {tgt_idx.shape}")
        self.assertEqual(H_context.shape, (2, 5, 128), f"H_context shape mismatch: {H_context.shape}")
        self.assertEqual(H_pred.shape, (2, 11, 128), f"H_pred shape mismatch: {H_pred.shape}")
        self.assertEqual(H_target.shape, (2, 11, 128), f"H_target shape mismatch: {H_target.shape}")

        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN")
        self.assertFalse(torch.isinf(loss).item(), "Loss is Inf")
        self.assertTrue(loss.ndim == 0, "Loss should be scalar")

        # Feature extraction
        feats_unpooled = model.extract_features(x, pool=False)
        feats_pooled = model.extract_features(x, pool=True)
        self.assertEqual(feats_unpooled.shape, (2, 16, 128))
        self.assertEqual(feats_pooled.shape, (2, 128))

        print(f"PASS: test_shape_1d.py | Input: {x.shape} -> H_ctx: {H_context.shape}, H_pred: {H_pred.shape}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    unittest.main()
