"""
Unit tests for Dual-Scale Diffusion Tokenizer, 3D Spatiotemporal Diffusion Masker,
and Full Physics-Grounded PECT-JEPA Pipeline.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from ..configs.config import Spatiotemporal5x5Config
from ..models.tokenizer_5x5 import DualScaleDiffusionTokenizer5x5
from ..masking.cluster_mask import SpatiotemporalDiffusionMasker5x5, build_masker_5x5
from ..models.jepa_5x5 import PECT_JEPA_5x5


class TestSpatiotemporalDiffusionJEPA(unittest.TestCase):

    def setUp(self):
        self.config = Spatiotemporal5x5Config(
            in_channels=128,
            embed_dim=64,
            grid_size=5,
            tokenizer_type="dual_scale_diffusion",
            masker_type="spatiotemporal_diffusion",
            num_spatial_cluster=8,
            num_cross_diffusion=8,
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            liftoff_invar_weight=0.05,
            phase_align_weight=0.05,
            var_weight=0.0,
            cov_weight=0.0,
            rank_barrier_weight=0.0,
        )

    def test_tokenizer_shapes_and_phase(self):
        tokenizer = DualScaleDiffusionTokenizer5x5(
            in_channels=128,
            embed_dim=64,
            grid_size=5,
            num_freq_bins=14,
        )
        B = 4
        x = torch.randn(B, 5, 5, 128)
        tokens, pos = tokenizer(x)

        # 50 tokens: 25 spatial x 2 scales (shallow and deep)
        self.assertEqual(tokens.shape, (B, 50, 64))
        self.assertEqual(pos.shape, (B, 50, 64))

        # Check Energy-Weighted Spectral Phase
        ew_phase = DualScaleDiffusionTokenizer5x5.compute_energy_weighted_phase(x, num_bins=14)
        self.assertEqual(ew_phase.shape, (B, 25))
        self.assertTrue(torch.all(ew_phase >= -1.0) and torch.all(ew_phase <= 1.0))

    def test_spatiotemporal_masker_disjoint_and_shapes(self):
        masker = SpatiotemporalDiffusionMasker5x5(
            grid_size=5,
            num_spatial_cluster=8,
            num_cross_diffusion=8,
        )
        B = 4
        ctx_idx, tgt_idx, mask_bool = masker.sample_mask(B)

        self.assertEqual(ctx_idx.shape, (B, 26))
        self.assertEqual(tgt_idx.shape, (B, 24))
        self.assertEqual(mask_bool.shape, (B, 50))

        # Check disjointness and coverage
        for b in range(B):
            ctx_set = set(ctx_idx[b].tolist())
            tgt_set = set(tgt_idx[b].tolist())
            self.assertEqual(len(ctx_set.intersection(tgt_set)), 0, "Context and target must be strictly disjoint!")
            self.assertEqual(len(ctx_set) + len(tgt_set), 50, "Context + target must cover all 50 tokens!")

            # Verify that cross-diffusion masking occurred:
            # Some spatial points have Shallow token in context, but Deep token in target
            shallow_in_ctx = {i // 2 for i in ctx_set if i % 2 == 0}
            deep_in_tgt = {i // 2 for i in tgt_set if i % 2 == 1}
            cross_diff_pts = shallow_in_ctx.intersection(deep_in_tgt)
            self.assertGreaterEqual(len(cross_diff_pts), 1, "Must have cross-diffusion masked points!")

    def test_full_model_forward_backward(self):
        model = PECT_JEPA_5x5(self.config)
        model.train()

        B = 2
        x = torch.randn(B, 5, 5, 128)
        out = model(x)

        self.assertIn("loss", out)
        self.assertIn("loss_pred", out)
        self.assertIn("loss_liftoff", out)
        self.assertIn("loss_phase", out)

        self.assertFalse(torch.isnan(out["loss"]))
        self.assertGreater(out["loss"].item(), 0.0)

        # Backward pass
        out["loss"].backward()

        # Check key gradients
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertIsNotNone(model.tokenizer.time_proj.weight.grad)
        self.assertIsNotNone(model.tokenizer.proj_deep.weight.grad)
        self.assertIsNotNone(model.tokenizer.proj_shallow.weight.grad)
        self.assertIsNotNone(model.context_encoder.blocks[0].mlp.fc1.weight.grad)
        self.assertIsNotNone(model.predictor.blocks[0].mlp.fc1.weight.grad)

    def test_feature_extraction_compat(self):
        model = PECT_JEPA_5x5(self.config)
        model.eval()

        B = 4
        x = torch.randn(B, 5, 5, 128)

        # Center feature must be exactly [B, D]
        z_center = model.extract_center_feature(x)
        self.assertEqual(z_center.shape, (B, 64))

        # All features must be exactly [B, 25, D] for C-scan compatibility
        z_all = model.extract_all_features(x)
        self.assertEqual(z_all.shape, (B, 25, 64))

        # Attention map must be [B, heads, 50, 50]
        attn = model.extract_attention_map(x)
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (B, 4, 50, 50))


if __name__ == "__main__":
    unittest.main()
