"""
Unit tests for DispersionConditionedContextEncoder5x5 and build_context_encoder_5x5 factory.
Verifies EXP-36: Harmonic Dispersion Conditioning and Anisotropic Skin-Depth Attention Bias.
"""

import unittest
import torch
import torch.nn as nn

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.context_encoder import (
    ContextEncoder5x5,
    DispersionConditionedContextEncoder5x5,
    DispersionConditionedTransformerBlock,
    AnisotropicSkinDepthAttentionBias,
    build_context_encoder_5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5


class TestDispersionConditionedEncoder(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.D = 64
        self.N = 25
        self.num_heads = 4

    def test_dispersion_conditioned_transformer_block(self):
        blk = DispersionConditionedTransformerBlock(
            embed_dim=self.D,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            cond_dim=self.D,
        )
        x = torch.randn(self.B, self.N, self.D)
        cond = torch.randn(self.B, self.D)
        out = blk(x, cond=cond)
        self.assertEqual(out.shape, (self.B, self.N, self.D))
        self.assertFalse(torch.isnan(out).any())

        # Test return_attention
        out, attn = blk(x, cond=cond, return_attention=True)
        self.assertEqual(out.shape, (self.B, self.N, self.D))
        self.assertEqual(attn.shape, (self.B, self.num_heads, self.N, self.N))

    def test_anisotropic_skin_depth_bias(self):
        bias_module = AnisotropicSkinDepthAttentionBias(
            num_heads=self.num_heads,
            init_gamma=0.05,
            init_alpha=0.1,
            init_alpha_x=1.0,
            init_alpha_y=1.0,
            spatial_topology="concentric_star",
            star_radii=(1, 3, 7),
        )

        # 1. Full 25 probes
        omega_bar = torch.tensor([0.1, 0.2, 0.35, 0.5])
        bias_full = bias_module(
            context_indices=None,
            N_tokens=25,
            freq_condition=omega_bar,
            batch_size=self.B,
        )
        self.assertEqual(bias_full.shape, (self.B, self.num_heads, 25, 25))
        # Attention bias should be negative or zero (log-polynomial diffusion penalty)
        self.assertTrue((bias_full <= 1e-6).all())

        # 2. Subsampled context (outer 16 probes: Ring 2 + Ring 3)
        ctx_indices = torch.arange(9, 25).unsqueeze(0).expand(self.B, -1)  # [B, 16]
        bias_ctx = bias_module(
            context_indices=ctx_indices,
            N_tokens=16,
            freq_condition=omega_bar,
            batch_size=self.B,
        )
        self.assertEqual(bias_ctx.shape, (self.B, self.num_heads, 16, 16))
        self.assertTrue((bias_ctx <= 1e-6).all())

    def test_dispersion_conditioned_context_encoder(self):
        encoder = DispersionConditionedContextEncoder5x5(
            embed_dim=self.D,
            depth=3,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            use_radial_attention_bias=True,
            spatial_topology="concentric_star",
            star_radii=(1, 3, 7),
        )

        x_tokens = torch.randn(self.B, 16, self.D)
        x_pos = torch.randn(self.B, 16, self.D)
        ctx_idx = torch.arange(9, 25).unsqueeze(0).expand(self.B, -1)
        freq_cond = torch.tensor([0.105, 0.356, 0.343, 0.200])

        out = encoder(
            context_tokens=x_tokens,
            context_pos=x_pos,
            context_indices=ctx_idx,
            freq_condition=freq_cond,
        )
        self.assertEqual(out.shape, (self.B, 16, self.D))
        self.assertFalse(torch.isnan(out).any())

        # Test backward pass
        loss = out.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Gradient missing for {name}")

    def test_build_context_encoder_factory(self):
        cfg_disp = Spatiotemporal5x5Config()
        cfg_disp.encoder_type = "dispersion_conditioned"
        enc_disp = build_context_encoder_5x5(cfg_disp)
        self.assertIsInstance(enc_disp, DispersionConditionedContextEncoder5x5)

        cfg_std = Spatiotemporal5x5Config()
        cfg_std.encoder_type = "standard"
        enc_std = build_context_encoder_5x5(cfg_std)
        self.assertIsInstance(enc_std, ContextEncoder5x5)

    def test_full_pect_jepa_integration(self):
        cfg = Spatiotemporal5x5Config()
        cfg.encoder_type = "dispersion_conditioned"
        cfg.tokenizer_type = "energy_adaptive_dual_domain"
        cfg.masker_type = "radial_diffusion"
        cfg.radial_mask_mode = "inward_core"
        cfg.embed_dim = self.D
        cfg.encoder_depth = 2
        cfg.predictor_depth = 2

        model = PECT_JEPA_5x5(cfg)
        x = torch.randn(self.B, 5, 5, cfg.in_channels)

        # Forward self-supervised step
        loss_dict = model(x)
        self.assertIn("loss", loss_dict)
        self.assertIn("loss_pred", loss_dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]))

        # Backward step
        loss_dict["loss"].backward()

        # Inference extractions
        z_unified = model.extract_unified_features(x)
        self.assertEqual(z_unified.shape, (self.B, 2 * self.D))
        self.assertFalse(torch.isnan(z_unified).any())

        all_feats = model.extract_all_features(x)
        self.assertEqual(all_feats.shape, (self.B, 25, self.D))

        anomaly_score = model.compute_center_anomaly_score(x)
        self.assertEqual(anomaly_score.shape, (self.B,))
        self.assertFalse(torch.isnan(anomaly_score).any())


if __name__ == "__main__":
    unittest.main()
