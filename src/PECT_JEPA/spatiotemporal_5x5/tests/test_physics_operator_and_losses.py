"""
Unit tests for Physics Operator Diffusion Predictor, Lift-off Invariance Loss,
and Self-Supervised Phase-Depth Alignment Loss (Option B).
"""

import unittest
import torch
import torch.nn as nn

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5
from ..models.predictor import OperatorDiffusionPredictor5x5, DiffusionOperatorEmbedding
from ..losses.jepa_loss import JEPALoss5x5


class TestPhysicsOperatorAndLosses(unittest.TestCase):
    def setUp(self):
        self.config = Spatiotemporal5x5Config(
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            predictor_type="operator_diffusion",
            predictor_depth=2,
            predictor_heads=4,
            tokenizer_type="dual_domain_attention",
            liftoff_invar_weight=0.05,
            phase_align_weight=0.05,
            uniformity_weight=0.05,
        )

    def test_diffusion_operator_embedding(self):
        op_emb = DiffusionOperatorEmbedding(embed_dim=64, num_freq_bands=8)
        # Default query
        q_def = op_emb()
        self.assertEqual(q_def.shape, (1, 1, 64))

        # Explicit frequency query
        freq_norm = torch.tensor([0.2, 0.5, 0.8, 1.0])
        q_freq = op_emb(freq_norm)
        self.assertEqual(q_freq.shape, (4, 1, 64))

    def test_operator_diffusion_predictor(self):
        pred = OperatorDiffusionPredictor5x5(
            embed_dim=64,
            depth=2,
            num_heads=4,
            num_freq_bins=14,
        )
        B, N_ctx, N_tgt, D = 4, 15, 10, 64
        H_ctx = torch.randn(B, N_ctx, D)
        target_pos = torch.randn(B, N_tgt, D)
        freq_cond = torch.randint(1, 15, (B,))

        out = pred(H_ctx, target_pos, freq_condition=freq_cond)
        self.assertEqual(out.shape, (B, N_tgt, D))
        self.assertFalse(torch.isnan(out).any())

    def test_jepa_loss_with_physical_losses(self):
        loss_fn = JEPALoss5x5(
            loss_type="smooth_l1",
            liftoff_invar_weight=0.05,
            phase_align_weight=0.05,
        )
        B, N_tgt, N_ctx, D = 4, 10, 15, 64
        H_pred = torch.randn(B, N_tgt, D, requires_grad=True)
        H_tgt = torch.randn(B, N_tgt, D)
        H_ctx = torch.randn(B, N_ctx, D, requires_grad=True)
        H_ctx_pert = H_ctx + 0.05 * torch.randn(B, N_ctx, D)
        z_depth = torch.randn(B, N_ctx, requires_grad=True)
        phase_ctx = torch.randn(B, N_ctx)

        out = loss_fn(
            H_pred=H_pred,
            H_target=H_tgt,
            H_ctx=H_ctx,
            H_ctx_pert=H_ctx_pert,
            z_depth=z_depth,
            phase_ctx=phase_ctx,
        )

        self.assertIn("loss", out)
        self.assertIn("loss_pred", out)
        self.assertIn("loss_liftoff", out)
        self.assertIn("loss_phase", out)

        self.assertGreater(out["loss"].item(), 0.0)
        self.assertGreater(out["loss_liftoff"].item(), 0.0)
        self.assertGreater(out["loss_phase"].item(), 0.0)

        # Verify gradient backpropagation
        out["loss"].backward()
        self.assertIsNotNone(H_pred.grad)
        self.assertIsNotNone(H_ctx.grad)
        self.assertIsNotNone(z_depth.grad)

    def test_full_model_forward_and_backward(self):
        model = PECT_JEPA_5x5(self.config)
        model.train()

        B, C = 2, 128
        x = torch.randn(B, 5, 5, C)
        loss_dict = model(x)

        self.assertIn("loss", loss_dict)
        self.assertIn("loss_liftoff", loss_dict)
        self.assertIn("loss_phase", loss_dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]))

        loss_dict["loss"].backward()

    def test_compute_characteristic_frequency(self):
        B, C = 4, 128
        # Signal 1: Pure low frequency (slow sine wave)
        t = torch.linspace(0, 1, C)
        x_low = torch.sin(2 * torch.pi * 2 * t).view(1, 1, 1, C).expand(2, 5, 5, C)
        # Signal 2: High frequency (fast sine wave)
        x_high = torch.sin(2 * torch.pi * 20 * t).view(1, 1, 1, C).expand(2, 5, 5, C)
        x = torch.cat([x_low, x_high], dim=0)  # [4, 5, 5, C]

        freq_bar = PECT_JEPA_5x5.compute_characteristic_frequency(x, num_bins=14)
        self.assertEqual(freq_bar.shape, (4,))
        self.assertTrue((freq_bar > 0.0).all() and (freq_bar <= 1.0).all())

        # Low frequency signals must have strictly lower omega_bar than high frequency signals
        self.assertLess(freq_bar[0].item(), freq_bar[2].item())
        self.assertLess(freq_bar[1].item(), freq_bar[3].item())

    def test_greens_diffusion_attention_bias_properties(self):
        pred = OperatorDiffusionPredictor5x5(
            embed_dim=64,
            depth=2,
            num_heads=4,
            num_freq_bins=14,
            gamma_init=1.0,
            beta_init=0.5,
        )
        B = 2
        # Target at center (2, 2) shallow: index 24 (spatial 12, scale 0)
        # Context 1 at near neighbor (2, 3) shallow: index 26 (spatial 13, scale 0), dist = 1.0
        # Context 2 at far corner (0, 0) shallow: index 0 (spatial 0, scale 0), dist = sqrt(2^2+2^2) = 2.828
        target_indices = torch.tensor([[24]], dtype=torch.long).expand(B, -1)  # [B, 1]
        context_indices = torch.tensor([[26, 0]], dtype=torch.long).expand(B, -1)  # [B, 2]

        H_ctx = torch.randn(B, 2, 64)
        target_pos = torch.randn(B, 1, 64)

        # 1. Forward with frequency condition
        f_low = torch.tensor([0.2, 0.2])
        f_high = torch.tensor([0.9, 0.9])

        out_low = pred(
            H_context=H_ctx,
            target_pos=target_pos,
            context_indices=context_indices,
            target_indices=target_indices,
            freq_condition=f_low,
        )
        self.assertEqual(out_low.shape, (B, 1, 64))

        # 2. Check gradient flow into raw_gamma and raw_beta
        loss = (out_low ** 2).sum()
        loss.backward()
        self.assertIsNotNone(pred.raw_gamma.grad)
        self.assertIsNotNone(pred.raw_beta.grad)
        self.assertGreater(pred.raw_gamma.grad.abs().item(), 0.0)

    def test_dual_scale_50_tokens_with_operator_predictor(self):
        config_50 = Spatiotemporal5x5Config(
            in_channels=128,
            embed_dim=64,
            tokenizer_type="dual_scale_diffusion",
            predictor_type="operator_diffusion",
            masker_type="spatiotemporal_diffusion",
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            num_spatial_cluster=8,
            num_cross_diffusion=8,
            liftoff_invar_weight=0.05,
            phase_align_weight=0.05,
            uniformity_weight=0.05,
        )
        model = PECT_JEPA_5x5(config_50)
        model.train()

        B, C = 2, 128
        x = torch.randn(B, 5, 5, C, requires_grad=True)
        loss_dict = model(x)

        self.assertIn("loss", loss_dict)
        self.assertIn("loss_pred", loss_dict)
        self.assertIn("loss_unif", loss_dict)
        self.assertEqual(loss_dict["H_pred"].shape, (B, 24, 64))
        self.assertEqual(loss_dict["H_tgt"].shape, (B, 24, 64))
        self.assertEqual(loss_dict["H_ctx"].shape, (B, 26, 64))

        loss = loss_dict["loss"]
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.predictor.raw_gamma.grad)
        self.assertIsNotNone(model.predictor.raw_beta.grad)

    def test_vicreg_variance_and_covariance_mechanics(self):
        loss_fn = JEPALoss5x5(
            loss_type="l1",
            var_weight=1.0,
            cov_weight=1.0,
            var_gamma=1.0,
        )
        B, N, D = 4, 16, 64

        # 1. Collapsed representations (zero variance across batch)
        # All points are identical constant vector -> std = 0 -> variance hinge must equal gamma = 1.0
        H_collapsed = torch.ones(B, N, D, requires_grad=True)
        var_loss = loss_fn.variance_hinge(H_collapsed)
        self.assertAlmostEqual(var_loss.item(), 1.0, places=2)

        # 2. High variance representations (std >= 1.0) -> hinge loss must equal 0.0
        H_high_var = 5.0 * torch.randn(B, N, D)
        var_loss_high = loss_fn.variance_hinge(H_high_var)
        self.assertAlmostEqual(var_loss_high.item(), 0.0, places=2)

        # 3. Covariance penalty on perfectly correlated features
        # All 64 channels identical -> off-diagonal covariance is large
        z_base = torch.randn(B * N, 1)
        H_corr = z_base.expand(B * N, D).reshape(B, N, D)
        cov_loss_corr = loss_fn.covariance_penalty(H_corr)
        self.assertGreater(cov_loss_corr.item(), 1.0)

        # 4. Forward with VICReg active
        H_pred = torch.randn(B, 10, D, requires_grad=True)
        H_tgt = torch.randn(B, 10, D)
        H_ctx = torch.randn(B, N, D, requires_grad=True)
        out = loss_fn(H_pred=H_pred, H_target=H_tgt, H_ctx=H_ctx)

        self.assertIn("loss_var", out)
        self.assertIn("loss_cov", out)
        self.assertGreater(out["loss_var"].item(), 0.0)
        self.assertGreater(out["loss_cov"].item(), 0.0)

        out["loss"].backward()
        self.assertIsNotNone(H_ctx.grad)
        self.assertGreater(H_ctx.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
