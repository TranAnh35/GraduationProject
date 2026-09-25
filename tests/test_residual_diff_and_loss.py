import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import torch
import torch.nn.functional as F

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.predictor import ResidualDiffusionPredictor5x5, build_predictor_5x5
from src.PECT_JEPA.spatiotemporal_5x5.losses.jepa_loss import JEPALoss5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5


class TestResidualDiffusionAndLoss(unittest.TestCase):
    def test_residual_diffusion_predictor_forward_backward(self):
        B, N_ctx, N_tgt, D = 4, 60, 40, 64
        predictor = ResidualDiffusionPredictor5x5(
            embed_dim=D,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            num_freq_bins=14,
        )

        H_context = torch.randn(B, N_ctx, D, requires_grad=True)
        target_pos = torch.randn(B, N_tgt, D)
        context_indices = torch.randint(0, 100, (B, N_ctx))
        target_indices = torch.randint(0, 100, (B, N_tgt))
        freq_condition = torch.rand(B)

        # Forward with return_residual=True
        H_pred, delta_pred, h_base = predictor(
            H_context=H_context,
            target_pos=target_pos,
            context_indices=context_indices,
            target_indices=target_indices,
            freq_condition=freq_condition,
            return_residual=True,
        )

        self.assertEqual(H_pred.shape, (B, N_tgt, D))
        self.assertEqual(delta_pred.shape, (B, N_tgt, D))
        self.assertEqual(h_base.shape, (B, 1, D))
        self.assertFalse(torch.isnan(H_pred).any())

        # Exact residual identity check: H_pred == h_base + delta_pred
        expected_H_pred = h_base.expand(B, N_tgt, -1) + delta_pred
        self.assertTrue(torch.allclose(H_pred, expected_H_pred, atol=1e-5))

        # Backward check
        loss = H_pred.sum()
        loss.backward()
        self.assertIsNotNone(H_context.grad)
        self.assertIsNotNone(predictor.residual_head[0].weight.grad)
        self.assertFalse(torch.isnan(predictor.residual_head[0].weight.grad).any())

    def test_disturbance_weights(self):
        loss_fn = JEPALoss5x5(adaptive_disturbance_weight=2.0)
        B, H, W, C = 4, 5, 5, 128

        # Create 2 uniform sound metal samples (std ~ 0.0) and 2 high-contrast defect samples
        x_uniform = torch.ones(2, H, W, C) * 0.5 + 0.001 * torch.randn(2, H, W, C)
        x_defect = torch.ones(2, H, W, C) * 0.5
        x_defect[:, 2, 2, :] = 5.0  # Large local defect disturbance
        x_raw = torch.cat([x_uniform, x_defect], dim=0)

        weights = loss_fn.compute_disturbance_weights(x_raw)
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (B, 1, 1))

        # Defect weights should be significantly higher than uniform weights
        w_uniform = weights[:2].mean().item()
        w_defect = weights[2:].mean().item()
        self.assertAlmostEqual(w_uniform, 1.0, delta=0.1)
        self.assertGreater(w_defect, 2.8)

    def test_temporal_diffusion_monotonicity_loss(self):
        loss_fn = JEPALoss5x5(temporal_mono_weight=0.05)
        B, N_tgt, D = 4, 40, 64

        # Create target indices containing stages 0, 1, 2, 3
        # In 100-token grid: index % 4 gives stage
        stages = torch.tensor([0, 1, 2, 3] * 10).view(1, 40).expand(B, 40)
        target_indices = stages * 1 + torch.arange(40).unsqueeze(0) * 0
        # Make indices >= 50 to activate stage decomposition, using multiple of 4 (52 % 4 == 0)
        target_indices = 52 + stages

        # 1. Monotonic increasing norms: norm(stg 0) < norm(stg 1) < norm(stg 2) < norm(stg 3)
        delta_mono = torch.zeros(B, N_tgt, D)
        for s in range(4):
            mask = (stages == s)
            scale = (s + 1) * 2.0  # 2.0, 4.0, 6.0, 8.0
            delta_mono[mask] = scale / (D ** 0.5)

        loss_mono_clean = loss_fn.temporal_diffusion_monotonicity_loss(delta_mono, target_indices, margin=0.05)
        self.assertAlmostEqual(loss_mono_clean.item(), 0.0, places=4)

        # 2. Inverted norms: norm(stg 0) > norm(stg 1) > norm(stg 2) > norm(stg 3) (violates diffusion)
        delta_inverted = torch.zeros(B, N_tgt, D, requires_grad=True)
        with torch.no_grad():
            for s in range(4):
                mask = (stages == s)
                scale = (4 - s) * 2.0  # 8.0, 6.0, 4.0, 2.0
                delta_inverted[mask] = scale / (D ** 0.5)

        loss_inverted = loss_fn.temporal_diffusion_monotonicity_loss(delta_inverted, target_indices, margin=0.05)
        self.assertGreater(loss_inverted.item(), 0.5)

        # Gradient check
        loss_inverted.backward()
        self.assertIsNotNone(delta_inverted.grad)
        self.assertFalse(torch.isnan(delta_inverted.grad).any())

    def test_jepa_5x5_unified_stop_gradient(self):
        config = Spatiotemporal5x5Config(
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            predictor_depth=2,
            tokenizer_type="spatiotemporal_patch",
            masker_type="complementary_st",
            predictor_type="residual_diffusion",
            use_target_ema=False,
            adaptive_disturbance_weight=2.0,
            temporal_mono_weight=0.05,
            var_weight=1.0,
            cov_weight=1.0,
            fluct_weight=2.0,
            mixed_precision=False,
            device="cpu"
        )
        model = PECT_JEPA_5x5(config)
        model.train()

        B, H, W, C = 2, 5, 5, 128
        x = torch.randn(B, H, W, C, requires_grad=True)

        loss_dict = model(x)
        self.assertIn("loss", loss_dict)
        self.assertIn("loss_pred", loss_dict)
        self.assertIn("loss_fluct", loss_dict)
        self.assertIn("loss_mono", loss_dict)
        self.assertIn("loss_var", loss_dict)
        self.assertIn("loss_cov", loss_dict)

        total_loss = loss_dict["loss"]
        self.assertFalse(torch.isnan(total_loss).any())
        self.assertFalse(torch.isinf(total_loss).any())

        # Target should be detached (stop-gradient for target prediction)
        self.assertFalse(loss_dict["H_tgt"].requires_grad)

        # Backward verification
        total_loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Verify context encoder and predictor received gradients
        for name, param in model.context_encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing grad for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad for {name}")

        for name, param in model.predictor.named_parameters():
            if param.requires_grad and "default_op" not in name:
                self.assertIsNotNone(param.grad, f"Missing grad for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad for {name}")

    def test_jepa_5x5_legacy_ema_backward_compatibility(self):
        config = Spatiotemporal5x5Config(
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            predictor_depth=2,
            tokenizer_type="spatiotemporal_patch",
            masker_type="complementary_st",
            predictor_type="parabolic_diffusion",
            use_target_ema=True,
            var_weight=1.0,
            cov_weight=1.0,
            mixed_precision=False,
            device="cpu"
        )
        model = PECT_JEPA_5x5(config)
        model.train()

        B, H, W, C = 2, 5, 5, 128
        x = torch.randn(B, H, W, C)
        loss_dict = model(x)
        self.assertIn("loss", loss_dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]).any())

        # EMA update should run without error
        model.update_target_encoder(momentum=0.99)


if __name__ == "__main__":
    unittest.main()
