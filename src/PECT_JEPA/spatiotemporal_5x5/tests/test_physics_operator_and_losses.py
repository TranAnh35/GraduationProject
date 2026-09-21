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
            var_weight=0.0,
            cov_weight=0.0,
            rank_barrier_weight=0.0,
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

        # Check gradients for key modules
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertIsNotNone(model.target_harmonic_proj[0].weight.grad)
        self.assertIsNotNone(model.context_encoder.blocks[0].mlp.fc1.weight.grad)
        self.assertIsNotNone(model.predictor.blocks[0].mlp.fc1.weight.grad)


if __name__ == "__main__":
    unittest.main()
