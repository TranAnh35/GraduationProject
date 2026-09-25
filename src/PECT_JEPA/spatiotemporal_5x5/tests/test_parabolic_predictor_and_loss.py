"""
Unit tests for ParabolicDiffusionPredictor5x5 and Context-Referenced Fluctuation Loss.
"""

import unittest
import torch

from ..configs.config import Spatiotemporal5x5Config
from ..models.predictor import ParabolicDiffusionPredictor5x5, build_predictor_5x5
from ..losses.jepa_loss import JEPALoss5x5
from ..models.jepa_5x5 import PECT_JEPA_5x5


class TestParabolicPredictorAndLoss(unittest.TestCase):

    def setUp(self):
        self.B = 4
        self.D = 64
        self.device = torch.device("cpu")
        self.config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=self.D,
            encoder_depth=2,
            encoder_heads=4,
            predictor_type="parabolic_diffusion",
            predictor_depth=2,
            predictor_heads=4,
            diffusion_gamma_init=1.0,
            diffusion_alpha_init=0.5,
            fluct_weight=2.0,
            phase_align_weight=0.05,
            liftoff_invar_weight=0.05,
            var_weight=1.0,
            cov_weight=1.0,
        )

    def test_parabolic_predictor_attention_bias_and_shapes(self):
        predictor = build_predictor_5x5(self.config)
        self.assertIsInstance(predictor, ParabolicDiffusionPredictor5x5)

        H_ctx = torch.randn(self.B, 34, self.D)
        target_pos = torch.randn(self.B, 16, self.D)

        # Context: 17 probes x {0, 1}; Target: 8 probes x {2, 3}
        c_idx = torch.randint(0, 50, (self.B, 34))
        t_idx = torch.randint(50, 100, (self.B, 16))
        freq_cond = torch.tensor([0.2, 0.5, 0.8, 1.0])

        H_pred = predictor(
            H_context=H_ctx,
            target_pos=target_pos,
            context_indices=c_idx,
            target_indices=t_idx,
            freq_condition=freq_cond,
        )

        self.assertEqual(H_pred.shape, (self.B, 16, self.D))
        self.assertTrue(torch.isfinite(H_pred).all())

    def test_fluctuation_loss_decomposition(self):
        loss_fn = JEPALoss5x5(loss_type="l1", fluct_weight=2.0)

        # Create target representations with common mean + distinct perturbations
        H_tgt = torch.randn(self.B, 16, self.D)
        H_pred = H_tgt + 0.1 * torch.randn(self.B, 16, self.D)
        t_idx = torch.tensor([[s * 4 + tau for s in range(8) for tau in (2, 3)] for _ in range(self.B)])

        total_loss, l_fluct = loss_fn.fluctuation_prediction_loss(H_pred, H_tgt, target_indices=t_idx)

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(l_fluct))
        self.assertGreater(total_loss.item(), 0.0)
        self.assertGreater(l_fluct.item(), 0.0)

    def test_full_model_forward_and_backward(self):
        model = PECT_JEPA_5x5(self.config)
        model.train()

        x = torch.randn(self.B, 5, 5, 128)
        out = model(x)

        self.assertIn("loss", out)
        self.assertIn("loss_pred", out)
        self.assertIn("loss_fluct", out)
        self.assertIn("loss_phase", out)
        self.assertIn("loss_var", out)
        self.assertIn("loss_cov", out)

        loss = out["loss"]
        self.assertTrue(torch.isfinite(loss))

        # Backward gradient flow check
        loss.backward()

        # Verify gradients on key physical coupling parameters
        self.assertIsNotNone(model.predictor.raw_gamma.grad)
        self.assertIsNotNone(model.predictor.raw_alpha.grad)
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertTrue(torch.isfinite(model.predictor.raw_gamma.grad))
        self.assertTrue(torch.isfinite(model.predictor.raw_alpha.grad))


if __name__ == "__main__":
    unittest.main()
