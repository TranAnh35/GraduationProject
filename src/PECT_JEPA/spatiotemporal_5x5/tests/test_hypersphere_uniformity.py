"""
Unit tests for Hypersphere Uniformity Dispersion Loss (Wang & Isola, ICML 2020).
Replaces VICReg isotropic whitening tests.

Verifies:
1. JEPALoss5x5 forward pass and dictionary keys (loss_unif).
2. Scale-invariance on the unit sphere via L2 normalization.
3. Sound metal stability: duplicate / identical tokens produce zero NaNs and bounded gradients.
4. Dispersion property: collapsed representations have significantly higher loss than dispersed ones.
5. Numerical stability under FP16 and FP32 precisions.
6. Optimization property: gradient descent on uniformity loss actively expands representation dispersion.
"""

import unittest
import torch
import torch.nn.functional as F
import numpy as np

from src.PECT_JEPA.spatiotemporal_5x5.losses.jepa_loss import JEPALoss5x5
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.manifold_dimension import estimate_twonn_dimension
from src.PECT_JEPA.spatiotemporal_5x5.training.trainer import compute_hypersphere_uniformity


class TestHypersphereUniformityLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 8
        self.N = 10
        self.D = 64

    def test_loss_keys_and_forward(self):
        """Test that forward pass returns loss_unif and gradients propagate smoothly."""
        loss_fn = JEPALoss5x5(
            loss_type="smooth_l1",
            liftoff_invar_weight=0.05,
            phase_align_weight=0.05,
            uniformity_weight=0.05,
            uniformity_t=2.0,
            uniformity_subsample=1024,
        )

        H_pred = torch.randn(self.B, self.N, self.D, requires_grad=True)
        H_tgt = torch.randn(self.B, self.N, self.D)
        H_ctx = torch.randn(self.B, 18, self.D, requires_grad=True)

        loss_dict = loss_fn(H_pred, H_tgt, H_ctx=H_ctx)

        self.assertIn("loss", loss_dict)
        self.assertIn("loss_pred", loss_dict)
        self.assertIn("loss_liftoff", loss_dict)
        self.assertIn("loss_phase", loss_dict)
        self.assertIn("loss_unif", loss_dict)
        self.assertIn("loss_norm", loss_dict)
        self.assertIn("mean_norm", loss_dict)

        total_loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        self.assertIsNotNone(H_pred.grad)
        self.assertIsNotNone(H_ctx.grad)
        self.assertTrue(torch.isfinite(H_ctx.grad).all())

    def test_scale_invariance_on_hypersphere(self):
        """Test that scaling vectors by positive constant does not change uniformity loss."""
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0)
        H = torch.randn(16, 25, self.D)

        loss_base = loss_fn.hypersphere_uniformity_loss(H)
        loss_scaled = loss_fn.hypersphere_uniformity_loss(H * 42.0)

        self.assertAlmostEqual(loss_base.item(), loss_scaled.item(), places=4)

    def test_duplicate_sound_metal_stability(self):
        """Test that duplicate / identical sound metal tokens produce zero NaNs and bounded gradients."""
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0)

        # 95% of tokens are identical (simulating sound metal dominance)
        base_token = torch.randn(1, 1, self.D, requires_grad=True)
        H_sound = base_token.expand(32, 25, self.D).clone().detach().requires_grad_(True)

        loss = loss_fn.hypersphere_uniformity_loss(H_sound)
        self.assertTrue(torch.isfinite(loss), "Uniformity loss must be finite even on identical tokens")

        loss.backward()
        self.assertIsNotNone(H_sound.grad)
        self.assertTrue(torch.isfinite(H_sound.grad).all(), "Gradients must remain finite on duplicate tokens")

    def test_collapsed_vs_dispersed_loss(self):
        """Test that collapsed representations have substantially higher uniformity loss than dispersed ones."""
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0)

        # Dispersed representation (Gaussian spread across sphere)
        H_dispersed = torch.randn(32, 25, self.D)
        loss_dispersed = loss_fn.hypersphere_uniformity_loss(H_dispersed).item()

        # Collapsed representation (points clustered in a tiny 1e-4 ball)
        center = torch.randn(1, 1, self.D)
        H_collapsed = center + 1e-4 * torch.randn(32, 25, self.D)
        loss_collapsed = loss_fn.hypersphere_uniformity_loss(H_collapsed).item()

        # Collapsed -> close to 4.0; Dispersed -> close to 0.0
        self.assertGreater(loss_collapsed, loss_dispersed + 0.5,
                           "Collapsed representations must have much higher uniformity loss than dispersed ones")
        self.assertGreaterEqual(loss_dispersed, 0.0, "Uniformity loss must be strictly non-negative")
        self.assertGreaterEqual(loss_collapsed, 0.0, "Collapsed uniformity loss must be strictly non-negative")

    def test_strictly_non_negative_by_jensen(self):
        """Test that uniformity loss is strictly non-negative across arbitrary random distributions."""
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0)
        for _ in range(10):
            H = torch.randn(8, 20, self.D)
            loss = loss_fn.hypersphere_uniformity_loss(H)
            self.assertGreaterEqual(loss.item(), 0.0, "Loss must be >= 0 by Jensen's inequality")

    def test_fp16_fp32_numerical_stability(self):
        """Test that FP16 inputs do not overflow or cause NaNs."""
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0)
        H_fp16 = (torch.randn(16, 25, self.D, dtype=torch.float16) * 5.0)

        loss = loss_fn.hypersphere_uniformity_loss(H_fp16)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_optimization_disperses_representations(self):
        """Test that optimizing uniformity loss increases distance between tokens."""
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0)

        # Start with tightly clustered points
        center = torch.randn(1, self.D)
        z = torch.nn.Parameter(center + 0.01 * torch.randn(64, self.D))

        with torch.no_grad():
            u_init = F.normalize(z, p=2, dim=-1)
            init_mean_dist = torch.pdist(u_init).mean().item()

        optimizer = torch.optim.Adam([z], lr=0.05)
        for _ in range(30):
            optimizer.zero_grad()
            l_unif = loss_fn.hypersphere_uniformity_loss(z.unsqueeze(0))
            l_unif.backward()
            optimizer.step()

        with torch.no_grad():
            u_final = F.normalize(z, p=2, dim=-1)
            final_mean_dist = torch.pdist(u_final).mean().item()

        self.assertGreater(final_mean_dist, init_mean_dist * 2.0,
                           "Optimizing uniformity must disperse clustered points across the sphere")

    def test_zero_vector_collapse_loophole_is_blocked(self):
        """
        Verify that a zero-vector collapse (z = 0) is blocked and penalized heavily
        with max bound 2t = 4.0, rather than escaping with 0.0.
        """
        loss_fn = JEPALoss5x5(uniformity_weight=1.0, uniformity_t=2.0, norm_floor_weight=0.1, norm_floor_target=1.0)

        # Zero vector representation
        H_zero = torch.zeros(16, 25, self.D)
        loss_zero = loss_fn.hypersphere_uniformity_loss(H_zero)

        # Must receive maximum collapse penalty 2t = 4.0
        self.assertAlmostEqual(loss_zero.item(), 4.0, places=4,
                               msg="Zero vector collapse must receive maximum penalty 4.0, NOT 0.0")

        # Norm floor barrier must also heavily penalize zero vector
        l_norm, mean_norm = loss_fn.norm_floor_loss(H_zero)
        self.assertAlmostEqual(mean_norm.item(), 0.0, places=4)
        self.assertAlmostEqual(l_norm.item(), 1.0, places=4,
                               msg="Norm floor barrier on zero vectors must be (1.0 - 0.0)^2 = 1.0")

    def test_norm_floor_barrier_behavior(self):
        """Verify that norm floor loss is zero when representation norm >= target, positive otherwise."""
        loss_fn = JEPALoss5x5(norm_floor_target=1.0)

        # Norm > target
        H_large = 2.0 * F.normalize(torch.randn(8, 20, self.D), p=2, dim=-1)
        l_norm_large, mean_norm_large = loss_fn.norm_floor_loss(H_large)
        self.assertEqual(l_norm_large.item(), 0.0)
        self.assertAlmostEqual(mean_norm_large.item(), 2.0, places=4)

        # Norm < target
        H_small = 0.5 * F.normalize(torch.randn(8, 20, self.D), p=2, dim=-1)
        l_norm_small, mean_norm_small = loss_fn.norm_floor_loss(H_small)
        self.assertAlmostEqual(l_norm_small.item(), (1.0 - 0.5) ** 2, places=4)
        self.assertAlmostEqual(mean_norm_small.item(), 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
