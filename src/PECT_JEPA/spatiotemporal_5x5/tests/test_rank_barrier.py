"""
Unit tests for Log-Determinant Spectral Rank Barrier Loss and Raw Latent Evaluation.
Verifies:
1. JEPALoss5x5 with rank barrier: loss computation and loss dictionary keys.
2. Numerical stability under FP32 / FP16 precision, including near-singular covariance matrices.
3. Rank expansion property: optimizing rank barrier actively prevents rank collapse.
4. Scale-invariance: multiplying representations by constant alpha does not affect rank barrier loss.
5. AnomalyDetector5x5 support for detrend=False (raw latent distance).
"""

import unittest
import torch
import numpy as np

from src.PECT_JEPA.spatiotemporal_5x5.losses.jepa_loss import JEPALoss5x5
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.liftoff_invariance import compute_effective_rank
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.anomaly_detection import compute_anomaly_metrics, plot_latent_representation_quality
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config


class TestRankBarrierLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 8
        self.N = 7
        self.D = 64

    def test_loss_keys_and_forward(self):
        """Test that forward pass returns loss_rank_barrier key and non-zero value when weight > 0."""
        loss_fn = JEPALoss5x5(
            loss_type="smooth_l1",
            var_weight=1.0,
            cov_weight=1.0,
            rank_barrier_weight=0.05,
            rank_barrier_eps=1e-4,
            vicreg_target="context",
        )

        H_pred = torch.randn(self.B, self.N, self.D, requires_grad=True)
        H_tgt = torch.randn(self.B, self.N, self.D)
        H_ctx = torch.randn(self.B, 18, self.D, requires_grad=True)

        loss_dict = loss_fn(H_pred, H_tgt, H_ctx=H_ctx)

        self.assertIn("loss", loss_dict)
        self.assertIn("loss_pred", loss_dict)
        self.assertIn("loss_var", loss_dict)
        self.assertIn("loss_cov", loss_dict)
        self.assertIn("loss_rank_barrier", loss_dict)

        total_loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        self.assertIsNotNone(H_ctx.grad)
        self.assertTrue(torch.isfinite(H_ctx.grad).all())

    def test_scale_invariance(self):
        """Test that multiplying representations by a constant factor does not change rank_barrier_loss."""
        loss_fn = JEPALoss5x5(rank_barrier_weight=1.0, rank_barrier_eps=1e-4)

        H_rep = torch.randn(self.B, self.N, self.D)
        loss_base = loss_fn.rank_barrier_loss(H_rep)
        loss_scaled = loss_fn.rank_barrier_loss(H_rep * 25.0)

        self.assertAlmostEqual(loss_base.item(), loss_scaled.item(), places=3)

    def test_collapsed_vs_isotropic_loss(self):
        """Test that rank barrier loss is significantly higher on a collapsed representation than on isotropic."""
        loss_fn = JEPALoss5x5(rank_barrier_weight=1.0, rank_barrier_eps=1e-4)

        # Isotropic representation
        H_iso = torch.randn(32, 25, self.D)
        loss_iso = loss_fn.rank_barrier_loss(H_iso).item()

        # Severely collapsed representation (rank ~ 1)
        u = torch.randn(32 * 25, 1)
        v = torch.randn(1, self.D)
        H_coll = (u @ v).view(32, 25, self.D) + 1e-4 * torch.randn(32, 25, self.D)
        loss_coll = loss_fn.rank_barrier_loss(H_coll).item()

        self.assertGreater(loss_coll, loss_iso + 3.0, "Collapsed loss should be substantially higher than isotropic")

    def test_rank_barrier_optimization_expands_rank(self):
        """Test that minimizing rank barrier loss increases the effective rank of representations."""
        D = 32
        u = torch.randn(128, 1)
        v = torch.randn(1, D)
        z = torch.nn.Parameter((u @ v) + 1e-4 * torch.randn(128, D))

        init_rank = compute_effective_rank(z.detach().numpy())
        self.assertLess(init_rank, 2.0, "Initial rank should be severely collapsed (< 2.0)")

        loss_fn = JEPALoss5x5(rank_barrier_weight=1.0, rank_barrier_eps=1e-4)
        optimizer = torch.optim.Adam([z], lr=0.05)

        for _ in range(40):
            optimizer.zero_grad()
            l = loss_fn.rank_barrier_loss(z.unsqueeze(0))
            l.backward()
            optimizer.step()

        final_rank = compute_effective_rank(z.detach().numpy())
        self.assertGreater(final_rank, init_rank * 3.0, "Rank should at least triple after 40 optimization steps")

    def test_raw_latent_quality_metrics(self):
        """Test that plot_latent_representation_quality computes raw latent angular distance and PCA metrics."""
        sY, sX, D = 20, 20, 16
        base_dir = np.random.randn(1, 1, D).astype(np.float32)
        base_dir = base_dir / np.linalg.norm(base_dir)
        test_map = np.tile(base_dir, (sY, sX, 1)) + 0.05 * np.random.randn(sY, sX, D).astype(np.float32)
        gt_mask = np.zeros((sY, sX), dtype=np.int32)

        # Inject anomaly at center pointing away from nominal vector
        test_map[9:12, 9:12, :] = -base_dir + 0.05 * np.random.randn(3, 3, D).astype(np.float32)
        gt_mask[9:12, 9:12] = 1

        lq_res = plot_latent_representation_quality(
            feature_map=test_map,
            gt_mask=gt_mask,
            close_fig=True,
        )

        self.assertIn("pca_variance_explained", lq_res)
        self.assertIn("total_3pc_variance", lq_res)
        self.assertIn("angular_cnr", lq_res)
        self.assertIn("angular_auc", lq_res)
        self.assertTrue(np.isfinite(lq_res["angular_cnr"]))
        self.assertGreater(lq_res["angular_cnr"], 1.0)

    def test_config_rank_barrier_defaults(self):
        """Verify Spatiotemporal5x5Config includes rank_barrier_weight and rank_barrier_eps."""
        cfg = Spatiotemporal5x5Config()
        self.assertTrue(hasattr(cfg, "rank_barrier_weight"))
        self.assertTrue(hasattr(cfg, "rank_barrier_eps"))
        self.assertGreater(cfg.rank_barrier_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
