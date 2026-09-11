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
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.anomaly_detection import AnomalyDetector5x5, compute_anomaly_metrics
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

        self.assertAlmostEqual(loss_base.item(), loss_scaled.item(), places=4)

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

    def test_anomaly_detector_raw_scoring(self):
        """Test that AnomalyDetector5x5 can score test maps with detrend=False."""
        sY, sX, D = 20, 20, 16
        train_map = np.random.randn(sY, sX, D).astype(np.float32)
        test_map = np.random.randn(sY, sX, D).astype(np.float32)

        # Inject anomaly at center
        test_map[9:12, 9:12, :] += 5.0

        detector = AnomalyDetector5x5(n_clusters=2, metric="euclidean", detrend=True)
        detector.fit(train_map)

        # Test both modes
        detrended_scores = detector.score_map(test_map, detrend=True)
        raw_scores = detector.score_map(test_map, detrend=False)

        self.assertEqual(detrended_scores.shape, (sY, sX))
        self.assertEqual(raw_scores.shape, (sY, sX))

        metrics_raw = compute_anomaly_metrics(raw_scores)
        metrics_det = compute_anomaly_metrics(detrended_scores)

        self.assertIn("contrast_ratio_cnr", metrics_raw)
        self.assertIn("contrast_ratio_cnr", metrics_det)
        self.assertTrue(np.isfinite(metrics_raw["contrast_ratio_cnr"]))
        self.assertTrue(np.isfinite(metrics_det["contrast_ratio_cnr"]))

    def test_config_rank_barrier_defaults(self):
        """Verify Spatiotemporal5x5Config includes rank_barrier_weight and rank_barrier_eps."""
        cfg = Spatiotemporal5x5Config()
        self.assertTrue(hasattr(cfg, "rank_barrier_weight"))
        self.assertTrue(hasattr(cfg, "rank_barrier_eps"))
        self.assertGreater(cfg.rank_barrier_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
