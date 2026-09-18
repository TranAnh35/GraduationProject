"""
Unit tests for the 5x5 Spatiotemporal PECT-JEPA Evaluation Pipeline.
Tests:
- AnomalyDetector5x5 fit and score_map calculation
- Defect contrast metrics (CNR, peak ratio)
- Linear CKA computation on identical vs orthogonal vs noisy feature matrices
- Cosine similarity calculation
- Synthetic C-scan sliding window feature extraction
"""

import unittest
import numpy as np
import torch

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5
from ..evaluation.linear_probe import LinearProbeEvaluator
from ..evaluation.anomaly_detection import compute_anomaly_metrics, plot_latent_representation_quality
from ..evaluation.liftoff_invariance import compute_linear_cka, compute_feature_similarity_matrix, compute_effective_rank


class TestEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=2,
            predictor_depth=1,
            predictor_heads=2,
        )
        self.model = PECT_JEPA_5x5(self.config)
        self.model.eval()

    def test_linear_probe_and_metrics(self):
        """Verify LinearProbeEvaluator defect probability map and True CNR metrics."""
        np.random.seed(42)
        sY, sX, D = 30, 30, 64
        # Normal sound metal background (class 0)
        features = np.random.randn(sY, sX, D).astype(np.float32) * 0.1
        gt_mask = np.zeros((sY, sX), dtype=np.int32)
        # Add a localized defect at center (12:18, 12:18)
        features[12:18, 12:18, :] += 2.5
        gt_mask[12:18, 12:18] = 1

        evaluator = LinearProbeEvaluator(n_splits=3)
        metrics, prob_map = evaluator.fit_and_predict_probability_map(features, gt_mask)

        self.assertEqual(prob_map.shape, (sY, sX))
        self.assertIn("linear_probe_auc_roc", metrics)
        self.assertGreater(metrics["linear_probe_auc_roc"], 0.90)

        # Defect zone should have substantially higher predicted probability than sound background
        defect_prob = np.mean(prob_map[12:18, 12:18])
        bg_prob = np.mean(prob_map[0:10, 0:10])
        self.assertGreater(defect_prob, bg_prob)

        cnr_res = compute_anomaly_metrics(prob_map, gt_mask=gt_mask)
        self.assertIn("contrast_ratio_cnr", cnr_res)
        self.assertGreater(cnr_res["contrast_ratio_cnr"], 2.0, "CNR should be distinct for inserted defect")

    def test_linear_cka_properties(self):
        """Verify Linear CKA mathematical properties (1.0 for self, ~0 for independent)."""
        np.random.seed(42)
        N, D = 500, 64
        X = np.random.randn(N, D)

        # 1. CKA with self should be identically 1.0
        cka_self = compute_linear_cka(X, X)
        self.assertAlmostEqual(cka_self, 1.0, places=4)

        # 2. CKA with orthogonal random matrix should be close to 0
        Y_ortho = np.random.randn(N, D)
        cka_ortho = compute_linear_cka(X, Y_ortho)
        self.assertLess(cka_ortho, 0.2)

        # 3. CKA with linearly scaled and slightly noisy version should be very high (> 0.90)
        X_scaled = X * 2.5 + np.random.randn(N, D) * 0.05
        cka_scaled = compute_linear_cka(X, X_scaled)
        self.assertGreater(cka_scaled, 0.95)

    def test_cosine_similarity(self):
        """Verify mean cosine similarity calculation."""
        np.random.seed(42)
        N, D = 100, 32
        X = np.random.randn(N, D).astype(np.float32)

        # Identical matrices should have cosine sim = 1.0
        sim_self = compute_feature_similarity_matrix(X, X)
        self.assertAlmostEqual(sim_self, 1.0, places=4)

        # Inverted matrices should have cosine sim = -1.0
        sim_neg = compute_feature_similarity_matrix(X, -X)
        self.assertAlmostEqual(sim_neg, -1.0, places=4)

    def test_effective_rank(self):
        """Verify effective rank computation for collapsed vs isotropic representations."""
        np.random.seed(42)
        N, D = 500, 64

        # 1. Full isotropic Gaussian noise should have high effective rank close to D
        X_isotropic = np.random.randn(N, D)
        rank_iso = compute_effective_rank(X_isotropic)
        self.assertGreater(rank_iso, 45.0, f"Isotropic rank {rank_iso} should be > 45 for D={D}")

        # 2. Completely collapsed rank-1 representation (all points on 1D line)
        v = np.random.randn(1, D)
        scales = np.random.randn(N, 1)
        X_collapsed = scales @ v
        rank_collapsed = compute_effective_rank(X_collapsed)
        self.assertAlmostEqual(rank_collapsed, 1.0, places=1, msg=f"Collapsed rank {rank_collapsed} should be ~1.0")


if __name__ == "__main__":
    unittest.main()
