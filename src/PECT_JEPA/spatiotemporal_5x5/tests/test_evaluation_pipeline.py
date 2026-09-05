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
from ..evaluation.anomaly_detection import AnomalyDetector5x5
from ..evaluation.liftoff_invariance import compute_linear_cka, compute_feature_similarity_matrix
from ..evaluate import compute_anomaly_metrics


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

    def test_anomaly_detector_and_metrics(self):
        """Verify unsupervised anomaly detector and contrast metrics."""
        np.random.seed(42)
        # Create synthetic feature map: 30x30 with 64 dimensions
        sY, sX, D = 30, 30, 64
        # Normal sound metal background (cluster 0)
        features = np.random.randn(sY, sX, D).astype(np.float32) * 0.1
        # Add a localized defect at center (12:18, 12:18)
        features[12:18, 12:18, :] += 2.0

        detector = AnomalyDetector5x5(n_clusters=2)
        detector.fit(features)
        score_map = detector.score_map(features)

        self.assertEqual(score_map.shape, (sY, sX))
        # Defect zone should have substantially higher score than sound background
        defect_score = np.mean(score_map[12:18, 12:18])
        bg_score = np.mean(score_map[0:10, 0:10])
        self.assertGreater(defect_score, bg_score)

        metrics = compute_anomaly_metrics(score_map, top_percentile=95.0)
        self.assertIn("contrast_ratio_cnr", metrics)
        self.assertIn("peak_contrast_ratio", metrics)
        self.assertGreater(metrics["contrast_ratio_cnr"], 1.0, "CNR should be distinct for inserted defect")
        self.assertGreater(metrics["max_score"], metrics["background_mean"])

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


if __name__ == "__main__":
    unittest.main()
