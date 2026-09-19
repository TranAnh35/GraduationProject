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

    def test_downstream_benchmarks_and_visualizations(self):
        """Verify downstream benchmarks (Tasks 1-3) and visualization generation in temporary folder."""
        import tempfile
        import os
        from ..evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
        from ..evaluation.visualizations import (
            plot_probability_heatmap,
            plot_roc_pr_curves,
            plot_depth_regression_maps,
            plot_depth_calibration_scatter,
            plot_severity_confusion_matrix,
            plot_liftoff_cka_heatmap,
        )

        np.random.seed(42)
        N, D = 200, 32
        features = np.random.randn(N, D).astype(np.float32)
        labels = np.random.choice([0, 1], size=N, p=[0.8, 0.2])
        labels[:10] = 1
        labels[10:20] = 0

        suite = DownstreamBenchmarkSuite(n_splits=3, max_iter=50)
        bin_res = suite.benchmark_binary_detection(features, labels)
        self.assertIn("linear_probe", bin_res)
        self.assertIn("mlp_2layer", bin_res)
        self.assertIn("representation_gap", bin_res)

        depths = np.random.uniform(0.0, 1.0, size=N).astype(np.float32)
        depths[labels == 0] = 0.0
        reg_res = suite.benchmark_depth_regression(features, depths)
        self.assertIn("linear_probe", reg_res)
        self.assertIn("r2_score", reg_res["linear_probe"])

        sev_labels = np.random.choice([0, 1, 2, 3], size=N)
        sev_res = suite.benchmark_severity_classification(features, sev_labels)
        self.assertIn("linear_probe", sev_res)
        self.assertIn("macro_f1", sev_res["linear_probe"])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test all visualization functions
            p_map = np.random.uniform(0.0, 1.0, size=(20, 20))
            p_path = os.path.join(tmpdir, "1_Anomaly_Detection", "prob.png")
            plot_probability_heatmap(p_map, p_path, title="Test Prob")
            self.assertTrue(os.path.isfile(p_path))

            roc_path = os.path.join(tmpdir, "1_Anomaly_Detection", "roc.png")
            plot_roc_pr_curves([0, 0.5, 1], [0, 0.8, 1], 0.85, [1, 0.5, 0.2], [0, 0.6, 1], 0.6, roc_path, "Test ROC")
            self.assertTrue(os.path.isfile(roc_path))

            d_map = os.path.join(tmpdir, "2_Depth_Regression", "depth_map.png")
            plot_depth_regression_maps(p_map, p_map * 0.9, d_map, title="Test Depth")
            self.assertTrue(os.path.isfile(d_map))

            d_scat = os.path.join(tmpdir, "2_Depth_Regression", "scatter.png")
            plot_depth_calibration_scatter(depths, depths * 0.9, d_scat, title="Test Scatter")
            self.assertTrue(os.path.isfile(d_scat))

            cm_path = os.path.join(tmpdir, "3_Severity_Classification", "cm.png")
            plot_severity_confusion_matrix(sev_labels, sev_labels, cm_path, title="Test CM")
            self.assertTrue(os.path.isfile(cm_path))

            cka_path = os.path.join(tmpdir, "4_Liftoff_Invariance", "cka.png")
            mat = np.eye(3)
            plot_liftoff_cka_heatmap(mat, ["z1", "z2", "z3"], cka_path, title="Test CKA")
            self.assertTrue(os.path.isfile(cka_path))


if __name__ == "__main__":
    unittest.main()
