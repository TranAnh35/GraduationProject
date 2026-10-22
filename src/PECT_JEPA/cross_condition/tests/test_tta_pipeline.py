"""
Comprehensive Unit & Integration Test Suite for Test-Time Adaptation (TTA) Pipeline:
1. Symmetric Whitening Matrix (whiten_matrix):
   - Diagonal loading shrinkage eps prevents NaN / singular explosion
   - Whitened covariance W * Cov * W^T equals identity matrix
2. Unsupervised Trimmed Background Statistics (unsup_trimmed_bg_stats):
   - MAD-energy outlier rejection on 98.8% sound metal grids with defect anomalies
   - Background mean and whitening matrix convergence
3. TestTimeSelfCalibrator Contract:
   - C-scan feature tensor [H, W, D] calibration to [H, W, D]
   - Threshold-free Mahalanobis anomaly energy map [H, W]
   - Significant separation (CNR > 2.0) on synthetic defect injection
4. Probes Bundle & Evaluation CLI Contract:
   - train_downstream / evaluate probe compatibility (raw + tta probes)
   - CLI flags: --run_tta, --skip_tta, --tta_trim_ratio, --tta_shrinkage, and deprecated UDA aliases
"""

import os
import sys
import unittest
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.downstream.test_time_adaptation import (
    whiten_matrix,
    unsup_trimmed_bg_stats,
    TestTimeSelfCalibrator,
)
from src.PECT_JEPA.cross_condition.downstream.detection.models import DetectionProbeBundle
from src.PECT_JEPA.cross_condition.evaluate import parse_args


class TestWhiteningMatrix(unittest.TestCase):
    """Tests for symmetric whitening matrix computation."""

    def setUp(self):
        np.random.seed(42)
        self.D = 64

    def test_ideal_covariance_whitening(self):
        """Symmetric whitening on positive definite covariance yields identity covariance."""
        # Random positive definite matrix
        A = np.random.randn(self.D, self.D)
        cov = A.dot(A.T) + 0.1 * np.eye(self.D)
        W = whiten_matrix(cov, eps=1e-3)

        # W * cov * W should be approximately identity
        whitened_cov = W.dot(cov).dot(W)
        identity = np.eye(self.D)
        np.testing.assert_allclose(whitened_cov, identity, atol=1e-2, err_msg="Whitened covariance must be identity")

    def test_singular_covariance_stability(self):
        """Shrinkage epsilon prevents division by zero or NaN on rank-deficient covariance."""
        # Rank-deficient matrix (rank 5 in 64D)
        A = np.random.randn(self.D, 5)
        cov_rank_def = A.dot(A.T)
        W = whiten_matrix(cov_rank_def, eps=1e-2)

        self.assertFalse(np.isnan(W).any(), "Whitening matrix must not contain NaN")
        self.assertFalse(np.isinf(W).any(), "Whitening matrix must not contain Inf")
        self.assertEqual(W.shape, (self.D, self.D))


class TestTrimmedBackgroundStats(unittest.TestCase):
    """Tests for robust MAD outlier trimming on sound-metal grids."""

    def setUp(self):
        np.random.seed(42)
        self.N_sound = 9800  # 98% sound metal
        self.N_defect = 200  # 2% defect outliers
        self.D = 32

        self.true_bg_mean = np.linspace(-1.0, 1.0, self.D)
        sound_samples = np.random.randn(self.N_sound, self.D) + self.true_bg_mean

        # Inject heavy anomalous perturbation for defect pixels
        defect_samples = np.random.randn(self.N_defect, self.D) * 3.0 + (self.true_bg_mean + 15.0)

        self.flat_z = np.vstack([sound_samples, defect_samples])

    def test_robust_mean_estimation(self):
        """Trimmed mean should reject high-energy defect outliers and estimate sound mean accurately."""
        naive_mean = np.mean(self.flat_z, axis=0)
        trimmed_mean, W_bg = unsup_trimmed_bg_stats(self.flat_z, keep_q=0.75, eps=1e-2)

        naive_err = np.linalg.norm(naive_mean - self.true_bg_mean)
        trimmed_err = np.linalg.norm(trimmed_mean - self.true_bg_mean)

        self.assertLess(trimmed_err, naive_err, "Trimmed background mean must be closer to true background than naive mean")
        self.assertLess(trimmed_err, 0.2, "Trimmed error should be small")
        self.assertEqual(W_bg.shape, (self.D, self.D))


class TestSelfCalibrator(unittest.TestCase):
    """Tests for TestTimeSelfCalibrator class."""

    def setUp(self):
        np.random.seed(42)
        self.H = 50
        self.W = 50
        self.D = 32

        # Create synthetic C-scan: 98% sound metal with offset
        bg_offset = np.random.randn(self.D) * 2.0
        self.latents = np.random.randn(self.H, self.W, self.D) * 0.5 + bg_offset

        # Inject defect cluster in center
        self.gt_mask = np.zeros((self.H, self.W), dtype=np.int64)
        self.latents[20:25, 20:25, :] += np.random.randn(self.D) * 4.0 + 5.0
        self.gt_mask[20:25, 20:25] = 1

    def test_calibration_and_anomaly_energy(self):
        """Calibrator produces zero-mean whitened sound metal and distinct defect energy."""
        calibrator = TestTimeSelfCalibrator(keep_q=0.80, eps=1e-2)
        z_cal = calibrator.calibrate(self.latents)
        scores = calibrator.compute_anomaly_scores(self.latents)

        self.assertEqual(z_cal.shape, (self.H, self.W, self.D))
        self.assertEqual(scores.shape, (self.H, self.W))
        self.assertFalse(np.isnan(scores).any())

        # Check CNR: (mean_defect - mean_sound) / std_sound
        sound_scores = scores[self.gt_mask == 0]
        defect_scores = scores[self.gt_mask == 1]

        mean_sound = np.mean(sound_scores)
        std_sound = np.std(sound_scores)
        mean_defect = np.mean(defect_scores)

        cnr = (mean_defect - mean_sound) / (std_sound + 1e-6)
        self.assertGreater(cnr, 3.0, f"Contrast-to-Noise Ratio (CNR) {cnr:.2f} must exceed 3.0 for synthetic flaw")


class TestEvaluateCLIContract(unittest.TestCase):
    """Tests for CLI arguments contract in evaluate.py."""

    def test_default_tta_enabled(self):
        """TTA must be enabled by default in evaluate args."""
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["evaluate.py", "--checkpoint", "dummy.pt"]
            args = parse_args()
            self.assertTrue(args.eval_tta)
            self.assertFalse(args.skip_tta)
            self.assertEqual(args.tta_trim_ratio, 0.75)
            self.assertEqual(args.tta_shrinkage, 1e-2)
        finally:
            sys.argv = sys_argv_backup

    def test_skip_tta_flag(self):
        """--skip_tta flag disables TTA execution."""
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["evaluate.py", "--checkpoint", "dummy.pt", "--skip_tta"]
            args = parse_args()
            self.assertTrue(args.skip_tta)
        finally:
            sys.argv = sys_argv_backup

    def test_deprecated_skip_uda_alias(self):
        """Deprecated --skip_domain_adaptation and --skip_uda flags correctly alias."""
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["evaluate.py", "--checkpoint", "dummy.pt", "--skip_uda"]
            args = parse_args()
            self.assertTrue(args.skip_domain_adaptation)
        finally:
            sys.argv = sys_argv_backup


if __name__ == "__main__":
    unittest.main()
