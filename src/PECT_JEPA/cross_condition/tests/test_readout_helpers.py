"""
Readout-standardization contract tests (work order item 3).

Covers the shared helpers that `evaluate.py`, `train_downstream.py` and
`diagnose_readouts.py` now all import from `evaluation/downstream_benchmarks.py`
plus the canonical TTA estimator in `downstream/test_time_adaptation.py`:

1. `halo_ring` / `mine_hard_negatives` (P2h): partial-volume boundary pixels are
   never fitted or mined as confident negatives.
2. `whiten_stats` / `unsup_bg_stats` / `scan_adaptive_features` (P1u_trim_s2):
   per-scan unsupervised background whitening, zero GT labels.
3. Log-target sizing: `fit_global_readouts(size_log_target=True)` records
   `size_target="log1p"`, `SizingRegressorBundle` inverts with expm1, and
   `benchmark_sizing_regression(log_target=True)` reports mm, not log-mm.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import (
    HALO_RADIUS_PX,
    DownstreamBenchmarkSuite,
    halo_ring,
    mine_hard_negatives,
    scan_adaptive_features,
    unsup_bg_stats,
    whiten_stats,
)
from src.PECT_JEPA.cross_condition.downstream.sizing.models import SizingRegressorBundle


class TestHaloRing(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((40, 40), dtype=np.int64)
        self.gt[10:20, 10:20] = 1

    def test_ring_is_boundary_only(self):
        ring = halo_ring(self.gt)
        self.assertEqual(ring.dtype, bool)
        self.assertGreater(int(ring.sum()), 0)
        self.assertFalse(np.any(ring & (self.gt == 1)), "halo ring must not include defect pixels")

    def test_ring_matches_dilation_minus_defect(self):
        from scipy import ndimage as ndi
        expected = ndi.binary_dilation(self.gt == 1, iterations=HALO_RADIUS_PX) & (self.gt != 1)
        np.testing.assert_array_equal(halo_ring(self.gt), expected)

    def test_no_defect_means_empty_ring(self):
        self.assertEqual(int(halo_ring(np.zeros((10, 10), dtype=np.int64)).sum()), 0)


class TestMineHardNegatives(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((40, 40), dtype=np.int64)
        self.gt[10:20, 10:20] = 1
        self.ring = halo_ring(self.gt).reshape(-1)
        # Halo pixels score highest (they are the label-noise trap), sound far lower.
        self.scores = np.full(40 * 40, 0.2, dtype=np.float32)
        self.scores[self.ring] = 0.99

    def test_ring_pixels_are_never_mined(self):
        idx = mine_hard_negatives(self.gt, self.scores, k=50, halo_radius=HALO_RADIUS_PX)
        self.assertEqual(len(idx), 50)
        flat_gt = self.gt.reshape(-1)
        self.assertTrue(np.all(flat_gt[idx] == 0), "mined a defect pixel as a negative")
        self.assertFalse(np.any(self.ring[idx]), "mined a halo (partial-volume) pixel as a negative")

    def test_k_is_capped_by_available_sound_pixels(self):
        idx = mine_hard_negatives(self.gt, self.scores, k=10 ** 9, halo_radius=HALO_RADIUS_PX)
        flat_gt = self.gt.reshape(-1)
        expected = int(np.sum((flat_gt == 0) & ~self.ring))
        self.assertEqual(len(idx), expected)

    def test_no_sound_pixels_returns_empty(self):
        gt_full = np.ones((10, 10), dtype=np.int64)
        idx = mine_hard_negatives(gt_full, np.zeros(100, dtype=np.float32), k=10)
        self.assertEqual(len(idx), 0)


class TestWhiteningHelpers(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.D = 8
        self.L = self.rng.randn(self.D, self.D) * 0.5
        self.mu_true = np.linspace(-1.0, 1.0, self.D)
        self.bg = self.rng.randn(4000, self.D).dot(self.L.T) + self.mu_true

    def test_whiten_stats_yields_identity_covariance(self):
        mu, W = whiten_stats(self.bg)
        np.testing.assert_allclose(mu, self.bg.mean(axis=0), atol=1e-5)
        cov_w = W.dot(np.cov(self.bg, rowvar=False)).dot(W)
        np.testing.assert_allclose(cov_w, np.eye(self.D), atol=2e-2)

    def test_unsup_bg_stats_respects_valid_mask(self):
        flat = np.vstack([self.bg, np.full((10, self.D), 1e3)])
        mask = np.zeros(len(flat), dtype=bool)
        mask[:len(self.bg)] = True
        mu, W = unsup_bg_stats(flat, mask)
        mu_ref, W_ref = whiten_stats(flat[mask])
        np.testing.assert_allclose(mu, mu_ref, atol=1e-6)
        np.testing.assert_allclose(W, W_ref, atol=1e-6)

    def test_scan_adaptive_features_is_finite_and_zero_centered(self):
        flat = self.bg.astype(np.float32)
        out = scan_adaptive_features(flat, keep_q=0.75, eps=1e-2)
        self.assertEqual(out.shape, flat.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())
        # mu is estimated on the lowest-energy 75% of valid pixels, i.e. the
        # background cloud itself, so the whitened cloud stays near zero mean.
        self.assertTrue(np.all(np.abs(out.mean(axis=0)) < 0.5),
                        f"whitened background is not zero-centered: {out.mean(axis=0)}")

    def test_scan_adaptive_features_valid_mask_does_not_change_shape(self):
        flat = self.bg.astype(np.float32)
        mask = np.ones(len(flat), dtype=bool)
        mask[-50:] = False
        out = scan_adaptive_features(flat, valid_mask=mask, keep_q=0.75, eps=1e-2)
        self.assertEqual(out.shape, flat.shape)


class TestLogTargetSizing(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(7)
        self.D = 6
        # Diameters spanning the real 2-14 mm range, log-space signal in X[:, 0]
        # (perfectly predictable), noise in the remaining features.
        self.y = np.expm1(np.linspace(np.log1p(2.0), np.log1p(14.0), 300)).astype(np.float32)
        X = rng.randn(300, self.D)
        X[:, 0] = np.log1p(self.y)
        self.X = X.astype(np.float32)
        self.X_det = rng.randn(200, self.D).astype(np.float32)
        self.y_det = (np.arange(200) % 2).astype(np.int64)

    def test_fit_global_readouts_records_log1p_target(self):
        suite = DownstreamBenchmarkSuite(n_splits=3, random_state=42)
        readouts = suite.fit_global_readouts(
            X_det=self.X_det, y_det=self.y_det, X_size=self.X, y_size=self.y, size_log_target=True)
        self.assertEqual(readouts["size_target"], "log1p")
        self.assertIn("scaler_size", readouts)
        self.assertIn("ridge_size", readouts)

        readouts_id = suite.fit_global_readouts(
            X_det=self.X_det, y_det=self.y_det, X_size=self.X, y_size=self.y, size_log_target=False)
        self.assertEqual(readouts_id["size_target"], "identity")

    def test_sizing_bundle_inverts_log1p(self):
        suite = DownstreamBenchmarkSuite(n_splits=3, random_state=42)
        readouts = suite.fit_global_readouts(
            X_det=self.X_det, y_det=self.y_det, X_size=self.X, y_size=self.y, size_log_target=True)
        bundle = SizingRegressorBundle(
            scaler=readouts["scaler_size"],
            ridge_regressor=readouts["ridge_size"],
            mlp_regressor=None,
            target_transform=readouts["size_target"],
        )
        pred = bundle.predict_ridge(self.X)
        self.assertEqual(pred.shape, (len(self.X),))
        self.assertTrue(np.all(pred >= 0.0))
        # Inverted predictions must live in mm, not log1p(mm).
        self.assertGreater(float(np.mean(pred)), 1.0)
        self.assertLess(float(np.mean(np.abs(pred - self.y))), 2.0,
                        "log1p inversion lost the mm scale")

    def test_benchmark_sizing_regression_reports_mm(self):
        suite = DownstreamBenchmarkSuite(n_splits=3, random_state=42)
        res = suite.benchmark_sizing_regression(self.X, self.y, log_target=True)
        self.assertNotIn("error", res)
        linear = res["linear_probe"]
        self.assertGreater(linear["r2_score"], 0.9)
        self.assertLess(linear["mae_mm"], 1.0, "MAE looks like log-mm, not mm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
