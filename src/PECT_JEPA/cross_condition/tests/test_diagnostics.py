"""
Diagnostics robustness contract (crash fix for LAPACK info=62).

`compute_svd_diagnostics` used to run `torch.linalg.eigvalsh` on a float32
covariance built straight from whatever the model emitted. Two inputs killed a
multi-hour Colab run inside the first epoch's telemetry:

1. non-finite rows (one AMP overflow batch) -> `_LinAlgError: linalg.eigh: The
   algorithm failed to converge ... (error code: 62)`;
2. a fp16 tensor left behind by autocast -> `NotImplementedError: linalg_eigh_cpu
   not implemented for 'Half'`.

Telemetry must never abort training, so these tests pin: no exception, finite
statistics, dropped rows counted, and the SVD fallback matching the eigensolver.
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.training.diagnostics import (  # noqa: E402
    compute_latent_snr,
    compute_svd_diagnostics,
)


class TestSvdDiagnosticsRobustness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(512, 64)

    def test_clean_input_matches_reference_spectrum(self):
        res = compute_svd_diagnostics(self.z)
        self.assertEqual(res["n_nonfinite_rows"], 0)
        self.assertTrue(0.0 < res["effective_rank"] <= 64.0)
        self.assertTrue(res["condition_number"] > 1.0)
        self.assertAlmostEqual(res["dimension"], 64.0)

    def test_nonfinite_rows_do_not_crash_and_are_counted(self):
        z = torch.cat([self.z[:502], torch.full((10, 64), float("nan"))])
        res = compute_svd_diagnostics(z)
        self.assertEqual(res["n_nonfinite_rows"], 10)
        self.assertTrue(torch.isfinite(torch.tensor(res["effective_rank"])))
        self.assertGreater(res["effective_rank"], 1.0)

    def test_inf_rows_do_not_crash(self):
        z = self.z.clone()
        z[3, 7] = float("inf")
        res = compute_svd_diagnostics(z)
        self.assertEqual(res["n_nonfinite_rows"], 1)

    def test_extreme_scale_does_not_crash(self):
        # Old path overflowed the fp32 covariance -> LAPACK info=62.
        res = compute_svd_diagnostics(self.z * 1e18)
        self.assertTrue(torch.isfinite(torch.tensor(res["effective_rank"])))
        self.assertTrue(torch.isfinite(torch.tensor(res["condition_number"])))

    def test_fp16_input_is_supported(self):
        res = compute_svd_diagnostics(self.z.half())
        self.assertTrue(torch.isfinite(torch.tensor(res["effective_rank"])))
        self.assertGreater(res["effective_rank"], 1.0)

    def test_3d_windows_are_flattened(self):
        res = compute_svd_diagnostics(self.z.reshape(32, 16, 64))
        self.assertGreater(res["effective_rank"], 1.0)

    def test_svd_fallback_matches_eigvalsh(self):
        ref = compute_svd_diagnostics(self.z)

        real = torch.linalg.eigvalsh
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("linalg.eigh: failed to converge (simulated info=62)")

        torch.linalg.eigvalsh = boom
        try:
            got = compute_svd_diagnostics(self.z)
        finally:
            torch.linalg.eigvalsh = real

        self.assertEqual(calls["n"], 1, "fallback path was not exercised")
        self.assertAlmostEqual(ref["effective_rank"], got["effective_rank"], places=6)
        self.assertAlmostEqual(ref["top1_ratio"], got["top1_ratio"], places=6)

    def test_all_nonfinite_returns_safe_defaults(self):
        res = compute_svd_diagnostics(torch.full((8, 16), float("nan")))
        self.assertEqual(res["effective_rank"], 0.0)
        self.assertEqual(res["top1_ratio"], 0.0)
        self.assertEqual(res["condition_number"], float("inf"))
        self.assertEqual(res["n_nonfinite_rows"], 8)

    def test_single_row_returns_safe_defaults(self):
        res = compute_svd_diagnostics(torch.randn(1, 64))
        self.assertEqual(res["effective_rank"], 0.0)
        self.assertEqual(res["dimension"], 64.0)

    def test_rank_deficient_input_is_stable(self):
        z = torch.randn(256, 1) @ torch.randn(1, 64)
        res = compute_svd_diagnostics(z)
        self.assertTrue(torch.isfinite(torch.tensor(res["effective_rank"])))
        self.assertLess(res["effective_rank"], 2.0)

    def test_effective_rank_is_scale_invariant(self):
        a = compute_svd_diagnostics(self.z)
        b = compute_svd_diagnostics(self.z * 37.0)
        self.assertAlmostEqual(a["effective_rank"], b["effective_rank"], places=6)


class TestLatentSnr(unittest.TestCase):
    def test_separable_latents_have_positive_snr(self):
        torch.manual_seed(0)
        z = torch.cat([torch.randn(100, 8), torch.randn(100, 8) + 10.0])
        y = torch.cat([torch.zeros(100), torch.ones(100)])
        self.assertGreater(compute_latent_snr(z, y), 5.0)

    def test_missing_class_returns_zero(self):
        self.assertEqual(compute_latent_snr(torch.randn(10, 4), torch.zeros(10)), 0.0)


if __name__ == "__main__":
    unittest.main()
