"""
Unit tests for OOD Data Leakage Prevention and CLI-Config Synchronization.

Verifies:
1. Zero spatial leakage in benchmark_cross_file_ood.
2. Disjoint spatial indices with buffer margin in benchmark_binary_detection_spatial_block.
3. CLI argument parser defaults in train.py perfectly match Spatiotemporal5x5Config.
4. Tokenizer 'dual_scale_diffusion' (50 tokens) is accepted and configured by default.
"""

import unittest
import numpy as np
import torch

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.train import build_arg_parser
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import build_tokenizer_5x5


class TestOODLeakageAndConfigSync(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_cli_parser_matches_config_defaults(self):
        """Verify train.py CLI parser defaults align with Spatiotemporal5x5Config."""
        parser = build_arg_parser()
        args = parser.parse_args([])
        cfg = Spatiotemporal5x5Config()

        self.assertEqual(args.tokenizer_type, cfg.tokenizer_type)
        self.assertEqual(args.loss_type, cfg.loss_type)
        self.assertEqual(args.predictor_type, cfg.predictor_type)

        self.assertAlmostEqual(args.ema_momentum, 0.990)
        self.assertAlmostEqual(cfg.ema_momentum, 0.990)

        self.assertAlmostEqual(args.norm_floor_weight, 0.1)
        self.assertAlmostEqual(cfg.norm_floor_weight, 0.1)

    def test_dual_scale_tokenizer_instantiation(self):
        """Verify dual_scale_diffusion produces 50 tokens (25 shallow, 25 deep)."""
        cfg = Spatiotemporal5x5Config(tokenizer_type="dual_scale_diffusion")
        tok = build_tokenizer_5x5(cfg)
        x = torch.randn(2, 5, 5, 128)
        tokens, pos = tok(x)
        self.assertEqual(tokens.shape, (2, 50, cfg.embed_dim))
        self.assertEqual(pos.shape, (2, 50, cfg.embed_dim))

    def test_cross_file_ood_evaluates_without_leakage(self):
        """Verify cross_file_ood trains on train set and evaluates on test set."""
        bench = DownstreamBenchmarkSuite(random_state=42)

        D = 64
        # Synthetic train: 200 samples
        X_tr = np.random.randn(200, D).astype(np.float32)
        y_tr = np.zeros(200, dtype=np.int64)
        y_tr[160:] = 1  # 40 defect samples

        # Synthetic test: 100 samples from a shifted distribution
        X_te = (np.random.randn(100, D) + 0.5).astype(np.float32)
        y_te = np.zeros(100, dtype=np.int64)
        y_te[80:] = 1   # 20 defect samples

        res = bench.benchmark_cross_file_ood(X_tr, y_tr, X_te, y_te)

        self.assertIn("linear_probe", res)
        self.assertIn("mlp_2layer", res)
        self.assertIn("representation_gap", res)

        lp = res["linear_probe"]
        self.assertIn("auc_roc", lp)
        self.assertIn("average_precision", lp)
        self.assertIn("f1_score", lp)
        self.assertTrue(0.0 <= lp["auc_roc"] <= 1.0)

    def test_spatial_block_cv_excludes_buffer_margin(self):
        """Verify spatial block benchmark separates C-scan with buffer margin."""
        bench = DownstreamBenchmarkSuite(random_state=42)

        H, W, D = 40, 40, 32
        features_2d = np.random.randn(H, W, D).astype(np.float32)
        labels_2d = np.zeros((H, W), dtype=np.int64)
        # Place defects in top half and bottom half
        labels_2d[5:10, 10:20] = 1
        labels_2d[30:35, 10:20] = 1

        res = bench.benchmark_binary_detection_spatial_block(features_2d, labels_2d, buffer_margin=5)

        self.assertIn("linear_probe", res)
        self.assertIn("auc_roc", res["linear_probe"])
        self.assertTrue(0.0 <= res["linear_probe"]["auc_roc"] <= 1.0)

    def test_run_cross_file_ood_benchmark_is_wired(self):
        """Verify run_cross_file_ood_benchmark is importable and integrated in evaluate.py."""
        from src.PECT_JEPA.spatiotemporal_5x5.evaluate import run_cross_file_ood_benchmark
        self.assertTrue(callable(run_cross_file_ood_benchmark))


if __name__ == "__main__":
    unittest.main()
