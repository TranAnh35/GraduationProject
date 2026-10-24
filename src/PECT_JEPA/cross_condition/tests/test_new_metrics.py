"""
Unit tests for new PECT-JEPA evaluation metrics:
1. compute_otsu_threshold
2. compute_instance_pod (Instance-level defect detection rate / POD)
3. benchmark_segmentation_iou (Otsu adaptive IoU, fixed IoU 0.5, Sound Metal FPR, F1-Max)
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import (
    compute_otsu_threshold,
    compute_instance_pod,
    DownstreamBenchmarkSuite,
)
from src.PECT_JEPA.cross_condition.evaluation.raw_signal_features import (
    filter_cscan_waveforms,
    extract_cscan_raw_features,
    extract_all_raw_heatmaps,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.visualizations import AsyncPlotSaver, Figure
from unittest.mock import MagicMock


class TestNewEvaluationMetrics(unittest.TestCase):
    def test_compute_otsu_threshold(self):
        """Verify Otsu threshold correctly separates bimodal distributions."""
        # 900 background samples around 0.1, 100 defect samples around 0.9
        rng = np.random.RandomState(42)
        bg = rng.normal(0.1, 0.05, 900)
        fg = rng.normal(0.9, 0.05, 100)
        vals = np.clip(np.concatenate([bg, fg]), 0.0, 1.0)

        tau = compute_otsu_threshold(vals, n_bins=256)
        # Threshold should be located between 0.2 and 0.8
        self.assertGreater(tau, 0.25)
        self.assertLess(tau, 0.75)

    def test_compute_instance_pod(self):
        """Verify instance-level Probability of Detection on mock CAD flaws."""
        H, W = 100, 100
        score_map = np.zeros((H, W), dtype=np.float32)

        # Mock config with 4 defects
        mock_config = {
            "features": [
                {"id": 1, "kind": "corrosion", "x": 20.0, "y": 20.0, "diameter": 8.0, "depth": 0.5},
                {"id": 2, "kind": "corrosion", "x": 50.0, "y": 20.0, "diameter": 6.0, "depth": 0.3},
                {"id": 3, "kind": "corrosion", "x": 80.0, "y": 20.0, "diameter": 4.0, "depth": 0.2},
                {"id": 4, "kind": "corrosion", "x": 20.0, "y": 70.0, "diameter": 10.0, "depth": 1.0},
                # Sound rivet (should be ignored)
                {"id": 5, "kind": "rivet only", "x": 50.0, "y": 70.0, "diameter": None, "depth": None},
            ]
        }

        # Place high detection scores on defects 1, 2, and 4; leave defect 3 at zero
        # Note: score_map is [row, col] where row=y, col=x
        # Defect 1: x=20, y=20
        score_map[18:23, 18:23] = 0.95  # Defect 1: Hit
        # Defect 2: x=50, y=20 -> row ~ 20, col ~ 50
        score_map[18:23, 48:53] = 0.85  # Defect 2: Hit
        # Defect 3: x=80, y=20 -> row ~ 20, col ~ 80 (Miss: left at 0.0)
        # Defect 4: x=20, y=70 -> row ~ 70, col ~ 20
        score_map[68:73, 18:23] = 0.90  # Defect 4: Hit

        res = compute_instance_pod(score_map, mock_config, threshold=0.5, margin_px=2)
        self.assertEqual(res["num_defects"], 4)
        self.assertEqual(res["num_hits"], 3)
        self.assertAlmostEqual(res["pod"], 0.75, places=3)
        self.assertEqual(len(res["hit_details"]), 4)
        self.assertTrue(res["hit_details"][0]["hit"])
        self.assertTrue(res["hit_details"][1]["hit"])
        self.assertFalse(res["hit_details"][2]["hit"])
        self.assertTrue(res["hit_details"][3]["hit"])

    def test_benchmark_segmentation_iou(self):
        """Verify Otsu and fixed 0.5 IoU, Dice, F1-Max, and Sound Metal FPR."""
        suite = DownstreamBenchmarkSuite()

        gt_mask = np.zeros((100, 100), dtype=np.int64)
        gt_mask[30:50, 30:50] = 1  # 400 defect pixels, 9600 sound pixels

        prob_map = np.full((100, 100), 0.05, dtype=np.float32)
        # Put high probability on the defect area with small bleed into sound metal
        prob_map[30:50, 30:50] = 0.92
        prob_map[28:30, 28:30] = 0.60  # 4 false positive pixels

        res = suite.benchmark_segmentation_iou(prob_map, gt_mask)

        self.assertIn("iou_05", res)
        self.assertIn("dice_05", res)
        self.assertIn("iou_otsu", res)
        self.assertIn("dice_otsu", res)
        self.assertIn("sound_fpr_05", res)
        self.assertIn("sound_fpr_otsu", res)
        self.assertIn("f1_max", res)

        # Defect detection should have high IoU and low sound metal FPR
        self.assertGreater(res["iou_05"], 0.8)
        self.assertGreater(res["iou_otsu"], 0.8)
        self.assertLess(res["sound_fpr_05"], 0.01)
        self.assertGreater(res["f1_max"], 0.9)

    def test_filter_cscan_waveforms(self):
        """Verify Butterworth LPF and DC baseline subtraction."""
        rng = np.random.RandomState(42)
        # Synthetic cscan [20, 20, 128] with DC offset 5.0 and high freq noise
        t = np.linspace(0, 1, 128)
        signal = np.sin(2 * np.pi * 3 * t)  # 3 Hz smooth signal
        noise = 0.5 * rng.randn(20, 20, 128)
        raw_cscan = 5.0 + signal[None, None, :] + noise

        filtered = filter_cscan_waveforms(raw_cscan, cutoff=0.15, order=2, subtract_baseline=True)
        self.assertEqual(filtered.shape, raw_cscan.shape)
        # First sample after baseline subtraction should be near 0
        self.assertTrue(np.allclose(filtered[:, :, 0], 0.0, atol=1e-5))
        # High frequency noise variance should be substantially reduced
        self.assertLess(float(np.var(np.diff(filtered, axis=-1))), float(np.var(np.diff(raw_cscan, axis=-1))))

    def test_extract_cscan_raw_features_and_all_methods(self):
        """Verify the 4 reference methods extract proper 2D heatmaps."""
        t = np.linspace(0, 1, 128)
        signal = np.sin(2 * np.pi * 3 * t)
        mock_cscan = np.zeros((30, 30, 128), dtype=np.float32)
        mock_cscan[:] = signal[None, None, :]
        # Add peak in center
        mock_cscan[10:20, 10:20, :] *= 2.0

        for method in ["slice", "loi", "ptp", "filtered_ptp"]:
            feat = extract_cscan_raw_features(mock_cscan, method=method)
            self.assertEqual(feat.shape, (30, 30))
            self.assertEqual(feat.dtype, np.float32)
            self.assertFalse(np.isnan(feat).any())

        all_feats = extract_all_raw_heatmaps(mock_cscan)
        self.assertIn("slice", all_feats)
        self.assertIn("loi", all_feats)
        self.assertIn("ptp", all_feats)
        self.assertIn("filtered_ptp", all_feats)
        self.assertEqual(len(all_feats), 4)

    def test_async_plot_saver(self):
        """Verify AsyncPlotSaver non-blocking execution and wait_all."""
        import tempfile
        saver = AsyncPlotSaver(max_workers=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_async.png")

            def dummy_save(path):
                fig = Figure(figsize=(4, 4))
                ax = fig.subplots()
                ax.plot([0, 1], [0, 1])
                fig.savefig(path)

            fut = saver.submit(dummy_save, out_path)
            saver.wait_all()
            saver.shutdown()
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_cscan_feature_extractor_coord_caching(self):
        """Verify CScanFeatureExtractor caches coordinate batches across calls."""
        import torch
        mock_model = MagicMock()
        mock_model.config.star_radii = (1, 3, 7)
        mock_model.config.latent_disentanglement = False
        mock_model.config.embed_dim = 64

        extractor = CScanFeatureExtractor(model=mock_model, device=torch.device("cpu"), batch_size=2048)
        n1, b1 = extractor._get_coord_batches(270, 270, pad=7)
        n2, b2 = extractor._get_coord_batches(270, 270, pad=7)

        self.assertEqual(n1, 270 * 270)
        self.assertIs(b1, b2)  # Identical cached object reference


if __name__ == "__main__":
    unittest.main()
