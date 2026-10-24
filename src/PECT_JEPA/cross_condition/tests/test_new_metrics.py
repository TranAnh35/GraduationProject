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


if __name__ == "__main__":
    unittest.main()
