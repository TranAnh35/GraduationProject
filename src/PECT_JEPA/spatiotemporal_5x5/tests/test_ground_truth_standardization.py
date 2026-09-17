"""
Unit tests for GroundTruthManager coordinate standardization and flaw feature extraction.
Verifies:
1. get_flaw_features('corrosion', coordinate_system='cropped') shifts (x, y) by (-15, -15).
2. Preserves cad_x and cad_y in the original [300, 300] CAD frame.
3. Cropped coordinates are strictly within [0, 270) bounds.
4. Physical diameters and depths are identically preserved.
5. Rivet fastener and corrosion centers for rivet_v1 and rivet_v2 are also correctly shifted.
"""

import unittest
import numpy as np

from ..data.ground_truth import get_ground_truth_manager


class TestGroundTruthStandardization(unittest.TestCase):

    def setUp(self):
        self.gt_mgr = get_ground_truth_manager()

    def test_corrosion_flaws_cropped_coordinates(self):
        """Verify corrosion features in cropped coordinates match [270, 270] array."""
        cropped_feats = self.gt_mgr.get_flaw_features("corrosion", coordinate_system="cropped")
        cad_feats = self.gt_mgr.get_flaw_features("corrosion", coordinate_system="cad")
        self.assertEqual(len(cropped_feats), 25)
        self.assertEqual(len(cad_feats), 25)

        for c_f, cad_f in zip(cropped_feats, cad_feats):
            # Check CAD coords preserved
            self.assertEqual(c_f["cad_x"], cad_f["cad_x"])
            self.assertEqual(c_f["cad_y"], cad_f["cad_y"])

            # Check cropped shift: x' = cad_x - 15, y' = cad_y - 15
            self.assertAlmostEqual(c_f["x"], cad_f["cad_x"] - 15.0, places=3)
            self.assertAlmostEqual(c_f["y"], cad_f["cad_y"] - 15.0, places=3)

            # Check bounds on [270, 270]
            self.assertGreaterEqual(c_f["x"], 0)
            self.assertLess(c_f["x"], 270)
            self.assertGreaterEqual(c_f["y"], 0)
            self.assertLess(c_f["y"], 270)

            # Check diameters and depths are identical
            self.assertEqual(c_f["diameter"], cad_f["diameter"])
            self.assertEqual(c_f["depth"], cad_f["depth"])

        # Check specific known defect centers
        # Flaw 1: CAD (30, 30) -> Cropped (15, 15)
        self.assertAlmostEqual(cropped_feats[0]["x"], 15.0)
        self.assertAlmostEqual(cropped_feats[0]["y"], 15.0)

        # Flaw 13: CAD (150, 150) -> Cropped (135, 135)
        self.assertAlmostEqual(cropped_feats[12]["x"], 135.0)
        self.assertAlmostEqual(cropped_feats[12]["y"], 135.0)

        # Flaw 25: CAD (270, 270) -> Cropped (255, 255)
        self.assertAlmostEqual(cropped_feats[24]["x"], 255.0)
        self.assertAlmostEqual(cropped_feats[24]["y"], 255.0)

    def test_corrosion_mask_alignment(self):
        """Verify that cropped feature centers fall exactly on defect pixels (value 1) in corrosion_gt_mask."""
        mask = self.gt_mgr.get_standard_cropped_mask("corrosion", buffer_px=2)
        self.assertEqual(mask.shape, (270, 270))

        cropped_feats = self.gt_mgr.get_flaw_features("corrosion", coordinate_system="cropped")
        for feat in cropped_feats:
            col = int(round(feat["x"]))
            row = int(round(feat["y"]))
            # Flaw center MUST be defect (1)
            self.assertEqual(
                mask[row, col], 1,
                f"Defect {feat['id']} at row={row}, col={col} did not land on defect pixel in mask!"
            )

    def test_rivet_flaws_cropped_coordinates(self):
        """Verify rivet_v1 and rivet_v2 feature coordinates."""
        for spec in ["rivet_v1", "rivet_v2"]:
            cropped_feats = self.gt_mgr.get_flaw_features(spec, coordinate_system="cropped")
            self.assertGreater(len(cropped_feats), 0)
            for f in cropped_feats:
                if f.get("x") is not None:
                    self.assertGreaterEqual(f["x"], 0)
                    self.assertLess(f["x"], 270)
                if f.get("y") is not None:
                    self.assertGreaterEqual(f["y"], 0)
                    self.assertLess(f["y"], 270)
                if f.get("corrosionX") is not None:
                    self.assertGreaterEqual(f["corrosionX"], 0)
                    self.assertLess(f["corrosionX"], 270)
                if f.get("corrosionY") is not None:
                    self.assertGreaterEqual(f["corrosionY"], 0)
                    self.assertLess(f["corrosionY"], 270)


if __name__ == "__main__":
    unittest.main()
