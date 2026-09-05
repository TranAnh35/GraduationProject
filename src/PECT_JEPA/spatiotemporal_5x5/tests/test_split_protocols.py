"""
Unit tests for Dataset Splitting Protocols in 5x5 PECT-JEPA.
Verifies all 5 protocols on the actual data directory:
- LOLO (Leave-One-Lift-off-Out)
- LOSO (Leave-One-Sensor-Out)
- LOWO (Leave-One-Waveform-Out)
- LODO (Leave-One-Specimen-Out)
- In-Domain Random Split
"""

import unittest
import glob
import os

from ..data.split import (
    extract_file_metadata,
    split_compound_ood,
    split_leave_one_liftoff,
    split_leave_one_sensor,
    split_leave_one_waveform,
    split_leave_one_specimen,
    get_dataset_split,
)


class TestDatasetSplitProtocols(unittest.TestCase):

    def setUp(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        data_dir = os.path.join(root_dir, "data")
        pattern = os.path.join(data_dir, "**", "*.tdms")
        self.files = sorted([f for f in glob.glob(pattern, recursive=True) if not f.endswith(".tdms_index")])

    def test_metadata_extraction(self):
        """Verify that all files are parsed into valid categories."""
        self.assertGreater(len(self.files), 0, "Expected TDMS files in data/")
        for fp in self.files:
            m = extract_file_metadata(fp)
            self.assertIn(m["sensor"], ["Hall_Air_Core", "Hall_Pot_Core", "TMR"])
            self.assertIn(m["specimen"], ["Corrosion", "Rivet_v1", "Rivet_v2"])
            self.assertIn(m["waveform"], ["Chirp", "Gaussian", "Square"])
            self.assertIn(m["liftoff"], ["z1", "z2", "z3"])

    def test_leave_one_liftoff(self):
        """Verify Leave-One-Lift-off-Out strictly isolates z3 without contamination."""
        train, val, test = split_leave_one_liftoff(self.files, test_liftoff="z3", val_ratio=0.1)
        self.assertEqual(len(train) + len(val) + len(test), len(self.files))
        self.assertEqual(len(test), 27)

        # Ensure NO z3 in train or val
        for fp in train + val:
            m = extract_file_metadata(fp)
            self.assertNotEqual(m["liftoff"], "z3", f"Leakage: z3 file found in training/val: {fp}")

        # Ensure ALL test files are z3
        for fp in test:
            m = extract_file_metadata(fp)
            self.assertEqual(m["liftoff"], "z3")

    def test_leave_one_sensor(self):
        """Verify Leave-One-Sensor-Out strictly isolates TMR."""
        train, val, test = split_leave_one_sensor(self.files, test_sensor="TMR", val_ratio=0.1)
        self.assertEqual(len(train) + len(val) + len(test), len(self.files))
        self.assertEqual(len(test), 27)

        for fp in train + val:
            m = extract_file_metadata(fp)
            self.assertNotEqual(m["sensor"], "TMR")
        for fp in test:
            m = extract_file_metadata(fp)
            self.assertEqual(m["sensor"], "TMR")

    def test_leave_one_waveform(self):
        """Verify Leave-One-Waveform-Out strictly isolates Chirp."""
        train, val, test = split_leave_one_waveform(self.files, test_waveform="Chirp", val_ratio=0.1)
        self.assertEqual(len(train) + len(val) + len(test), len(self.files))
        self.assertEqual(len(test), 27)

        for fp in train + val:
            m = extract_file_metadata(fp)
            self.assertNotEqual(m["waveform"], "Chirp")
        for fp in test:
            m = extract_file_metadata(fp)
            self.assertEqual(m["waveform"], "Chirp")

    def test_leave_one_specimen(self):
        """Verify Leave-One-Specimen-Out strictly isolates Rivet_v2."""
        train, val, test = split_leave_one_specimen(self.files, test_specimen="Rivet_v2", val_ratio=0.1)
        self.assertEqual(len(train) + len(val) + len(test), len(self.files))
        self.assertEqual(len(test), 27)

        for fp in train + val:
            m = extract_file_metadata(fp)
            self.assertNotEqual(m["specimen"], "Rivet_v2")
        for fp in test:
            m = extract_file_metadata(fp)
            self.assertEqual(m["specimen"], "Rivet_v2")

    def test_compound_ood(self):
        """Verify Option A: Compound OOD with z3, TMR, Chirp holdout."""
        train, val, test, summary = split_compound_ood(
            self.files,
            holdout_liftoff="z3",
            holdout_sensor="TMR",
            holdout_waveform="Chirp",
            val_files_count=4,
            seed=42,
        )

        self.assertEqual(len(train) + len(val) + len(test), 81)
        self.assertEqual(len(train), 20, "Expected 20 base domain training files")
        self.assertEqual(len(val), 4, "Expected 4 base domain validation files")
        self.assertEqual(len(test), 57, "Expected 57 held-out OOD test files")

        # Ensure NO z3, TMR, or Chirp in train or val
        for fp in train + val:
            m = extract_file_metadata(fp)
            self.assertNotEqual(m["liftoff"], "z3", f"Leakage: z3 in training/val: {fp}")
            self.assertNotEqual(m["sensor"], "TMR", f"Leakage: TMR in training/val: {fp}")
            self.assertNotEqual(m["waveform"], "Chirp", f"Leakage: Chirp in training/val: {fp}")

        # Check all 3 specimens exist in both train and val
        train_specs = {extract_file_metadata(fp)["specimen"] for fp in train}
        val_specs = {extract_file_metadata(fp)["specimen"] for fp in val}
        self.assertEqual(train_specs, {"Corrosion", "Rivet_v1", "Rivet_v2"})
        self.assertEqual(val_specs, {"Corrosion", "Rivet_v1", "Rivet_v2"})

        # Check slices in test set
        slices = summary["test_slices"]
        self.assertEqual(len(slices["single_liftoff"]), 12)
        self.assertEqual(len(slices["single_sensor"]), 12)
        self.assertEqual(len(slices["single_waveform"]), 12)
        self.assertEqual(len(slices["compound_double"]), 18)
        self.assertEqual(len(slices["compound_triple"]), 3)

    def test_unified_dispatcher(self):
        """Verify the unified dispatcher functions for all protocol names."""
        for proto, target, expected_test_len in [
            ("compound_ood", None, 57),
            ("leave_liftoff", "z3", 27),
            ("leave_sensor", "TMR", 27),
            ("leave_waveform", "Square", 27),
            ("leave_specimen", "Corrosion", 27),
        ]:
            train, val, test, summary = get_dataset_split(self.files, protocol=proto, holdout_target=target)
            self.assertEqual(len(test), expected_test_len)
            self.assertEqual(summary["protocol"], proto)


if __name__ == "__main__":
    unittest.main()
