"""
Split-protocol contract tests (work order item 1).

The `split_mode` switch is the only place where the source (Base) Domain is
defined, so a silent change here invalidates every downstream number. These
tests pin the two protocols on a synthetic 81-file tree built with the real
naming conventions (2 Hall sensors + TMR x 3 specimens x 3 waveforms x 3 liftoffs):

  'cross_specimen'  : Base 16 -> 12 train / 4 val, 65 test
  'cross_condition' : Base 24 (adds Rivet_v2) -> 18 train / 6 val, 57 test

Also pinned: determinism under a fixed seed, zero overlap, full coverage, and
the stratified 2-val-per-specimen composition of the new protocol.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.data.split import (
    SPLIT_MODES,
    check_split_mode,
    extract_file_metadata,
    filter_base_domain_files,
    get_canonical_train_val_test_splits,
    is_base_domain_file,
)

SENSORS = ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]
SPECIMENS = ["Corrosion", "Rivet_v1", "Rivet_v2"]
WAVEFORMS = ["Square", "Gaussian", "Chirp"]
LIFTOFFS = ["z1", "z2", "z3"]


def synthetic_files():
    files = []
    for spec in SPECIMENS:
        for sensor in SENSORS:
            for wf in WAVEFORMS:
                for z in LIFTOFFS:
                    files.append(f"data/{spec}/{sensor}_{wf}_{z}.tdms")
    return files


class TestSplitProtocols(unittest.TestCase):
    def setUp(self):
        self.files = synthetic_files()

    def test_dataset_geometry(self):
        self.assertEqual(len(self.files), 81)
        metas = [extract_file_metadata(f) for f in self.files]
        self.assertEqual({m["sensor"] for m in metas}, set(SENSORS))
        self.assertEqual({m["specimen"] for m in metas}, set(SPECIMENS))
        self.assertEqual({m["waveform"] for m in metas}, set(WAVEFORMS))
        self.assertEqual({m["liftoff"] for m in metas}, set(LIFTOFFS))

    def test_base_domain_sizes(self):
        self.assertEqual(len(filter_base_domain_files(self.files, "cross_specimen")), 16)
        self.assertEqual(len(filter_base_domain_files(self.files, "cross_condition")), 24)

    def test_cross_specimen_split(self):
        splits = get_canonical_train_val_test_splits(self.files, split_mode="cross_specimen", seed=42)
        self.assertEqual((len(splits["train"]), len(splits["val"]), len(splits["test"])), (12, 4, 65))

    def test_cross_condition_split(self):
        splits = get_canonical_train_val_test_splits(self.files, split_mode="cross_condition", seed=42)
        self.assertEqual((len(splits["train"]), len(splits["val"]), len(splits["test"])), (18, 6, 57))

    def test_no_overlap_and_full_coverage(self):
        for mode in SPLIT_MODES:
            with self.subTest(mode=mode):
                splits = get_canonical_train_val_test_splits(self.files, split_mode=mode, seed=42)
                train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
                self.assertEqual(len(train & val), 0)
                self.assertEqual(len(train & test), 0)
                self.assertEqual(len(val & test), 0)
                self.assertEqual(train | val | test, set(self.files))

    def test_cross_condition_val_is_stratified_two_per_specimen(self):
        splits = get_canonical_train_val_test_splits(self.files, split_mode="cross_condition", seed=42)
        val_specimens = [extract_file_metadata(f)["specimen"] for f in splits["val"]]
        for spec in SPECIMENS:
            self.assertEqual(val_specimens.count(spec), 2, f"{spec} contributes != 2 val files")
        train_specimens = [extract_file_metadata(f)["specimen"] for f in splits["train"]]
        for spec in SPECIMENS:
            self.assertEqual(train_specimens.count(spec), 6, f"{spec} contributes != 6 train files")

    def test_train_and_val_are_base_domain_and_test_is_shifted(self):
        splits = get_canonical_train_val_test_splits(self.files, split_mode="cross_condition", seed=42)
        for fp in splits["train"] + splits["val"]:
            self.assertTrue(is_base_domain_file(extract_file_metadata(fp), "cross_condition"), fp)
        for fp in splits["test"]:
            self.assertFalse(is_base_domain_file(extract_file_metadata(fp), "cross_condition"), fp)

    def test_determinism(self):
        for mode in SPLIT_MODES:
            with self.subTest(mode=mode):
                a = get_canonical_train_val_test_splits(self.files, split_mode=mode, seed=42)
                b = get_canonical_train_val_test_splits(list(reversed(self.files)), split_mode=mode, seed=42)
                self.assertEqual(a, b, "split is not stable under file-order changes")

    def test_different_seed_changes_val_but_not_sizes(self):
        a = get_canonical_train_val_test_splits(self.files, split_mode="cross_condition", seed=42)
        b = get_canonical_train_val_test_splits(self.files, split_mode="cross_condition", seed=7)
        self.assertEqual(len(a["val"]), len(b["val"]))
        self.assertNotEqual(a["val"], b["val"])

    def test_unknown_split_mode_raises(self):
        with self.assertRaises(ValueError):
            check_split_mode("cross_condition_v2")
        with self.assertRaises(ValueError):
            get_canonical_train_val_test_splits(self.files, split_mode="cross_condition_v2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
