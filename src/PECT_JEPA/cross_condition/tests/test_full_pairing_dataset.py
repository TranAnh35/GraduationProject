"""
Unit tests for Full 81-File Cross-Condition Dataset with grid_stride = 4.
Validates:
1. Directed pairs count across all 3 specimens (Corrosion, Rivet_v1, Rivet_v2): 702 * 3 = 2106 pairs.
2. Coordinate grid calculation with grid_stride = 4: 68 * 68 = 4624 coords.
3. Total exhaustive sample space: 2106 * 4624 = 9,738,144 patch pairs.
4. Partitioning sub-epoch slicing and sample retrieval (__getitem__).
"""

import os
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.data.paired_dataset import PairedPECTDataset
from src.PECT_JEPA.cross_condition.data.split import extract_file_metadata


class TestFullPairingDataset(unittest.TestCase):
    def test_mock_full_dataset_geometry(self):
        """Test dataset pairing and coordinate arithmetic with mock scan metadata."""
        # Create mock file metadata for 81 files (27 per specimen)
        mock_files = []
        for spec in ["Corrosion", "Rivet_v1", "Rivet_v2"]:
            for sensor in ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]:
                for wf in ["Square", "Gaussian", "Chirp"]:
                    for z in ["z1", "z2", "z3"]:
                        mock_files.append(f"data/{spec}/{sensor}_{wf}_{z}.tdms")

        self.assertEqual(len(mock_files), 81)

        # Mock preloaded entries in PairedPECTDataset without reading disk TDMS
        ds = object.__new__(PairedPECTDataset)
        ds.num_partitions = 10
        ds.seed = 42
        ds.use_full_dataset = True
        ds.split_mode = "cross_condition"
        ds.grid_size = 5
        ds.temporal_samples = 128
        ds.grid_stride = 4
        ds.offsets = np.zeros((25, 2), dtype=np.int32)
        ds.pad = 7

        # Group mock scans by specimen
        ds.scans_by_specimen = {"Corrosion": [], "Rivet_v1": [], "Rivet_v2": []}
        for f in mock_files:
            meta = extract_file_metadata(f)
            spec = meta["specimen"]
            ds.scans_by_specimen[spec].append({
                "file_path": f,
                "meta": meta,
                "sY": 270,
                "sX": 270,
                "C": 128,
                "padded": np.zeros((270 + 14, 270 + 14, 128), dtype=np.float32),
                "ref_patch": np.zeros((5, 5, 128), dtype=np.float32),
            })

        # Build directed pairs
        ds.directed_pairs = []
        for spec, scan_list in ds.scans_by_specimen.items():
            n = len(scan_list)
            self.assertEqual(n, 27)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ds.directed_pairs.append((scan_list[i], scan_list[j]))

        # 27 * 26 = 702 pairs per specimen * 3 specimens = 2106 pairs
        self.assertEqual(len(ds.directed_pairs), 2106)

        # Build coordinates with stride = 4
        coord_list = []
        for r in range(0, 270, ds.grid_stride):
            for c in range(0, 270, ds.grid_stride):
                coord_list.append((r, c))
        ds.coords = np.array(coord_list, dtype=np.int32)
        ds.coords_per_pair = len(ds.coords)

        # (ceil(270/4) = 68) -> 68 * 68 = 4624
        self.assertEqual(ds.coords_per_pair, 4624)

        ds.total_exhaustive_samples = len(ds.directed_pairs) * ds.coords_per_pair
        self.assertEqual(ds.total_exhaustive_samples, 2106 * 4624)
        self.assertEqual(ds.total_exhaustive_samples, 9738144)

        # Macro-cycle partitioning
        ds.samples_per_epoch = ds.total_exhaustive_samples // ds.num_partitions
        ds.use_partitioning = True
        self.assertEqual(ds.samples_per_epoch, 973814)

        ds.current_cycle_idx = -1
        ds.current_cycle_perm = None
        ds.active_indices = None
        ds.set_epoch(1)

        # Length of partition 1
        self.assertEqual(len(ds), 973814)

        # Retrieve one sample via __getitem__
        sample = ds[0]
        self.assertIn("x_a", sample)
        self.assertIn("x_b", sample)
        self.assertIn("ref_a", sample)
        self.assertIn("ref_b", sample)
        self.assertEqual(sample["x_a"].shape, (5, 5, 128))
        self.assertEqual(sample["x_b"].shape, (5, 5, 128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
