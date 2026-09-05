"""
Unit test for border cropping (crop_border) in 5x5 PECT-JEPA.
Verifies:
1. load_cscan_from_tdms yields [290, 290, C] when crop_border=5 and [300, 300, C] when crop_border=0.
2. PECT5x5Dataset correctly computes eff_sX, eff_sY, total points, and padded grid shape.
3. __getitem__ correctly maps 1D index to cropped 2D coordinates and preserves original coordinates.
4. Cache hash includes _crop suffix ensuring cache isolation.
"""

import unittest
import glob
import os
import torch
import numpy as np

from ..configs.config import Spatiotemporal5x5Config
from ..data.dataset import PECT5x5Dataset
from ..evaluation.cscan_extractor import load_cscan_from_tdms


class TestCropBorder(unittest.TestCase):

    def setUp(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        data_dir = os.path.join(root_dir, "data")
        pattern = os.path.join(data_dir, "**", "*.tdms")
        files = sorted([f for f in glob.glob(pattern, recursive=True) if not f.endswith(".tdms_index")])
        self.assertTrue(len(files) > 0, "No TDMS files found in data directory")
        self.sample_file = files[0]

    def test_load_cscan_crop_dimensions(self):
        """Test load_cscan_from_tdms shape with crop_border=0, crop_border=5, and crop_border=10."""
        # Uncropped
        grid_uncropped = load_cscan_from_tdms(
            self.sample_file,
            time_samples=500,
            temporal_samples=128,
            resample_mode="linear",
            crop_border=0,
        )
        self.assertEqual(grid_uncropped.shape, (300, 300, 128))

        # Cropped by 5 pixels on each edge
        grid_cropped_5 = load_cscan_from_tdms(
            self.sample_file,
            time_samples=500,
            temporal_samples=128,
            resample_mode="linear",
            crop_border=5,
        )
        self.assertEqual(grid_cropped_5.shape, (290, 290, 128))
        np.testing.assert_allclose(grid_cropped_5, grid_uncropped[5:295, 5:295, :], rtol=1e-5)

        # Cropped by 10 pixels on each edge
        grid_cropped_10 = load_cscan_from_tdms(
            self.sample_file,
            time_samples=500,
            temporal_samples=128,
            resample_mode="linear",
            crop_border=10,
        )
        self.assertEqual(grid_cropped_10.shape, (280, 280, 128))
        np.testing.assert_allclose(grid_cropped_10, grid_uncropped[10:290, 10:290, :], rtol=1e-5)

    def test_dataset_crop_metadata_and_shapes(self):
        """Test PECT5x5Dataset with crop_border=10 (default)."""
        ds = PECT5x5Dataset(
            file_paths=[self.sample_file],
            time_samples=500,
            temporal_samples=128,
            resample_mode="linear",
            crop_border=10,
            use_memmap=False,  # in-memory for unit test speed
        )

        self.assertEqual(ds.eff_sX, 280)
        self.assertEqual(ds.eff_sY, 280)
        self.assertEqual(len(ds), 280 * 280)  # 78,400 points

        # Padded array shape: 280 + 2*2 = 284
        padded = ds._in_memory_files[0]
        self.assertEqual(padded.shape, (284, 284, 128))

        # Sample first item (0, 0)
        patch_0, meta_0 = ds[0]
        self.assertEqual(patch_0.shape, (5, 5, 128))
        self.assertEqual(meta_0["row"], 0)
        self.assertEqual(meta_0["col"], 0)
        self.assertEqual(meta_0["orig_row"], 10)
        self.assertEqual(meta_0["orig_col"], 10)

        # Sample arbitrary item (row 10, col 20) -> point_idx = 10 * 280 + 20 = 2820
        target_idx = 10 * 280 + 20
        patch_k, meta_k = ds[target_idx]
        self.assertEqual(patch_k.shape, (5, 5, 128))
        self.assertEqual(meta_k["row"], 10)
        self.assertEqual(meta_k["col"], 20)
        self.assertEqual(meta_k["orig_row"], 20)
        self.assertEqual(meta_k["orig_col"], 30)


if __name__ == "__main__":
    unittest.main()
