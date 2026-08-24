"""
Real TDMS Dataset and Model Integration Test for PECT-JEPA v0.3.
"""

import os
import sys
import unittest
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.spatiotomporal_3d.data.dataset import PECTClipDataset, find_all_tdms_files, collate_pect_batch
from src.PECT_JEPA.spatiotomporal_3d.models.jepa import PECT_JEPA
from src.PECT_JEPA.spatiotomporal_3d.configs.config import get_default_config


class TestRealTDMS(unittest.TestCase):
    def test_real_tdms_loading_and_forward(self):
        files = find_all_tdms_files("data")
        if len(files) == 0:
            print("Skipping real TDMS test: no data found in data/")
            return

        print(f"Testing with real TDMS file: {files[0]}")
        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.mask.spatial_block_h = 4
        config.mask.spatial_block_w = 4
        config.mask.num_masked_frames = 8
        config.training.device = "cpu"

        ds = PECTClipDataset(
            file_paths=[files[0]],
            clip_length=16,
            clip_stride=8,
            spatial_crop_size=(64, 64),
            train=True,
            normalization="min_max",
            use_memmap=False
        )
        sample, meta = ds[0]
        self.assertEqual(sample.shape, (64, 64, 16), f"Unexpected sample shape: {sample.shape}")
        print(f"Loaded TDMS clip successfully. Shape: {sample.shape}, Metadata: {meta.get('sensor')}, {meta.get('waveform')}, {meta.get('liftoff')}")

        model = PECT_JEPA(config)
        model.eval()

        batch = collate_pect_batch([(sample, meta)])
        with torch.no_grad():
            out = model(batch["data"])
            feats = model.extract_features(batch["data"], pool_temporal=False)

        self.assertFalse(torch.isnan(out["loss"]).item(), "Loss is NaN")
        self.assertEqual(feats.shape, (1, 8, 8, 16, config.tokenizer.embed_dim), f"Unexpected feature map shape: {feats.shape}")
        print(f"PASS: Real TDMS Forward Pass & Full Feature Extraction | Loss: {out['loss'].item():.4f} | Feature Grid: {feats.shape}")


if __name__ == "__main__":
    unittest.main()
