"""
Real TDMS Dataset and Model Integration Test.
"""

import os
import sys
import unittest
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.data.dataset import PECTDataset, find_all_tdms_files, collate_pect_batch
from src.PECT_JEPA.models.jepa import PECT_JEPA
from src.PECT_JEPA.configs.config import get_default_config


class TestRealTDMS(unittest.TestCase):
    def test_real_tdms_loading_and_forward(self):
        files = find_all_tdms_files("data")
        if len(files) == 0:
            print("Skipping real TDMS test: no data found in data/")
            return

        print(f"Testing with real TDMS file: {files[0]}")
        config = get_default_config()
        config.data.spatial_size = (300, 300)
        config.temporal_encoder.t_prime = 64
        config.tokenizer.spatial_patch = 8  # Standard Ps=8 -> 37x37 tokens
        config.encoder.attention_type = "factorized"
        config.training.device = "cpu"

        ds = PECTDataset(
            file_paths=[files[0]],
            time_samples=500,
            spatial_size=(300, 300),
            normalization="min_max"
        )
        sample, meta = ds[0]
        self.assertEqual(sample.shape, (300, 300, 500), f"Unexpected sample shape: {sample.shape}")
        print(f"Loaded TDMS successfully. Shape: {sample.shape}, Metadata: {meta['sensor']}, {meta['waveform']}, {meta['liftoff']}")

        model = PECT_JEPA(config)
        model.eval()

        batch = collate_pect_batch([(sample, meta)])
        with torch.no_grad():
            out = model(batch["data"])
            feats = model.extract_features(batch["data"], pool_temporal=False)

        self.assertFalse(torch.isnan(out["loss"]).item(), "Loss is NaN")
        H_t, W_t = 300 // 8, 300 // 8
        self.assertEqual(feats.shape, (1, H_t, W_t, 7, config.tokenizer.embed_dim), f"Unexpected feature map shape: {feats.shape}")
        print(f"PASS: Real TDMS Forward Pass & Full 4D Feature Extraction | Loss: {out['loss'].item():.4f} | Feature Grid: {feats.shape}")


if __name__ == "__main__":
    unittest.main()
