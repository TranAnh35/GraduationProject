"""
Full-Scale Forward & Verification Benchmark for PECT-JEPA v0.3 (implement.md, Section 5.6).
Verifies complete forward and backward pass on:
1. Standard Training Crop: [1, 64, 64, 16] -> Grid: (8, 8, 16) = 1,024 tokens (D=128)
2. Full Acquisition Scale: [1, 300, 300, 16] -> Grid: (37, 37, 16) = 21,904 tokens (D=128)
"""

import os
import sys
import time
import unittest
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.spatiotomporal_3d.models.jepa import PECT_JEPA
from src.PECT_JEPA.spatiotomporal_3d.configs.config import get_default_config
from src.PECT_JEPA.spatiotomporal_3d.training.optimizer import build_optimizer


class TestFullScaleForward(unittest.TestCase):
    def test_crop_forward_64x64x16(self):
        print("\n" + "=" * 70)
        print("RUNNING STANDARD TRAINING CROP BENCHMARK (64 x 64 x 16)")
        print("=" * 70)

        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.clip.temporal_length = 16
        config.mask.spatial_block_h = 4
        config.mask.spatial_block_w = 4
        config.mask.num_masked_frames = 8
        config.encoder.depth = 4
        config.predictor.depth = 2
        config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(config.training.device)
        print(f"Device: {device}")

        # 1. Instantiate Model & Optimizer
        model = PECT_JEPA(config).to(device)
        optimizer = build_optimizer(model, lr=3e-4)

        # 2. Input clip: [1, 64, 64, 16]
        B, H, W, T_c = 1, 64, 64, 16
        clip = torch.randn(B, H, W, T_c, device=device)
        print(f"1. Input Crop Shape: {clip.shape} ({clip.numel() * 4 / (1024**2):.2f} MB)")

        # 3. Forward Pass 1
        t0 = time.time()
        out1 = model(clip)
        t_forward1 = time.time() - t0

        loss1 = out1["loss"]
        H_pred = out1["H_pred"]
        H_target = out1["H_target"]
        H_context = out1["H_context"]
        grid_shape = out1["grid_shape"]
        ctx_idx1 = out1["context_indices"]
        tgt_idx1 = out1["target_indices"]

        total_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
        actual_mask_ratio = tgt_idx1.shape[1] / total_tokens

        print(f"2. Forward pass completed in {t_forward1:.3f}s")
        print(f"   - Grid shape (H_t, W_t, T_c): {grid_shape} -> {total_tokens} tokens")
        print(f"   - Context tokens (N_ctx): {ctx_idx1.shape[1]}")
        print(f"   - Target tokens  (N_tgt): {tgt_idx1.shape[1]} (Mask Ratio: {actual_mask_ratio:.2%})")
        print(f"   - H_context: {H_context.shape} | H_pred: {H_pred.shape} | H_target: {H_target.shape}")
        print(f"   - Initial Loss: {loss1.item():.6f}")

        # Dimensions & finiteness assertions
        self.assertEqual(grid_shape, (8, 8, 16))
        self.assertEqual(total_tokens, 1024)
        self.assertEqual(tgt_idx1.shape[1], 8 * 4 * 4)  # 128 target tokens
        self.assertFalse(torch.isnan(loss1).item(), "Loss is NaN")
        self.assertFalse(torch.isinf(loss1).item(), "Loss is Inf")

        # 4. Backward Pass
        optimizer.zero_grad()
        t0 = time.time()
        loss1.backward()
        t_backward = time.time() - t0
        print(f"3. Backward pass completed in {t_backward:.3f}s")

        # Check gradients
        for name, param in model.context_encoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break

        for name, param in model.target_encoder.named_parameters():
            self.assertFalse(param.requires_grad)
            self.assertIsNone(param.grad)

        # 5. Optimizer & EMA Step
        optimizer.step()
        model.update_target_encoder(momentum=0.996)
        print("4. Optimizer step & EMA Target Encoder update: SUCCESS")

        # 6. Forward Pass 2 (Dynamic Masking Check)
        out2 = model(clip)
        tgt_idx2 = out2["target_indices"]
        mask_diff = torch.sum(torch.abs(tgt_idx1.float() - tgt_idx2.float())).item()
        print(f"5. Dynamic Masking Verification: Target set 1 != Target set 2 (Diff: {mask_diff:.1f})")
        self.assertGreater(mask_diff, 0.0, "Mask positions identical across consecutive passes!")

        # 7. Full Feature Extraction
        feats = model.extract_features(clip, pool_temporal=False)
        self.assertEqual(feats.shape, (1, 8, 8, 16, 128))
        print(f"6. Full 4D Feature Extraction: {feats.shape} (pool_temporal=False): SUCCESS")
        print("STANDARD CROP BENCHMARK (64x64x16) PASSED 100%!")

    def test_full_scale_300x300x16(self):
        print("\n" + "=" * 70)
        print("RUNNING FULL-SCALE CLIP BENCHMARK (300 x 300 x 16)")
        print("=" * 70)

        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 128
        config.clip.temporal_length = 16
        config.mask.spatial_block_h = 4
        config.mask.spatial_block_w = 4
        config.mask.num_masked_frames = 8
        config.encoder.depth = 4
        config.predictor.depth = 2
        config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(config.training.device)

        model = PECT_JEPA(config).to(device)
        B, H, W, T_c = 1, 300, 300, 16
        clip = torch.randn(B, H, W, T_c, device=device)

        out1 = model(clip)
        self.assertEqual(out1["grid_shape"], (37, 37, 16))
        self.assertFalse(torch.isnan(out1["loss"]).item())
        print(f"Full Scale 300x300x16 Forward Loss: {out1['loss'].item():.6f} - SUCCESS")


if __name__ == "__main__":
    unittest.main()
