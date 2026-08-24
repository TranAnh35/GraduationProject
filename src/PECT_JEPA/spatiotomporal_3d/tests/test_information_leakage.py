"""
Information Leakage Test for PECT-JEPA v0.2 (implement.md, Section 4.3).
Verifies:
- Randomly altering raw pixel values in the Target region produces ZERO change in H_context and H_pred.
- Predictor depends ONLY on context representations and target positional queries (no content leakage).
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config


class TestInformationLeakage(unittest.TestCase):
    def test_predictor_and_context_no_leakage(self):
        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 64
        config.encoder.depth = 2
        config.predictor.depth = 2
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        # Create two inputs: clip1 and clip2
        B, H, W, T_c = 1, 32, 32, 16
        clip1 = torch.ones(B, H, W, T_c)
        clip2 = torch.ones(B, H, W, T_c)

        # Token grid: H_t = 4, W_t = 4, T_c = 16 -> total 256 tokens
        # In [H_t, W_t, T_c] layout, flat index is h*(W_t*T_c) + w*T_c + t
        # For patch (h=0, w=0) at all 16 frames (t=0..15), flat indices are 0..15
        tgt_indices = torch.tensor([[t for t in range(16)]], dtype=torch.long)
        ctx_indices = torch.tensor([[i for i in range(256) if i not in tgt_indices[0].tolist()]], dtype=torch.long)

        # Alter raw pixels ONLY in the target region (spatial patch h=0, w=0 -> pixels 0..7, 0..7) in clip2
        clip2[:, :8, :8, :] += 50.0

        # Forward pass on both with the same fixed mask
        out1 = model(clip1, custom_context_indices=ctx_indices, custom_target_indices=tgt_indices)
        out2 = model(clip2, custom_context_indices=ctx_indices, custom_target_indices=tgt_indices)

        # 1. Context representations must be identical (Context is unaffected by target changes)
        ctx_diff = torch.sum(torch.abs(out1["H_context"] - out2["H_context"])).item()
        self.assertAlmostEqual(ctx_diff, 0.0, places=5, msg="H_context changed when target region was altered!")

        # 2. Predicted target representations must be 100% IDENTICAL (No Target Content Leakage to Predictor!)
        pred_diff = torch.sum(torch.abs(out1["H_pred"] - out2["H_pred"])).item()
        self.assertAlmostEqual(pred_diff, 0.0, places=5, msg="H_pred leaked target content! Must depend ONLY on context!")

        # 3. Meanwhile, Target Encoder output MUST differ because it encodes the actual target content
        tgt_diff = torch.sum(torch.abs(out1["H_target"] - out2["H_target"])).item()
        self.assertGreater(tgt_diff, 0.1, "Target Encoder failed to perceive the altered target content!")

        print(f"PASS: test_information_leakage.py | H_ctx diff: {ctx_diff:.7f} | H_pred diff: {pred_diff:.7f} (No Leakage!) | H_target diff: {tgt_diff:.4f}")


if __name__ == "__main__":
    unittest.main()
