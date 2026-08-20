"""
Information Leakage Test (Section 22.5).
Explicitly verifies that the Predictor never receives target token content/values,
and only receives visible context latents and target positional embeddings.
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config


class TestInformationLeakage(unittest.TestCase):
    def test_predictor_inputs(self):
        config = get_default_config()
        config.temporal_encoder.t_prime = 64
        config.temporal_encoder.raw_samples = 500
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 64
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.eval()

        # Create two inputs where target region has different values, but context region is identical
        B, H, W, L = 1, 32, 32, 500
        x1 = torch.ones(B, H, W, L)
        x2 = torch.ones(B, H, W, L)

        # Perturb a specific region (say patch 0..8, 0..8)
        x2[:, :8, :8, :] += 5.0

        # Define fixed mask where target is exactly the perturbed patch
        # Patch grid is 4x4x7. Target = patch (0, 0, :) -> indices [0, 1, 2, 3, 4, 5, 6]
        tgt_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
        ctx_indices = torch.tensor([[i for i in range(7, 4 * 4 * 7)]], dtype=torch.long)

        # Forward pass on both with fixed custom context/target indices
        out1 = model(x1, custom_context_indices=ctx_indices, custom_target_indices=tgt_indices)
        out2 = model(x2, custom_context_indices=ctx_indices, custom_target_indices=tgt_indices)

        # Context latents should be identical because context input region is identical
        ctx_diff = torch.sum(torch.abs(out1["H_context"] - out2["H_context"])).item()
        self.assertAlmostEqual(ctx_diff, 0.0, places=4, msg="Context representations differ for identical context input!")

        # Predicted target latents MUST be identical because Predictor ONLY sees H_context and target_pos (not target content!)
        pred_diff = torch.sum(torch.abs(out1["H_pred"] - out2["H_pred"])).item()
        self.assertAlmostEqual(pred_diff, 0.0, places=4, msg="Predictor leaked target content! Predictions must depend ONLY on context!")

        # Meanwhile, target encoder latents MUST differ because target encoder receives the actual target tokens
        tgt_encoder_diff = torch.sum(torch.abs(out1["H_target"] - out2["H_target"])).item()
        self.assertGreater(tgt_encoder_diff, 0.1, "Target encoder failed to distinguish different target content!")

        print(f"PASS: No-Information-Leakage Test | Pred diff: {pred_diff:.6f} (isolated) | Target Enc diff: {tgt_encoder_diff:.4f}")


if __name__ == "__main__":
    unittest.main()
