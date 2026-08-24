"""
Information Leakage Test for 1D Temporal PECT-JEPA (implement.md, Section 5.3).
Verifies:
- Altering values in the Target region of the waveform produces ZERO change in H_context and H_pred.
- Predictor depends ONLY on context representations and target positional queries (Zero Leakage).
"""

import torch
import unittest
from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import get_default_config_1d


class TestLeakage1D(unittest.TestCase):
    def test_zero_leakage_1d(self):
        config = get_default_config_1d()
        config.patch_length = 32
        config.stride = 31
        config.embed_dim = 64
        config.encoder_depth = 2
        config.predictor_depth = 2
        config.device = "cpu"

        model = PECT_JEPA_1D(config)
        model.eval()

        # Create two 1D waveforms
        B, T = 1, 500
        x1 = torch.ones(B, T)
        x2 = torch.ones(B, T)

        # In late_decay, target is patches 5..15 (points from ~155 to 500)
        # Context is patches 0..4 (points 0 to ~155)
        # Modify points in the target region only (e.g. points 200..500) in x2
        x2[:, 200:500] += 50.0

        ctx_indices = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        tgt_indices = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.long)

        out1 = model(x1, custom_context_indices=ctx_indices, custom_target_indices=tgt_indices)
        out2 = model(x2, custom_context_indices=ctx_indices, custom_target_indices=tgt_indices)

        # 1. H_context must be 100% identical
        ctx_diff = torch.sum(torch.abs(out1["H_context"] - out2["H_context"])).item()
        self.assertAlmostEqual(ctx_diff, 0.0, places=5, msg="H_context changed when target region was altered!")

        # 2. H_pred must be 100% identical (No Target Content Leakage to Predictor!)
        pred_diff = torch.sum(torch.abs(out1["H_pred"] - out2["H_pred"])).item()
        self.assertAlmostEqual(pred_diff, 0.0, places=5, msg="H_pred leaked target content! Must depend ONLY on context!")

        # 3. H_target must differ because it encodes the altered target content
        tgt_diff = torch.sum(torch.abs(out1["H_target"] - out2["H_target"])).item()
        self.assertGreater(tgt_diff, 0.1, "Target Encoder failed to perceive the altered target content!")

        print(f"PASS: test_leakage_1d.py | H_ctx diff: {ctx_diff:.7f} | H_pred diff: {pred_diff:.7f} (No Leakage!) | H_target diff: {tgt_diff:.4f}")


if __name__ == "__main__":
    unittest.main()
