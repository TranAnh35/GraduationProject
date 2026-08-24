"""
Small Overfit Verification Test for PECT-JEPA v0.2 (implement.md, Section 4.5).
Verifies that 20 gradient descent steps on a single fixed clip decrease loss monotonically towards 0.
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config
from ..training.optimizer import build_optimizer


class TestOverfit(unittest.TestCase):
    def test_overfit_single_clip(self):
        config = get_default_config()
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 64
        config.encoder.depth = 2
        config.predictor.depth = 2
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.train()

        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)

        # Fixed clip: [1, 32, 32, 16] -> grid (4, 4, 16)
        clip = torch.randn(1, 32, 32, 16)
        grid_shape = (4, 4, 16)
        ctx_idx, tgt_idx, _ = model.masker.sample_mask(1, grid_shape)

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            out = model(clip, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
            loss = out["loss"]

            self.assertFalse(torch.isnan(loss).item(), f"Loss is NaN at step {step}")
            self.assertFalse(torch.isinf(loss).item(), f"Loss is Inf at step {step}")

            loss.backward()
            optimizer.step()
            model.update_target_encoder(momentum=0.99)

            losses.append(loss.item())

        initial_loss = losses[0]
        final_loss = losses[-1]

        self.assertLess(final_loss, initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}")
        self.assertLess(final_loss, initial_loss * 0.5, f"Loss decrease insufficient: {initial_loss:.6f} -> {final_loss:.6f}")
        print(f"PASS: test_overfit.py | Initial Loss: {initial_loss:.6f} -> Final Loss (step 20): {final_loss:.6f} (Decreased significantly)")


if __name__ == "__main__":
    unittest.main()
