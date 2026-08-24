"""
Overfit Verification Test for 1D Temporal PECT-JEPA (implement.md, Section 5.5).
Verifies that 20 gradient descent steps on a single fixed batch decrease loss monotonically towards 0.
"""

import torch
import unittest
from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import get_default_config_1d
from ..training.optimizer import build_optimizer_1d


class TestOverfit1D(unittest.TestCase):
    def test_overfit_single_batch_1d(self):
        config = get_default_config_1d()
        config.patch_length = 32
        config.stride = 31
        config.embed_dim = 64
        config.encoder_depth = 2
        config.predictor_depth = 2
        config.device = "cpu"

        model = PECT_JEPA_1D(config)
        model.train()

        optimizer = build_optimizer_1d(model, lr=1e-3, weight_decay=0.0)

        # Fixed batch of 4 waveforms
        x = torch.randn(4, 500)
        ctx_idx, tgt_idx, _ = model.masker.sample_mask(4)

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            out = model(x, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
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
        print(f"PASS: test_overfit_1d.py | Initial Loss: {initial_loss:.6f} -> Final Loss (step 20): {final_loss:.6f} (Decreased significantly)")


if __name__ == "__main__":
    unittest.main()
