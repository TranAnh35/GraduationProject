"""
Small Overfit and Optimization Test (Section 22.4).
Verifies that optimization step decreases loss on a fixed batch,
predictions remain finite, and EMA updates without NaN/Inf.
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config
from ..training.optimizer import build_optimizer


class TestOverfit(unittest.TestCase):
    def test_overfit_single_batch(self):
        config = get_default_config()
        config.temporal_encoder.t_prime = 64
        config.temporal_encoder.raw_samples = 500
        config.tokenizer.spatial_patch = 8
        config.tokenizer.embed_dim = 64
        config.encoder.depth = 2
        config.predictor.depth = 2
        config.training.device = "cpu"

        model = PECT_JEPA(config)
        model.train()

        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)

        # Fixed sample and fixed mask
        x = torch.randn(1, 32, 32, 500)
        grid_shape = (4, 4, 7)
        ctx_idx, tgt_idx, _ = model.masker.sample_mask(1, grid_shape)

        initial_loss = None
        final_loss = None

        losses = []
        for step in range(15):
            optimizer.zero_grad()
            out = model(x, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
            loss = out["loss"]

            self.assertFalse(torch.isnan(loss).item(), f"Loss is NaN at step {step}")
            self.assertFalse(torch.isinf(loss).item(), f"Loss is Inf at step {step}")

            loss.backward()
            optimizer.step()
            model.update_target_encoder(momentum=0.99)

            loss_val = loss.item()
            losses.append(loss_val)

        initial_loss = losses[0]
        final_loss = losses[-1]

        self.assertLess(final_loss, initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}")
        print(f"PASS: Overfit Test | Initial Loss: {initial_loss:.6f} -> Final Loss: {final_loss:.6f} (Decreased)")


if __name__ == "__main__":
    unittest.main()
