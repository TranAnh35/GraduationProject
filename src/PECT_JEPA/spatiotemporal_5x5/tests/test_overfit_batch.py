"""
Single Batch Overfitting Sanity Test for 5x5 PECT-JEPA.
Verifies that the model, loss, and optimization loop can converge on a single fixed batch.
"""

import unittest
import torch
import torch.optim as optim

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5


class TestOverfitBatch(unittest.TestCase):

    def test_overfit_single_batch(self):
        torch.manual_seed(42)
        config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            min_masked=10,
            max_masked=12,
            var_weight=0.1,  # Keep regularization small for fast overfit test
            cov_weight=0.01,
            rank_barrier_weight=0.0,
        )
        model = PECT_JEPA_5x5(config)
        model.train()

        # Fixed synthetic batch
        B = 4
        x = torch.randn(B, 5, 5, 128)

        # Fixed mask for overfitting
        ctx_idx, tgt_idx, _ = model.masker.sample_mask(B, seed=42)

        optimizer = optim.AdamW(
            list(model.context_encoder.parameters()) +
            list(model.predictor.parameters()) +
            list(model.tokenizer.parameters()),
            lr=3e-3,
            weight_decay=1e-4
        )

        initial_loss = None
        final_loss = None
        fixed_freq = torch.ones(B, dtype=torch.long)

        for step in range(100):
            optimizer.zero_grad()
            out = model(
                x,
                custom_context_indices=ctx_idx,
                custom_target_indices=tgt_idx,
                freq_condition=fixed_freq
            )
            loss = out["loss"]

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Target encoder EMA update
            model.update_target_encoder(momentum=0.98)

            final_loss = loss.item()

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        # Verify loss decreased significantly (at least 50%)
        self.assertLess(
            final_loss,
            initial_loss * 0.50,
            f"Overfitting failed: initial loss {initial_loss:.4f} -> final loss {final_loss:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
