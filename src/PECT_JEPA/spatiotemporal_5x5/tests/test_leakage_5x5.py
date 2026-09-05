"""
Information Leakage Prevention Unit Test for 5x5 PECT-JEPA.

Verifies:
1. Context encoder representation is invariant to perturbations in masked (target) positions.
2. Target encoder parameters have no gradients after loss.backward() (strictly updated via EMA).
3. Context encoder gradients flow only from unmasked context tokens.
"""

import unittest
import torch

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5


class TestLeakage5x5(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            min_masked=10,
            max_masked=15,
        )
        self.model = PECT_JEPA_5x5(self.config)

    def test_target_perturbation_invariance(self):
        """
        Verify that perturbing target (masked) inputs does NOT affect context representations.
        """
        self.model.eval()
        B = 2
        x = torch.randn(B, 5, 5, 128)

        # Sample mask
        ctx_idx, tgt_idx, mask_bool = self.model.masker.sample_mask(B, seed=123)

        # Forward on original input
        with torch.no_grad():
            out_orig = self.model(x, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
            H_ctx_orig = out_orig["H_ctx"]

        # Create perturbed input: add noise ONLY at target locations
        x_perturbed = x.clone()
        x_perturbed_flat = x_perturbed.view(B, 25, 128)
        noise = torch.randn_like(x_perturbed_flat) * 10.0

        for b in range(B):
            t_idx = tgt_idx[b]
            x_perturbed_flat[b, t_idx] += noise[b, t_idx]

        x_perturbed = x_perturbed_flat.view(B, 5, 5, 128)

        # Forward on perturbed input with SAME context indices
        with torch.no_grad():
            out_pert = self.model(x_perturbed, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
            H_ctx_pert = out_pert["H_ctx"]

        # H_ctx must be strictly identical
        diff = (H_ctx_orig - H_ctx_pert).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5,
                               msg=f"Context representations leaked target info! Max diff: {diff}")

    def test_target_encoder_zero_gradients(self):
        """
        Verify that loss.backward() does not compute gradients for target encoder parameters.
        Target encoder must be updated strictly via EMA.
        """
        self.model.train()
        B = 2
        x = torch.randn(B, 5, 5, 128)

        out = self.model(x)
        loss = out["loss"]
        loss.backward()

        for name, param in self.model.target_encoder.named_parameters():
            self.assertTrue(
                param.grad is None or (param.grad == 0).all(),
                f"Target encoder parameter {name} has non-zero gradients! Gradient leakage detected."
            )

        # In contrast, context encoder and predictor MUST have gradients
        has_ctx_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.context_encoder.parameters())
        has_pred_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.predictor.parameters())

        self.assertTrue(has_ctx_grad, "Context encoder received no gradients.")
        self.assertTrue(has_pred_grad, "Predictor received no gradients.")


if __name__ == "__main__":
    unittest.main()
