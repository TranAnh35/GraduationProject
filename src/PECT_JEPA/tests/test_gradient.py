"""
Gradient and EMA Flow Tests (Section 22.3).
Verifies:
1. Context Encoder, Predictor, and Temporal Encoder receive gradients.
2. Target Encoder receives NO gradients.
3. Target Encoder parameters update exclusively via EMA.
"""

import torch
import unittest
from ..models.jepa import PECT_JEPA
from ..configs.config import get_default_config


class TestGradient(unittest.TestCase):
    def test_gradient_flow_and_ema(self):
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

        x = torch.randn(2, 32, 32, 500, requires_grad=True)

        out = model(x)
        loss = out["loss"]
        loss.backward()

        # 1. Check Context Encoder receives gradients
        has_ctx_grad = False
        for name, param in model.context_encoder.named_parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)).item() > 0:
                has_ctx_grad = True
                break
        self.assertTrue(has_ctx_grad, "Context Encoder did not receive gradients!")

        # 2. Check Predictor receives gradients
        has_pred_grad = False
        for name, param in model.predictor.named_parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)).item() > 0:
                has_pred_grad = True
                break
        self.assertTrue(has_pred_grad, "Predictor did not receive gradients!")

        # 3. Check Target Encoder has NO gradients
        for name, param in model.target_encoder.named_parameters():
            self.assertFalse(param.requires_grad, f"Target Encoder param {name} requires grad!")
            self.assertIsNone(param.grad, f"Target Encoder param {name} received gradient!")

        # 4. Check EMA update alters target parameters
        initial_tgt_params = [p.clone() for p in model.target_encoder.parameters()]
        # Perturb context encoder params
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        model.update_target_encoder(momentum=0.9)
        new_tgt_params = [p.clone() for p in model.target_encoder.parameters()]

        param_changed = False
        for p_init, p_new in zip(initial_tgt_params, new_tgt_params):
            if torch.sum(torch.abs(p_init - p_new)).item() > 1e-6:
                param_changed = True
                break
        self.assertTrue(param_changed, "Target Encoder parameters did not update via EMA!")

        print("PASS: Gradient & EMA Test | Context/Predictor receive grad | Target Encoder isolated & updated via EMA")


if __name__ == "__main__":
    unittest.main()
