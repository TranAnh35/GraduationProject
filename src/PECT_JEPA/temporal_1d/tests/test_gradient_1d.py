"""
Gradient & EMA Verification Tests for 1D Temporal PECT-JEPA (implement.md, Section 5.4).
Verifies:
1. Context Encoder, Predictor, and Tokenizer 1D receive finite gradients.
2. Target Encoder receives NO gradients (requires_grad = False, grad is None).
3. Target Encoder updates exclusively via EMA.
"""

import torch
import unittest
from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import get_default_config_1d


class TestGradient1D(unittest.TestCase):
    def test_gradient_flow_and_ema_1d(self):
        config = get_default_config_1d()
        config.patch_length = 32
        config.stride = 31
        config.embed_dim = 64
        config.encoder_depth = 2
        config.predictor_depth = 2
        config.device = "cpu"

        model = PECT_JEPA_1D(config)
        model.train()

        x = torch.randn(4, 500, requires_grad=True)

        out = model(x)
        loss = out["loss"]
        loss.backward()

        # 1. Check Context Encoder receives gradients
        has_ctx_grad = False
        for name, param in model.context_encoder.named_parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)).item() > 0:
                has_ctx_grad = True
                break
        self.assertTrue(has_ctx_grad, "Context Encoder 1D did not receive gradients!")

        # 2. Check Predictor receives gradients
        has_pred_grad = False
        for name, param in model.predictor.named_parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)).item() > 0:
                has_pred_grad = True
                break
        self.assertTrue(has_pred_grad, "Predictor 1D did not receive gradients!")

        # 3. Check Tokenizer receives gradients
        has_tok_grad = False
        for name, param in model.tokenizer.named_parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)).item() > 0:
                has_tok_grad = True
                break
        self.assertTrue(has_tok_grad, "Tokenizer 1D did not receive gradients!")

        # 4. Check Target Encoder has NO gradients
        for name, param in model.target_encoder.named_parameters():
            self.assertFalse(param.requires_grad, f"Target Encoder param {name} requires_grad is True!")
            self.assertIsNone(param.grad, f"Target Encoder param {name} received grad!")

        # 5. Check EMA update alters target parameters
        initial_tgt_params = [p.clone() for p in model.target_encoder.parameters()]
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
        self.assertTrue(param_changed, "Target Encoder 1D parameters did not update via EMA!")

        print("PASS: test_gradient_1d.py | Context, Predictor, Tokenizer receive grad | Target Encoder isolated & updated via EMA")


if __name__ == "__main__":
    unittest.main()
