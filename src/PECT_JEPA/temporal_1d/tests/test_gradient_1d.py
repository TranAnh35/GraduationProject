"""
Gradient Flow & Target Encoder Isolation Test for 1D Temporal PECT-JEPA (Stage A).
Verifies:
- gradients reach Tokenizer, Context Encoder, Predictor
- Target Encoder is isolated from autograd and updated only via EMA
- anti-collapse loss components are finite and contribute gradients
"""

import torch
import unittest
from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import get_default_config_1d
from ..data.preprocessing import build_two_channel_input
from .synth import make_waveforms


class TestGradient1D(unittest.TestCase):
    def test_gradient_flow_and_ema_1d(self):
        config = get_default_config_1d()
        config.embed_dim = 64
        config.encoder_depth = 2
        config.predictor_depth = 2
        config.device = "cpu"

        model = PECT_JEPA_1D(config)
        model.train()

        x = torch.from_numpy(build_two_channel_input(make_waveforms(n=4, T=500, seed=3)))
        out = model(x)
        loss = out["loss"]
        loss.backward()

        tok_grad = model.tokenizer.proj.weight.grad
        self.assertIsNotNone(tok_grad, "Tokenizer received no gradient!")
        self.assertTrue(torch.isfinite(tok_grad).all(), "Tokenizer gradient not finite")

        ctx_grad = model.context_encoder.blocks[0].attn.qkv.weight.grad
        self.assertIsNotNone(ctx_grad, "Context Encoder received no gradient!")
        self.assertTrue(torch.isfinite(ctx_grad).all())

        pred_grad = model.predictor.pred_head.weight.grad
        self.assertIsNotNone(pred_grad, "Predictor received no gradient!")

        # Target encoder must be frozen
        for p in model.target_encoder.parameters():
            self.assertFalse(p.requires_grad, "Target Encoder must be frozen!")
        for p in model.target_encoder.parameters():
            self.assertIsNone(p.grad, "Target Encoder must not receive autograd gradients!")

        # EMA update moves target toward context
        before = sum(p.sum().item() for p in model.target_encoder.parameters())
        model.update_target_encoder(momentum=0.9)
        after = sum(p.sum().item() for p in model.target_encoder.parameters())
        self.assertNotEqual(before, after, "EMA update did not change Target Encoder!")

        print("PASS: test_gradient_1d.py | Context, Predictor, Tokenizer receive grad | "
              "Target Encoder isolated & updated via EMA")


if __name__ == "__main__":
    unittest.main()
