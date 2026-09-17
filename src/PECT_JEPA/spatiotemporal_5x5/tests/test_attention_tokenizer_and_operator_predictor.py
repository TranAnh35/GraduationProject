"""
Unit and Integration Tests for DualDomainAttentionTokenizer5x5 and OperatorDiffusionPredictor5x5.
"""

import unittest
import torch

from ..configs.config import Spatiotemporal5x5Config
from ..models.tokenizer_5x5 import DualDomainAttentionTokenizer5x5, build_tokenizer_5x5
from ..models.predictor import (
    DiffusionOperatorEmbedding,
    OperatorDiffusionPredictor5x5,
    build_predictor_5x5,
)
from ..models.jepa_5x5 import PECT_JEPA_5x5


class TestAttentionTokenizerAndOperatorPredictor(unittest.TestCase):

    def setUp(self):
        self.B = 4
        self.grid_size = 5
        self.in_channels = 128
        self.embed_dim = 64
        self.num_freq_bins = 14

    def test_dual_domain_attention_tokenizer_forward_and_grads(self):
        tok = DualDomainAttentionTokenizer5x5(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            grid_size=self.grid_size,
            num_freq_bins=self.num_freq_bins,
            num_heads=4,
        )
        x = torch.randn(self.B, self.grid_size, self.grid_size, self.in_channels, requires_grad=True)
        tokens, pos = tok(x)

        # Verify output shapes
        self.assertEqual(tokens.shape, (self.B, 25, self.embed_dim))
        self.assertEqual(pos.shape, (self.B, 25, self.embed_dim))
        self.assertFalse(torch.isnan(tokens).any(), "NaN detected in tokens!")
        self.assertFalse(torch.isinf(tokens).any(), "Inf detected in tokens!")

        # Verify backward gradient flow to input and domain embeddings
        loss = tokens.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(tok.domain_time.grad)
        self.assertIsNotNone(tok.domain_freq.grad)
        self.assertIsNotNone(tok.time_proj.weight.grad)
        self.assertIsNotNone(tok.freq_proj.weight.grad)

    def test_diffusion_operator_embedding(self):
        op_emb = DiffusionOperatorEmbedding(embed_dim=self.embed_dim, num_freq_bands=8)

        # 1. Continuous frequency in range (0, 1]
        freq_cont = torch.tensor([0.1, 0.5, 0.8, 1.0])
        q_cont = op_emb(freq_cont)
        self.assertEqual(q_cont.shape, (4, 1, self.embed_dim))
        self.assertFalse(torch.isnan(q_cont).any())

        # 2. Fallback default condition
        q_default = op_emb(None)
        self.assertEqual(q_default.shape, (1, 1, self.embed_dim))

    def test_operator_diffusion_predictor_forward(self):
        pred = OperatorDiffusionPredictor5x5(
            embed_dim=self.embed_dim,
            depth=2,
            num_heads=4,
            num_freq_bins=self.num_freq_bins,
        )
        N_ctx = 13
        N_tgt = 12
        H_ctx = torch.randn(self.B, N_ctx, self.embed_dim)
        target_pos = torch.randn(self.B, N_tgt, self.embed_dim)
        freq_idx = torch.randint(1, 15, (self.B,))

        # Forward with frequency conditioning
        H_pred = pred(H_context=H_ctx, target_pos=target_pos, freq_condition=freq_idx)
        self.assertEqual(H_pred.shape, (self.B, N_tgt, self.embed_dim))
        self.assertFalse(torch.isnan(H_pred).any())

        # Forward without frequency condition (uses default base operator)
        H_pred_default = pred(H_context=H_ctx, target_pos=target_pos)
        self.assertEqual(H_pred_default.shape, (self.B, N_tgt, self.embed_dim))

    def test_end_to_end_jepa_64dim_forward_and_backward(self):
        config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=64,
            tokenizer_type="dual_domain_attention",
            predictor_type="operator_diffusion",
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            min_masked=10,
            max_masked=15,
        )
        model = PECT_JEPA_5x5(config)
        model.train()

        x = torch.randn(self.B, 5, 5, 128, requires_grad=True)
        out = model(x)

        self.assertIn("loss", out)
        self.assertIn("loss_pred", out)
        self.assertIn("loss_var", out)
        self.assertIn("loss_cov", out)
        self.assertFalse(torch.isnan(out["loss"]).any())

        # Test backward pass
        loss = out["loss"]
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
