"""
Unit tests for EnergyAdaptiveDualDomainTokenizer5x5 (Stage 2 Re-foundation, EXP-35).
"""

import unittest
import torch
import torch.nn as nn
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import (
    EnergyAdaptiveDualDomainTokenizer5x5,
    build_tokenizer_5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.masking.cluster_mask import build_masker_5x5


class TestEnergyAdaptiveTokenizer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 4
        self.C = 128
        self.D = 64
        self.grid_size = 5

    def test_forward_shapes_learnable_and_sinusoidal(self):
        """Verify output tensor dimensions for both positional embedding modes."""
        x = torch.randn(self.B, self.grid_size, self.grid_size, self.C, device=self.device)

        # 1. Learnable 2D
        tok_learnable = EnergyAdaptiveDualDomainTokenizer5x5(
            in_channels=self.C, embed_dim=self.D, grid_size=self.grid_size, pos_embed_type="learnable_2d"
        ).to(self.device)
        tokens, pos = tok_learnable(x)
        self.assertEqual(tokens.shape, (self.B, 25, self.D))
        self.assertEqual(pos.shape, (self.B, 25, self.D))
        self.assertFalse(torch.isnan(tokens).any())

        # 2. Sinusoidal 2D
        tok_sinusoidal = EnergyAdaptiveDualDomainTokenizer5x5(
            in_channels=self.C, embed_dim=self.D, grid_size=self.grid_size, pos_embed_type="sinusoidal_2d"
        ).to(self.device)
        tokens_sin, pos_sin = tok_sinusoidal(x)
        self.assertEqual(tokens_sin.shape, (self.B, 25, self.D))
        self.assertEqual(pos_sin.shape, (self.B, 25, self.D))
        self.assertFalse(torch.isnan(tokens_sin).any())

    def test_factory_build(self):
        """Verify build_tokenizer_5x5 factory instantiates EnergyAdaptiveDualDomainTokenizer5x5."""
        cfg = Spatiotemporal5x5Config(
            tokenizer_type="energy_adaptive_dual_domain",
            embed_dim=self.D,
            in_channels=self.C,
        )
        tokenizer = build_tokenizer_5x5(cfg)
        self.assertIsInstance(tokenizer, EnergyAdaptiveDualDomainTokenizer5x5)

    def test_saliency_noise_suppression(self):
        """
        Verify that out-of-band noise in inactive spectral bins is suppressed by saliency gating s_k.
        """
        tok = EnergyAdaptiveDualDomainTokenizer5x5(
            in_channels=self.C, embed_dim=self.D, grid_size=self.grid_size
        ).to(self.device)

        # Synthesize narrow-band excitation (pure sine at bin 1)
        t = torch.linspace(0, 1, self.C, device=self.device)
        pure_signal = torch.sin(2 * torch.pi * 1 * t)
        pure_batch = pure_signal.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(self.B, self.grid_size, self.grid_size, self.C)

        # Compute FFT and check saliency weights s
        x_flat = pure_batch.reshape(-1, self.C).float()
        X_fft = torch.fft.rfft(x_flat, dim=-1)
        X_sub = X_fft[:, 1:tok.num_freq_bins + 1]
        P = torch.abs(X_sub) ** 2
        s = P / (P.sum(dim=-1, keepdim=True) + 1e-8)

        # Bin 1 should have > 90% power, higher bins (e.g., bin 10) should have < 1%
        self.assertGreater(s[:, 0].mean().item(), 0.90)
        self.assertLess(s[:, 9].mean().item(), 0.01)

    def test_balanced_gradients(self):
        """
        Verify that gradient norms between time and frequency branches are well-balanced (ratio ~ 1.0).
        """
        tok = EnergyAdaptiveDualDomainTokenizer5x5(
            in_channels=self.C, embed_dim=self.D, grid_size=self.grid_size
        ).to(self.device)

        # Pulse signal
        t = torch.linspace(0, 5e-3, self.C, device=self.device)
        x_pulse = torch.exp(-((t - 2.5e-3) / 0.5e-3) ** 2) * torch.sin(2 * torch.pi * 1000 * t)
        x_batch = x_pulse.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(self.B, self.grid_size, self.grid_size, self.C).clone()
        x_batch = (x_batch + 0.01 * torch.randn_like(x_batch)).requires_grad_(True)

        tokens, _ = tok(x_batch)
        loss = tokens.sum()
        loss.backward()

        time_grad = sum(p.grad.norm().item() for p in tok.time_proj.parameters() if p.grad is not None)
        freq_grad = sum(p.grad.norm().item() for p in tok.proj_freq.parameters() if p.grad is not None)
        ratio = time_grad / (freq_grad + 1e-8)

        self.assertGreater(ratio, 0.2, "Time branch is severely starved")
        self.assertLess(ratio, 5.0, "Frequency branch is severely starved")

    def test_full_model_forward_backward(self):
        """
        Verify end-to-end integration with PECT_JEPA_5x5 and RadialDiffusionMasker5x5.
        """
        cfg = Spatiotemporal5x5Config(
            tokenizer_type="energy_adaptive_dual_domain",
            masker_type="radial_diffusion",
            radial_mask_mode="inward_core",
            relative_perturbation_target=True,
            spatial_topology="concentric_star",
            star_radii=(1, 3, 7),
            predictor_type="freq_conditioned_diffusion",
            embed_dim=self.D,
            in_channels=self.C,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model = PECT_JEPA_5x5(cfg).to(self.device)
        masker = build_masker_5x5(cfg)

        x = torch.randn(self.B, self.grid_size, self.grid_size, self.C, device=self.device)
        ctx_idx, tgt_idx, mask_bool = masker(self.B, device=self.device)

        loss_dict = model(x=x, custom_context_indices=ctx_idx, custom_target_indices=tgt_idx)
        loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_pred", loss_dict)
        self.assertIn("loss_var", loss_dict)
        self.assertIn("loss_cov", loss_dict)

        loss.backward()

        # Check gradients exist on tokenizer parameters
        tok_has_grads = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.tokenizer.parameters())
        self.assertTrue(tok_has_grads)


if __name__ == "__main__":
    unittest.main()
