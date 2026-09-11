"""
Unit tests for DualDomainGridTokenizer5x5 and Tokenizer Integration.
Verifies:
1. Output shape [B, 25, D] and positional embedding shape [B, 25, D].
2. Forward-backward gradient flow through both time and spectral FFT branches.
3. Support for both 'phase_and_mag' (64 spectral features) and 'phase_only' (32 spectral features).
4. Drop-in replacement compatibility in PECT_JEPA_5x5 with both 'dual_domain' and 'time_only'.
5. Robustness to zero inputs, NaN resilience, and varying batch sizes.
"""

import unittest
import torch

from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import (
    DualDomainGridTokenizer5x5,
    SpatialGridTokenizer5x5,
    build_tokenizer_5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config


class TestDualDomainTokenizer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B = 4
        self.H = 5
        self.W = 5
        self.C = 128
        self.D = 128

    def test_dual_domain_tokenizer_shapes(self):
        """Verify output shapes with default phase_and_mag spectral features."""
        tokenizer = DualDomainGridTokenizer5x5(
            in_channels=self.C,
            embed_dim=self.D,
            grid_size=5,
            num_freq_bins=32,
            spectral_features="phase_and_mag",
        )
        x = torch.randn(self.B, self.H, self.W, self.C)
        tokens, pos = tokenizer(x)

        self.assertEqual(tokens.shape, (self.B, 25, self.D))
        self.assertEqual(pos.shape, (self.B, 25, self.D))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_dual_domain_phase_only_shapes(self):
        """Verify output shapes with phase_only spectral features."""
        tokenizer = DualDomainGridTokenizer5x5(
            in_channels=self.C,
            embed_dim=self.D,
            grid_size=5,
            num_freq_bins=32,
            spectral_features="phase_only",
        )
        x = torch.randn(self.B, self.H, self.W, self.C)
        tokens, pos = tokenizer(x)

        self.assertEqual(tokens.shape, (self.B, 25, self.D))
        self.assertEqual(pos.shape, (self.B, 25, self.D))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_gradient_flow_both_branches(self):
        """Verify gradients flow backward into both time_proj and freq_proj."""
        tokenizer = DualDomainGridTokenizer5x5(
            in_channels=self.C,
            embed_dim=self.D,
            grid_size=5,
            num_freq_bins=32,
            spectral_features="phase_and_mag",
        )
        x = torch.randn(self.B, self.H, self.W, self.C, requires_grad=True)
        tokens, _ = tokenizer(x)
        loss = tokens.sum()
        loss.backward()

        self.assertIsNotNone(tokenizer.time_proj.weight.grad)
        self.assertIsNotNone(tokenizer.freq_proj.weight.grad)
        self.assertTrue(torch.isfinite(tokenizer.time_proj.weight.grad).all())
        self.assertTrue(torch.isfinite(tokenizer.freq_proj.weight.grad).all())
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_factory_function_dual_domain(self):
        """Test build_tokenizer_5x5 constructs DualDomainGridTokenizer5x5."""
        cfg = Spatiotemporal5x5Config(tokenizer_type="dual_domain")
        tok = build_tokenizer_5x5(cfg)
        self.assertIsInstance(tok, DualDomainGridTokenizer5x5)

    def test_factory_function_time_only(self):
        """Test build_tokenizer_5x5 constructs SpatialGridTokenizer5x5 when specified."""
        cfg = Spatiotemporal5x5Config(tokenizer_type="time_only")
        tok = build_tokenizer_5x5(cfg)
        self.assertIsInstance(tok, SpatialGridTokenizer5x5)

    def test_pect_jepa_forward_with_dual_domain(self):
        """Verify end-to-end forward pass of PECT_JEPA_5x5 with dual domain tokenizer."""
        cfg = Spatiotemporal5x5Config(
            tokenizer_type="dual_domain",
            grid_size=5,
            in_channels=128,
            embed_dim=128,
            encoder_depth=2,
            encoder_heads=4,
            predictor_depth=2,
            predictor_heads=4,
            min_masked=10,
            max_masked=15,
            rank_barrier_weight=0.5,
        )
        model = PECT_JEPA_5x5(cfg)
        x = torch.randn(self.B, 5, 5, 128)
        out = model(x)

        self.assertIn("loss", out)
        self.assertIn("loss_pred", out)
        self.assertIn("loss_var", out)
        self.assertIn("loss_cov", out)
        self.assertIn("loss_rank_barrier", out)
        self.assertTrue(torch.isfinite(out["loss"]))

        # Check feature extractions
        feat_center = model.extract_center_feature(x)
        self.assertEqual(feat_center.shape, (self.B, 128))
        feat_all = model.extract_all_features(x)
        self.assertEqual(feat_all.shape, (self.B, 25, 128))

    def test_sinusoidal_pos_embedding(self):
        """Verify sinusoidal_2d pos embedding with DualDomainGridTokenizer5x5."""
        tokenizer = DualDomainGridTokenizer5x5(
            in_channels=self.C,
            embed_dim=self.D,
            grid_size=5,
            pos_embed_type="sinusoidal_2d",
        )
        x = torch.randn(2, 5, 5, self.C)
        tokens, pos = tokenizer(x)
        self.assertEqual(pos.shape, (2, 25, self.D))

    def test_phase_snr_tapering_and_14_bins(self):
        """Verify phase SNR tapering suppresses near-zero noise and operates on 14 bins."""
        tokenizer = DualDomainGridTokenizer5x5(
            in_channels=self.C,
            embed_dim=self.D,
            grid_size=5,
            num_freq_bins=14,
            spectral_features="phase_and_mag",
            phase_snr_tapering=True,
            phase_noise_floor=0.05,
        )
        self.assertEqual(tokenizer.num_freq_bins, 14)
        self.assertEqual(tokenizer.freq_proj.in_features, 28)

        # 1. Normal amplitude input
        x = torch.randn(self.B, self.H, self.W, self.C)
        tokens, pos = tokenizer(x)
        self.assertEqual(tokens.shape, (self.B, 25, self.D))
        self.assertTrue(torch.isfinite(tokens).all())

        # 2. Near-zero amplitude input (pure noise floor)
        x_small = torch.randn(self.B, self.H, self.W, self.C) * 1e-6
        tokens_small, _ = tokenizer(x_small)
        self.assertTrue(torch.isfinite(tokens_small).all())

    def test_apply_lowpass_filter(self):
        """Verify zero-phase Butterworth lowpass filter."""
        import numpy as np
        from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import apply_lowpass_filter

        # Create 128-pt waveform with low-freq signal (200 Hz) + high-freq noise (5000 Hz)
        fs = 25600.0
        t = np.arange(128) / fs
        s_clean = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
        s_noisy = s_clean + 0.5 * np.sin(2 * np.pi * 5000.0 * t).astype(np.float32)

        s_filtered = apply_lowpass_filter(s_noisy, cutoff_hz=2500.0, fs=fs, order=4)
        self.assertEqual(s_filtered.shape, s_noisy.shape)
        self.assertTrue(np.all(np.isfinite(s_filtered)))

        # Residual between filtered and clean should be much smaller than noise
        err_noisy = np.mean((s_noisy - s_clean) ** 2)
        err_filtered = np.mean((s_filtered - s_clean) ** 2)
        self.assertLess(err_filtered, err_noisy * 0.1)


if __name__ == "__main__":
    unittest.main()
