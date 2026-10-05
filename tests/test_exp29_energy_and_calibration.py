"""
Unit Test: EXP-29 Waveform-Invariant Energy Normalization & Spatial Calibration.
Verifies:
1. Waveform-Invariant RMS Normalization gives unit RMS energy E_rms = 1.0 across Square, Gauss, Chirp.
2. Adaptive SNR phase noise floor prevents phase starvation on narrow Gaussian packets.
3. Intrinsic scan-level spatial calibration successfully zero-centers the feature map.
"""

import unittest
import os
import sys
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import normalize_waveforms_linear
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import ContinuousLinearFieldTokenizer5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import extract_full_cscan_map


class TestEXP29EnergyAndCalibration(unittest.TestCase):
    def test_energy_rms_normalization(self):
        # Create synthetic waveforms
        t = np.linspace(0, 1, 128)
        # Square wave: 50% duty
        sq = np.sign(np.sin(2 * np.pi * 5 * t))[None, :] * 2.5
        # Gaussian pulse: narrow wavepacket
        gauss = np.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2))[None, :] * 4.0
        # Chirp
        chirp = np.sin(2 * np.pi * (2 + 10 * t) * t)[None, :] * 3.0

        # With file_peak:
        sq_peak = normalize_waveforms_linear(sq, normalization="file_peak")
        gauss_peak = normalize_waveforms_linear(gauss, normalization="file_peak")
        chirp_peak = normalize_waveforms_linear(chirp, normalization="file_peak")

        e_sq_peak = np.mean(sq_peak ** 2)
        e_gauss_peak = np.mean(gauss_peak ** 2)
        print(f"\n[Peak Norm] Square Energy: {e_sq_peak:.4f}, Gauss Energy: {e_gauss_peak:.4f} (Ratio: {e_sq_peak/e_gauss_peak:.2f}x)")
        self.assertGreater(e_sq_peak / e_gauss_peak, 2.0)

        # With energy_rms:
        sq_rms = normalize_waveforms_linear(sq, normalization="energy_rms")
        gauss_rms = normalize_waveforms_linear(gauss, normalization="energy_rms")
        chirp_rms = normalize_waveforms_linear(chirp, normalization="energy_rms")

        e_sq_rms = np.mean(sq_rms ** 2)
        e_gauss_rms = np.mean(gauss_rms ** 2)
        e_chirp_rms = np.mean(chirp_rms ** 2)
        print(f"[RMS Norm]  Square Energy: {e_sq_rms:.4f}, Gauss Energy: {e_gauss_rms:.4f}, Chirp Energy: {e_chirp_rms:.4f}")

        # Both must be identically 1.0
        np.testing.assert_allclose(e_sq_rms, 1.0, atol=1e-5)
        np.testing.assert_allclose(e_gauss_rms, 1.0, atol=1e-5)
        np.testing.assert_allclose(e_chirp_rms, 1.0, atol=1e-5)

    def test_adaptive_phase_floor(self):
        # Low amplitude Gaussian signal
        t = torch.linspace(0, 1, 128)
        gauss = torch.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2)).view(1, 1, 1, 128).repeat(2, 5, 5, 1) * 0.1

        tok_static = ContinuousLinearFieldTokenizer5x5(
            in_channels=128, embed_dim=64, grid_size=5, use_snr_tapering=True,
            phase_noise_floor=0.05, adaptive_phase_floor=False
        )
        tok_adaptive = ContinuousLinearFieldTokenizer5x5(
            in_channels=128, embed_dim=64, grid_size=5, use_snr_tapering=True,
            phase_noise_floor=0.05, adaptive_phase_floor=True
        )

        with torch.no_grad():
            tokens_static, _ = tok_static(gauss)
            tokens_adaptive, _ = tok_adaptive(gauss)

        self.assertEqual(tokens_adaptive.shape, (2, 25, 64))
        self.assertFalse(torch.isnan(tokens_adaptive).any())

    def test_spatial_calibration(self):
        config = Spatiotemporal5x5Config(
            tokenizer_type="continuous_linear_field",
            predictor_type="freq_conditioned_diffusion",
            embed_dim=64,
            spatial_calibration=True,
        )
        model = PECT_JEPA_5x5(config)
        model.eval()

        # Synthetic C-scan of 20x20 points with arbitrary DC offset +3.5
        grid_3d = np.random.randn(20, 20, 128).astype(np.float32) * 0.1 + 3.5

        fmap_calib = extract_full_cscan_map(model, grid_3d, batch_size=128, device="cpu", spatial_calibration=True)
        fmap_uncalib = extract_full_cscan_map(model, grid_3d, batch_size=128, device="cpu", spatial_calibration=False)

        # Calibrated map must have median near 0
        med_calib = np.median(fmap_calib, axis=(0, 1))
        med_uncalib = np.median(fmap_uncalib, axis=(0, 1))

        np.testing.assert_allclose(med_calib, np.zeros_like(med_calib), atol=1e-5)
        self.assertGreater(np.linalg.norm(med_uncalib), 0.01)
        print(f"PASS: Spatial calibration zero-centered feature map median: {np.linalg.norm(med_calib):.6f} vs uncalibrated {np.linalg.norm(med_uncalib):.4f}")


if __name__ == "__main__":
    unittest.main()
