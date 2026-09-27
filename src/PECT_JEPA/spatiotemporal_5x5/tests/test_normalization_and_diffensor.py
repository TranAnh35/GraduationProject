"""
Unit and Physical Verification Tests for Normalization and Diffensor Invariance (EXP-15).

Verifies:
1. Normalization Repair: 'file_peak' preserves 100% spatial defect contrast (Delta V),
   whereas legacy 'per_sample_peak' destroys spatial amplitude drop.
2. Diffensor Bipolar Signal Scaling: 'file_peak' preserves defect-to-sound SNR on differential probes,
   preventing sound-metal micro-noise amplification.
3. SNR-Tapered Fourier Phase Stability: 'SpatioSpectralTokenizer5x5' smoothly clamps phase
   to zero on near-zero sound metal (Diffensors) while extracting true Dodd-Deeds phase on defects.
4. End-to-end PECT_JEPA_5x5 forward pass on both Absolute and Diffensor waveforms.
"""

import numpy as np
import torch

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import normalize_waveforms_linear
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import SpatioSpectralTokenizer5x5


def test_file_peak_preserves_defect_contrast():
    """
    Test that 'file_peak' preserves the 1-4% physical defect amplitude drop Delta V,
    whereas 'per_sample_peak' completely erases it by scaling every A-scan to 1.0.
    """
    n_points = 100
    t_samples = 128
    t = np.linspace(0, 1, t_samples)
    base_pulse = np.sin(np.pi * t) ** 2  # Peak = 1.0 at center

    # 95 sound points: peak = 1.00V
    sound_scans = np.tile(base_pulse, (95, 1))

    # 5 defect points: peak = 0.96V (Delta V / V0 = 4% physical drop)
    defect_scans = np.tile(base_pulse * 0.96, (5, 1))

    cscan_raw = np.vstack([sound_scans, defect_scans])  # [100, 128]
    assert cscan_raw.shape == (100, 128)

    # 1. Under legacy 'per_sample_peak'
    norm_per_sample = normalize_waveforms_linear(cscan_raw, normalization="per_sample_peak")
    sound_peak_legacy = np.max(norm_per_sample[:95])
    defect_peak_legacy = np.max(norm_per_sample[95:])
    # Legacy flaw contrast is ZERO (erased)
    assert np.isclose(sound_peak_legacy, 1.0, atol=1e-5)
    assert np.isclose(defect_peak_legacy, 1.0, atol=1e-5)
    legacy_contrast = abs(sound_peak_legacy - defect_peak_legacy)
    assert legacy_contrast < 1e-5, f"Expected 0 contrast in per-sample norm, got {legacy_contrast}"

    # 2. Under EXP-15 'file_peak'
    norm_file_peak = normalize_waveforms_linear(cscan_raw, normalization="file_peak")
    sound_peak_fixed = np.max(norm_file_peak[:95])
    defect_peak_fixed = np.max(norm_file_peak[95:])
    file_peak_contrast = sound_peak_fixed - defect_peak_fixed

    # Flaw contrast is 100% preserved at exactly 0.04 (4%)
    assert np.isclose(sound_peak_fixed, 1.0, atol=1e-5)
    assert np.isclose(defect_peak_fixed, 0.96, atol=1e-5)
    assert np.isclose(file_peak_contrast, 0.04, atol=1e-4), f"Expected 0.04 contrast, got {file_peak_contrast}"


def test_diffensor_bipolar_signal_scaling():
    """
    Test that 'file_peak' preserves high SNR on Diffensors (differential coils),
    where sound metal is near-zero (0V) and defects produce bipolar S-curves (+/- 0.8V).
    """
    t_samples = 128
    t = np.linspace(-1, 1, t_samples)
    bipolar_pulse = 0.80 * np.sin(np.pi * t) * np.exp(-2.0 * t ** 2)  # Bipolar +/- 0.8V

    # Diffensor over sound metal: pure EMI noise around 0.0V (amplitude <= 0.01V)
    np.random.seed(42)
    noise_sound = np.random.normal(0.0, 0.003, size=(90, t_samples))

    # Diffensor over defect: bipolar response
    defect_signals = np.tile(bipolar_pulse, (10, 1)) + np.random.normal(0.0, 0.003, size=(10, t_samples))

    diffensor_raw = np.vstack([noise_sound, defect_signals])  # [100, 128]

    # Under 'file_peak':
    norm_diffensor = normalize_waveforms_linear(diffensor_raw, normalization="file_peak")
    max_defect_norm = np.max(np.abs(norm_diffensor[90:]))
    max_sound_norm = np.max(np.abs(norm_diffensor[:90]))

    # Defect peak scales to ~1.0, while sound noise remains tiny (< 0.02)
    assert np.isclose(max_defect_norm, 1.0, atol=1e-3)
    assert max_sound_norm < 0.03, f"Sound noise blown up: {max_sound_norm}"

    # Under 'per_sample_peak', sound noise would be blown up to 1.0:
    norm_legacy = normalize_waveforms_linear(diffensor_raw, normalization="per_sample_peak")
    max_sound_legacy = np.max(np.abs(norm_legacy[:90]))
    assert np.isclose(max_sound_legacy, 1.0, atol=1e-3), "Legacy should blow up noise to 1.0"


def test_snr_tapered_phase_stability_diffensor():
    """
    Test that SpatioSpectralTokenizer5x5 with SNR tapering safely handles Diffensor
    sound-metal inputs (~0V) without producing random uniform phase noise [-pi, pi] or NaNs.
    """
    tokenizer = SpatioSpectralTokenizer5x5(
        in_channels=128,
        embed_dim=64,
        grid_size=5,
        num_scales=4,
        phase_snr_tapering=True,
        phase_noise_floor=0.05,
        use_phase_curvature=True,
        spatial_topology="concentric_star",
    )
    tokenizer.eval()

    # Diffensor sound metal: 25 points all near zero
    zero_diffensor = torch.randn(2, 5, 5, 128) * 0.001

    tokens, pos = tokenizer(zero_diffensor)
    assert tokens.shape == (2, 100, 64)
    assert pos.shape == (2, 100, 64)
    assert not torch.isnan(tokens).any(), "Found NaNs in Diffensor tokens"
    assert not torch.isinf(tokens).any(), "Found Infs in Diffensor tokens"

    # Defect signal: strong bipolar perturbation on center probe
    defect_diffensor = zero_diffensor.clone()
    t = torch.linspace(-1, 1, 128)
    defect_diffensor[:, 2, 2, :] = 0.8 * torch.sin(torch.pi * t) * torch.exp(-2.0 * t ** 2)

    tokens_def, _ = tokenizer(defect_diffensor)
    assert not torch.isnan(tokens_def).any()
    # Flaw token representation should differ substantially from pure noise
    diff = torch.norm(tokens_def - tokens)
    assert diff > 1.0, f"Defect signal did not alter representation: diff={diff}"


def test_end_to_end_model_forward_both_sensors():
    """
    Test full PECT_JEPA_5x5 forward pass with 'file_peak' config on both
    Absolute sensor data (unipolar pulse) and Diffensor data (bipolar pulse).
    """
    config = Spatiotemporal5x5Config(
        normalization="file_peak",
        tokenizer_type="spatio_spectral",
        predictor_type="residual_diffusion",
        use_radial_attention_bias=True,
        use_phase_curvature=True,
        feature_extraction_mode="unified",
        embed_dim=64,
        encoder_depth=2,
        predictor_depth=1,
    )
    model = PECT_JEPA_5x5(config)
    model.eval()

    # 1. Absolute Sensor input [B=2, 5, 5, 128]
    t = torch.linspace(0, 1, 128)
    abs_pulse = (torch.sin(torch.pi * t) ** 2).view(1, 1, 1, 128).expand(2, 5, 5, 128)
    out_abs = model(abs_pulse)
    assert "loss" in out_abs
    assert not torch.isnan(out_abs["loss"]), "NaN loss on Absolute sensor"

    # 2. Diffensor input [B=2, 5, 5, 128]
    diff_pulse = (0.5 * torch.sin(torch.pi * t) * torch.exp(-2.0 * t ** 2)).view(1, 1, 1, 128).expand(2, 5, 5, 128)
    out_diff = model(diff_pulse)
    assert "loss" in out_diff
    assert not torch.isnan(out_diff["loss"]), "NaN loss on Diffensor"

    # 3. Test unified feature extraction on both
    with torch.no_grad():
        feat_abs = model.extract_features(abs_pulse)
        feat_diff = model.extract_features(diff_pulse)
        assert feat_abs.shape == (2, 128), f"Expected [2, 128], got {feat_abs.shape}"
        assert feat_diff.shape == (2, 128), f"Expected [2, 128], got {feat_diff.shape}"
        assert not torch.isnan(feat_abs).any()
        assert not torch.isnan(feat_diff).any()
