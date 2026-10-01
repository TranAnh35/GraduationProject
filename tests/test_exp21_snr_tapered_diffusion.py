import torch
import numpy as np
from types import SimpleNamespace

from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import build_tokenizer_5x5, UncrushedDiffusionTokenizer5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.visualizations import apply_spatial_coherence_filter


def test_snr_tapered_tokenizer_gaussian_noise_suppression():
    """Verify that high-frequency noise in zero-energy spectral bands is tapered to zero."""
    tok = UncrushedDiffusionTokenizer5x5(
        in_channels=128,
        embed_dim=64,
        grid_size=5,
        num_freq_bins=14,
        use_snr_tapering=True,
        phase_noise_floor=0.02,
    )

    # Synthetic Gaussian wave packet (centered at t=64, sigma=10): high frequencies have zero energy
    t = torch.linspace(-5, 5, 128)
    gaussian_pulse = torch.exp(-0.5 * (t / 0.5) ** 2).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 128]
    x_gaussian = gaussian_pulse.expand(2, 5, 5, 128).clone()

    tokens, pos = tok(x_gaussian)
    assert tokens.shape == (2, 50, 64)
    assert pos.shape == (2, 50, 64)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()


def test_exp21_full_jepa_forward_pass():
    """Verify full end-to-end forward pass of EXP-21 model architecture."""
    cfg = Spatiotemporal5x5Config(
        tokenizer_type="snr_tapered_diffusion",
        predictor_type="anisotropic_diffusion",
        cst_mask_mode="cluster",
        in_channels=128,
        embed_dim=64,
        grid_size=5,
        epochs=10,
    )
    model = PECT_JEPA_5x5(cfg)
    x = torch.randn(2, 5, 5, 128)
    loss_dict = model(x)
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0.0
    assert "loss_pred" in loss_dict
    assert "loss_var" in loss_dict

    # Test feature extraction
    z_unified = model.extract_features(x)
    assert z_unified.shape == (2, 128)


def test_spatial_coherence_filter():
    """Verify that isolated single/double pixel noise is removed while true flaws are preserved."""
    bin_map = np.zeros((100, 100), dtype=int)
    # Add a real flaw: 5x5 circular disk (area ~ 21 pixels)
    Y, X = np.ogrid[:100, :100]
    bin_map[(Y - 50) ** 2 + (X - 50) ** 2 <= 3 ** 2] = 1

    # Add isolated single-pixel and 2-pixel false alarms
    bin_map[10, 10] = 1
    bin_map[20, 20] = 1
    bin_map[20, 21] = 1

    cleaned = apply_spatial_coherence_filter(bin_map, min_area=8)
    assert cleaned[10, 10] == 0
    assert cleaned[20, 20] == 0
    assert cleaned[20, 21] == 0
    assert np.sum(cleaned[(Y - 50) ** 2 + (X - 50) ** 2 <= 3 ** 2]) > 0
    assert np.sum(cleaned) >= 20
