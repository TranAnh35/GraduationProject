import os, sys
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np

from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import ContinuousSTFTokenizer5x5, build_tokenizer_5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config


def test_continuous_stf_tokenizer_shapes_and_gradients():
    B, H, W, C, D = 4, 5, 5, 128, 64
    tokenizer = ContinuousSTFTokenizer5x5(
        in_channels=C,
        embed_dim=D,
        grid_size=H,
        num_freq_bins=14,
        pos_embed_type="learnable_2d",
        dropout=0.0
    )

    x = torch.randn(B, H, W, C, requires_grad=True)
    tokens, pos = tokenizer(x)

    assert tokens.shape == (B, 25, D), f"Expected tokens shape (4, 25, 64), got {tokens.shape}"
    assert pos.shape == (B, 25, D), f"Expected pos shape (4, 25, 64), got {pos.shape}"
    assert not torch.isnan(tokens).any(), "NaN found in output tokens"
    assert not torch.isinf(tokens).any(), "Inf found in output tokens"

    # Backward pass verification
    loss = tokens.mean()
    loss.backward()

    assert tokenizer.conv_short.weight.grad is not None
    assert tokenizer.conv_med.weight.grad is not None
    assert tokenizer.conv_long.weight.grad is not None
    assert tokenizer.freq_proj.weight.grad is not None
    assert tokenizer.fuse_proj.weight.grad is not None
    assert not torch.isnan(tokenizer.conv_short.weight.grad).any()
    assert not torch.isnan(tokenizer.freq_proj.weight.grad).any()


def test_continuous_stf_multi_waveform_inputs():
    """Verify tokenizer processes Square, Gaussian, and Chirp waveforms cleanly without NaN."""
    B, H, W, C, D = 2, 5, 5, 128, 64
    tokenizer = ContinuousSTFTokenizer5x5(in_channels=C, embed_dim=D, grid_size=H)

    t = torch.linspace(0, 1, C)

    # 1. Synthetic Square pulse with exponential decay
    sq = torch.where(t < 0.3, torch.ones_like(t), torch.exp(-5.0 * (t - 0.3)))
    x_sq = sq.view(1, 1, 1, C).expand(B, H, W, C).clone()

    # 2. Synthetic Gaussian wave packet
    gauss = torch.exp(-0.5 * ((t - 0.5) / 0.1) ** 2) * torch.cos(2 * torch.pi * 10.0 * t)
    x_gauss = gauss.view(1, 1, 1, C).expand(B, H, W, C).clone()

    # 3. Synthetic Chirp (linear frequency sweep from f1 to f2)
    chirp = torch.cos(2 * torch.pi * (5.0 * t + 15.0 * (t ** 2)))
    x_chirp = chirp.view(1, 1, 1, C).expand(B, H, W, C).clone()

    for name, inp in [("Square", x_sq), ("Gaussian", x_gauss), ("Chirp", x_chirp)]:
        tok, pos = tokenizer(inp)
        assert tok.shape == (B, 25, D), f"Failed shape for {name}"
        assert not torch.isnan(tok).any(), f"NaN in tokens for {name}"
        assert not torch.isinf(tok).any(), f"Inf in tokens for {name}"


def test_continuous_stf_jepa_integration():
    """Verify end-to-end forward pass in PECT_JEPA_5x5 full model."""
    cfg = Spatiotemporal5x5Config(
        tokenizer_type="continuous_stf",
        embed_dim=64,
        in_channels=128,
        encoder_depth=2,
        predictor_depth=2,
        var_weight=1.0,
        cov_weight=1.0,
        liftoff_invar_weight=0.05,
        phase_align_weight=0.05
    )

    model = PECT_JEPA_5x5(cfg)
    x = torch.randn(4, 5, 5, 128)

    # Forward self-supervised step
    loss_dict = model(x)
    assert "loss" in loss_dict
    assert "loss_pred" in loss_dict
    assert "loss_var" in loss_dict
    assert "loss_cov" in loss_dict
    assert not torch.isnan(loss_dict["loss"])

    # Center feature extraction
    z_center = model.extract_center_feature(x)
    assert z_center.shape == (4, 64)
    assert not torch.isnan(z_center).any()

    # All features extraction
    z_all = model.extract_all_features(x)
    assert z_all.shape == (4, 25, 64)
    assert not torch.isnan(z_all).any()


if __name__ == "__main__":
    print("Running test_continuous_stf_tokenizer_shapes_and_gradients...")
    test_continuous_stf_tokenizer_shapes_and_gradients()
    print("Running test_continuous_stf_multi_waveform_inputs...")
    test_continuous_stf_multi_waveform_inputs()
    print("Running test_continuous_stf_jepa_integration...")
    test_continuous_stf_jepa_integration()
    print("ALL TESTS PASSED SUCCESSFULLY!")

