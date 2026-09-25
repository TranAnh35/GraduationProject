"""
Unit and integration test suite for EXP-12: Spatio-Spectral Skin-Depth JEPA.
Tests subband decomposition energy conservation, gradient flow, and full forward/backward pass.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config as PECT5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import SpatioSpectralTokenizer5x5, build_tokenizer_5x5
from src.PECT_JEPA.spatiotemporal_5x5.masking.cluster_mask import build_masker_5x5, ComplementarySpatiotemporalMasker5x5


def test_subband_energy_conservation():
    """Verify that analytic subband decomposition conserves 100% of raw signal energy with zero loss."""
    tokenizer = SpatioSpectralTokenizer5x5(in_channels=128, embed_dim=64, grid_size=5, num_scales=4)
    x = torch.randn(4, 25, 128, dtype=torch.float32)

    X_fft = torch.fft.rfft(x, dim=-1)  # [4, 25, 65]
    x_recon = torch.zeros_like(x)

    for k in range(4):
        X_k = X_fft * tokenizer.windows[k].view(1, 1, -1)
        x_k = torch.fft.irfft(X_k, n=128, dim=-1)
        x_recon = x_recon + x_k

    max_diff = (x - x_recon).abs().max().item()
    assert max_diff < 1e-5, f"Subband reconstruction error too high: {max_diff}"


def test_spatio_spectral_tokenizer_shape_and_grad():
    """Verify tokenizer produces exactly 100 tokens [B, 100, D] and gradients flow cleanly back to x."""
    B, H, W, C, D = 2, 5, 5, 128, 64
    tokenizer = SpatioSpectralTokenizer5x5(in_channels=C, embed_dim=D, grid_size=H, num_scales=4)

    x = torch.randn(B, H, W, C, requires_grad=True)
    tokens, pos = tokenizer(x)

    assert tokens.shape == (B, 100, D), f"Expected (B, 100, D), got {tokens.shape}"
    assert pos.shape == (B, 100, D), f"Expected pos (B, 100, D), got {pos.shape}"
    assert not torch.isnan(tokens).any()

    # Test gradient flow through both time and frequency branches
    loss = tokens.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert x.grad.norm().item() > 0.0


def test_spatio_spectral_masker_surface_to_depth():
    """Verify that ComplementarySpatiotemporalMasker5x5 in surface_to_depth mode correctly assigns surface to ctx and deep to tgt."""
    config = PECT5x5Config(
        tokenizer_type="spatio_spectral",
        cst_mask_mode="surface_to_depth",
        num_spatial_cluster=8,
    )
    masker = build_masker_5x5(config)
    assert isinstance(masker, ComplementarySpatiotemporalMasker5x5)

    B = 4
    ctx_idx, tgt_idx, mask_bool = masker.sample_mask(B)

    # 17 context probes * 2 scales = 34 tokens
    # 8 target probes * 2 scales = 16 tokens
    assert ctx_idx.shape == (B, 34)
    assert tgt_idx.shape == (B, 16)

    # In surface_to_depth mode, target tokens are scales 0 and 1 (deep subsurface)
    tgt_scales = tgt_idx % 4
    assert torch.all((tgt_scales == 0) | (tgt_scales == 1)), "Target tokens must be deep scales 0 and 1"

    # Context tokens are scales 2 and 3 (surface)
    ctx_scales = ctx_idx % 4
    assert torch.all((ctx_scales == 2) | (ctx_scales == 3)), "Context tokens must be surface scales 2 and 3"


def test_full_exp12_jepa_forward_backward():
    """Verify full end-to-end forward pass, loss calculation, and backward update for EXP-12."""
    config = PECT5x5Config(
        tokenizer_type="spatio_spectral",
        cst_mask_mode="surface_to_depth",
        predictor_type="residual_diffusion",
        use_target_ema=False,
        liftoff_invar_weight=0.0,
        phase_align_weight=0.0,
        var_weight=1.0,
        cov_weight=1.0,
        embed_dim=64,
        encoder_depth=2,
        predictor_depth=2,
    )

    model = PECT_JEPA_5x5(config)
    model.train()

    B = 2
    x = torch.randn(B, 5, 5, 128, requires_grad=True)

    loss_dict = model(x)

    assert "loss" in loss_dict
    assert "loss_pred" in loss_dict
    assert "loss_var" in loss_dict
    assert "loss_cov" in loss_dict

    total_loss = loss_dict["loss"]
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)

    total_loss.backward()
    assert x.grad is not None
    assert x.grad.norm().item() > 0.0

    # Test inference feature extraction
    model.eval()
    feat_center = model.extract_center_feature(x)
    assert feat_center.shape == (B, 64)

    feat_all = model.extract_all_features(x)
    assert feat_all.shape == (B, 25, 64)


if __name__ == "__main__":
    print("Running EXP-12 Test Suite...")
    test_subband_energy_conservation()
    print("  [Pass] Test 1: Subband energy conservation")
    test_spatio_spectral_tokenizer_shape_and_grad()
    print("  [Pass] Test 2: Tokenizer shape [B, 100, D] and gradient flow")
    test_spatio_spectral_masker_surface_to_depth()
    print("  [Pass] Test 3: Surface-to-depth CST masker")
    test_full_exp12_jepa_forward_backward()
    print("  [Pass] Test 4: Full EXP-12 forward/backward and feature readouts")
    print("\nALL EXP-12 PIPELINE TESTS PASSED 100%!")
