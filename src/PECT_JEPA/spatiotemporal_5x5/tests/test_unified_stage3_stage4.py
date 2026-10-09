"""
Unit Test Suite for Unified Stage 3 & 4 Architecture (EXP-37):
- HarmonicIsometricContextEncoder5x5 (Stage 3: Bilinear Coupled FFN + Normalized Receptive Field Bias)
- DipolarScatteringPredictor5x5 (Stage 4: Analytical Helmholtz Carrier + Maxwell Dipolar Cross-Attention Bias + Dedicated Scattering Head)
- Full End-to-End PECT_JEPA_5x5 Integration, Gradient Flow, Feature Extraction, and Latent Rank.
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.context_encoder import (
    HarmonicIsometricContextEncoder5x5,
    HarmonicIsometricAttentionBias,
    BilinearCoupledFFN,
    build_context_encoder_5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.predictor import (
    DipolarScatteringPredictor5x5,
    AnalyticalHelmholtzCarrierPropagator,
    DipolarAttentionBias,
    build_predictor_5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5


def test_bilinear_coupled_ffn():
    B, N, D = 4, 16, 64
    ffn = BilinearCoupledFFN(embed_dim=D, hidden_dim=D * 4)
    x = torch.randn(B, N, D)
    out = ffn(x)
    assert out.shape == (B, N, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_harmonic_isometric_attention_bias():
    num_heads = 4
    bias_module = HarmonicIsometricAttentionBias(num_heads=num_heads, r_coil_ref=3.0)
    dist = torch.tensor([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]) # [3, 3]
    
    # Test without frequency condition
    bias_default = bias_module(dist)
    assert bias_default.shape == (1, num_heads, 3, 3)
    
    # Test with batch frequency condition (e.g. Chirp vs Square)
    omega_bar = torch.tensor([0.37, 0.105]) # [2]
    dist_batch = dist.unsqueeze(0).expand(2, -1, -1) # [2, 3, 3]
    bias_freq = bias_module(dist_batch, omega_bar=omega_bar)
    assert bias_freq.shape == (2, num_heads, 3, 3)
    assert not torch.isnan(bias_freq).any()


def test_analytical_helmholtz_carrier_propagator():
    propagator = AnalyticalHelmholtzCarrierPropagator(r_coil_ref=3.0)
    B, N_tgt, N_ctx, D = 2, 9, 16, 64
    
    pos_tgt = torch.randn(B, N_tgt, 2)
    pos_ctx = torch.randn(B, N_ctx, 2)
    H_ctx = torch.randn(B, N_ctx, D)
    freq = torch.tensor([0.35, 0.12])
    
    H_carrier, dist_r, delta_eff, delta_r = propagator(pos_tgt, pos_ctx, H_ctx, freq_val=freq)
    
    assert H_carrier.shape == (B, N_tgt, D)
    assert dist_r.shape == (B, N_tgt, N_ctx)
    assert delta_eff.shape == (B, 1, 1)
    assert delta_r.shape == (B, N_tgt, N_ctx, 2)
    
    # Physical property: Square (low freq, sample 1) must have larger skin depth than Chirp (sample 0)
    assert delta_eff[1, 0, 0] > delta_eff[0, 0, 0]


def test_dipolar_attention_bias():
    num_heads = 4
    bias_module = DipolarAttentionBias(num_heads=num_heads)
    B, N_tgt, N_ctx = 2, 9, 16
    
    dist_r = torch.rand(B, N_tgt, N_ctx) * 5.0
    delta_eff = torch.tensor([[[2.5]], [[4.5]]])
    delta_r = torch.randn(B, N_tgt, N_ctx, 2)
    
    bias = bias_module(dist_r, delta_eff, delta_r)
    assert bias.shape == (B, num_heads, N_tgt, N_ctx)
    assert not torch.isnan(bias).any()
    assert not torch.isinf(bias).any()


def test_dipolar_scattering_predictor_standalone():
    predictor = DipolarScatteringPredictor5x5(
        embed_dim=64,
        depth=2,
        num_heads=4,
        r_coil_ref=3.0,
        relative_perturbation_target=True,
    )
    B, N_tgt, N_ctx, D = 4, 9, 16, 64
    H_ctx = torch.randn(B, N_ctx, D)
    target_pos = torch.randn(B, N_tgt, D)
    
    # Inward core mask indices: tgt 0..8, ctx 9..24
    tgt_idx = torch.arange(0, 9).unsqueeze(0).expand(B, -1)
    ctx_idx = torch.arange(9, 25).unsqueeze(0).expand(B, -1)
    freq_cond = torch.tensor([0.3, 0.2, 0.15, 0.35])
    
    H_pred, delta_pred, h_carrier_base = predictor(
        H_context=H_ctx,
        target_pos=target_pos,
        context_indices=ctx_idx,
        target_indices=tgt_idx,
        freq_condition=freq_cond,
        return_residual=True,
    )
    
    assert H_pred.shape == (B, N_tgt, D)
    assert delta_pred.shape == (B, N_tgt, D)
    assert h_carrier_base.shape == (B, N_tgt, D)
    # Check decomposition: H_pred == h_carrier_base + delta_pred
    assert torch.allclose(H_pred, h_carrier_base + delta_pred, atol=1e-5)


def test_full_unified_stage3_stage4_jepa():
    config = Spatiotemporal5x5Config(
        embed_dim=64,
        tokenizer_type="energy_adaptive_dual_domain",
        masker_type="radial_diffusion",
        radial_mask_mode="inward_core",
        encoder_type="harmonic_isometric",
        encoder_depth=4,
        predictor_type="dipolar_scattering",
        predictor_depth=2,
        r_coil_ref=3.0,
        relative_perturbation_target=True,
        keep_absolute_center_feature=True,
        feature_extraction_mode="unified",
        in_channels=128,
        use_target_ema=False,
    )
    
    model = PECT_JEPA_5x5(config)
    
    # 1. Forward Pass
    B = 4
    x = torch.randn(B, 5, 5, 128)
    file_ids = torch.tensor([0, 0, 1, 1])
    
    loss_dict = model(x, file_ids=file_ids)
    assert "loss" in loss_dict
    assert "loss_pred" in loss_dict
    assert "loss_var" in loss_dict
    assert "loss_cov" in loss_dict
    assert not torch.isnan(loss_dict["loss"]).item()
    
    # 2. Backward Pass & Gradient Flow
    loss = loss_dict["loss"]
    loss.backward()
    
    # Context Encoder gradients
    encoder_has_grad = False
    for p in model.context_encoder.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            encoder_has_grad = True
            break
    assert encoder_has_grad, "Context encoder must receive non-zero gradients"
    
    # Predictor gradients
    predictor_has_grad = False
    for p in model.predictor.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            predictor_has_grad = True
            break
    assert predictor_has_grad, "Predictor must receive non-zero gradients"
    
    # Tokenizer gradients
    tok_has_grad = False
    for p in model.tokenizer.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            tok_has_grad = True
            break
    assert tok_has_grad, "Tokenizer must receive non-zero gradients"
    
    # 3. Feature Extraction & Latent Geometry Check
    model.eval()
    with torch.no_grad():
        z_feat = model.extract_features(x)
        assert z_feat.shape == (B, 128), f"Expected [B, 128], got {z_feat.shape}"
        assert not torch.isnan(z_feat).any()
        
        # Test batch SVD effective rank
        x_large = torch.randn(32, 5, 5, 128)
        z_large = model.extract_features(x_large)
        z_centered = z_large - z_large.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(z_centered)
        singular_probs = S / S.sum()
        shannon_entropy = -(singular_probs * torch.log(singular_probs + 1e-12)).sum()
        eff_rank = torch.exp(shannon_entropy).item()
        
        # SVD rank must be healthy (> 10D at init, far above the 1.4D collapse)
        assert eff_rank >= 10.0, f"Effective rank {eff_rank:.2f} is collapsed!"


if __name__ == "__main__":
    test_bilinear_coupled_ffn()
    test_harmonic_isometric_attention_bias()
    test_analytical_helmholtz_carrier_propagator()
    test_dipolar_attention_bias()
    test_dipolar_scattering_predictor_standalone()
    test_full_unified_stage3_stage4_jepa()
    print("ALL TESTS PASSED SUCCESSFULLY!")
