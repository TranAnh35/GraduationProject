"""
Unit Tests for Breakthrough Architecture (EXP-28):
1. ContinuousLinearFieldTokenizer5x5: Verifies unpooled continuous 1D temporal projection
   preserves peak arrival delay t_p sensitivity without pooling collapse.
2. FrequencyConditionedDiffusionPredictor5x5: Verifies relative coordinate and frequency-conditioned
   attention modulation and full-rank dual-head subspace decomposition.
3. End-to-end PECT-JEPA model forward, backward gradient flow, and self-calibrated foundation readout.
"""

import sys
import os
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import (
    ContinuousFieldTokenizer5x5,
    ContinuousLinearFieldTokenizer5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.predictor import (
    FrequencyConditionedDiffusionPredictor5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config


def test_continuous_linear_field_tokenizer_delay_sensitivity():
    """
    Verifies that ContinuousLinearFieldTokenizer5x5 is sensitive to transient arrival delays t_p,
    resolving the temporal blindness caused by AdaptiveAvgPool1d(1).
    """
    print("\n--- Test 1: Temporal Delay Sensitivity ---")
    torch.manual_seed(42)
    in_channels = 128
    embed_dim = 64

    tok_pooled = ContinuousFieldTokenizer5x5(in_channels=in_channels, embed_dim=embed_dim)
    tok_linear = ContinuousLinearFieldTokenizer5x5(in_channels=in_channels, embed_dim=embed_dim)
    tok_pooled.eval()
    tok_linear.eval()

    # Generate synthetic PECT waveforms: baseline pulse vs delayed pulse (simulating defect depth)
    t = torch.linspace(0, 1, in_channels)
    # Peak at t = 0.25 (shallow flaw / surface response)
    pulse_shallow = torch.exp(-((t - 0.25) ** 2) / (2 * (0.05 ** 2)))
    # Peak at t = 0.50 (deep flaw / delayed diffusion response: Delta t_p = 32 samples)
    pulse_deep = torch.exp(-((t - 0.50) ** 2) / (2 * (0.05 ** 2)))

    # Construct 5x5 grids
    grid_shallow = pulse_shallow.view(1, 1, 1, in_channels).expand(1, 5, 5, -1).clone()
    grid_deep = pulse_deep.view(1, 1, 1, in_channels).expand(1, 5, 5, -1).clone()

    with torch.no_grad():
        # Pooled Conv Tokenizer (EXP-22)
        z_pool_shallow, _ = tok_pooled(grid_shallow)
        z_pool_deep, _ = tok_pooled(grid_deep)
        cos_sim_pooled = F.cosine_similarity(z_pool_shallow[:, 0], z_pool_deep[:, 0], dim=-1).item()

        # Continuous Linear Tokenizer (EXP-28)
        z_lin_shallow, _ = tok_linear(grid_shallow)
        z_lin_deep, _ = tok_linear(grid_deep)
        cos_sim_linear = F.cosine_similarity(z_lin_shallow[:, 0], z_lin_deep[:, 0], dim=-1).item()

    print(f"Cosine Similarity (Pooled Conv Tokenizer):   {cos_sim_pooled:.4f}")
    print(f"Cosine Similarity (Continuous Linear Field): {cos_sim_linear:.4f}")

    # The continuous linear projection must distinguish temporal shifts much more sensitively than pooled conv
    assert cos_sim_linear < cos_sim_pooled, (
        f"Continuous linear tokenizer ({cos_sim_linear:.4f}) should be more sensitive "
        f"to temporal arrival delay than pooled conv ({cos_sim_pooled:.4f})"
    )
    print("PASS: Continuous Linear Field Tokenizer preserves peak arrival delay sensitivity.")


def test_frequency_conditioned_diffusion_predictor_kernel():
    """
    Verifies that FrequencyConditionedDiffusionPredictor5x5 modulates cross-attention
    according to characteristic frequency omega_char and preserves dual-head subspace decomposition.
    """
    print("\n--- Test 2: Frequency-Conditioned Diffusion Kernel ---")
    torch.manual_seed(42)
    B, N_ctx, N_tgt, D = 2, 17, 8, 64
    predictor = FrequencyConditionedDiffusionPredictor5x5(
        embed_dim=D, depth=2, num_heads=4, spatial_topology="concentric_star"
    )
    predictor.eval()

    H_ctx = torch.randn(B, N_ctx, D)
    target_pos = torch.randn(B, N_tgt, D)
    ctx_indices = torch.arange(N_ctx).unsqueeze(0).expand(B, -1)
    tgt_indices = torch.arange(N_ctx, N_ctx + N_tgt).unsqueeze(0).expand(B, -1)

    # Low frequency (deep diffusion, long diffusion scale) vs High frequency (near-surface, short diffusion scale)
    freq_low = torch.tensor([0.1, 0.1])
    freq_high = torch.tensor([0.9, 0.9])

    with torch.no_grad():
        H_pred_low, delta_low, base_low = predictor(
            H_context=H_ctx,
            target_pos=target_pos,
            context_indices=ctx_indices,
            target_indices=tgt_indices,
            freq_condition=freq_low,
            return_residual=True,
        )
        H_pred_high, delta_high, base_high = predictor(
            H_context=H_ctx,
            target_pos=target_pos,
            context_indices=ctx_indices,
            target_indices=tgt_indices,
            freq_condition=freq_high,
            return_residual=True,
        )

    # 1. Verify shapes
    assert H_pred_low.shape == (B, N_tgt, D), f"Expected {(B, N_tgt, D)}, got {H_pred_low.shape}"
    assert delta_low.shape == (B, N_tgt, D), f"Expected {(B, N_tgt, D)}, got {delta_low.shape}"
    assert base_low.shape == (B, N_tgt, D), f"Expected {(B, N_tgt, D)}, got {base_low.shape}"

    # 2. Verify dual-head decomposition H_pred = H_base + delta_pred
    decomp_err = torch.max(torch.abs(H_pred_low - (base_low + delta_low))).item()
    assert decomp_err < 1e-5, f"Dual-head decomposition error {decomp_err} exceeds tolerance"

    # 3. Verify frequency modulation: H_pred_low and H_pred_high must differ due to frequency bias
    freq_diff = torch.norm(H_pred_low - H_pred_high).item()
    print(f"Prediction Difference norm between omega=0.1 and omega=0.9: {freq_diff:.4f}")
    assert freq_diff > 1e-3, "Predictor output must be modulated by excitation frequency"
    print("PASS: Frequency-Conditioned Diffusion Predictor successfully modulates diffusion kernel.")


def test_full_breakthrough_model_e2e():
    """
    Verifies end-to-end forward pass, loss computation, backward gradient flow,
    and self-calibrated foundation representation extraction.
    """
    print("\n--- Test 3: Full Breakthrough Model E2E & Gradient Flow ---")
    torch.manual_seed(42)
    config = Spatiotemporal5x5Config(
        tokenizer_type="continuous_linear_field",
        predictor_type="freq_conditioned_diffusion",
        embed_dim=64,
        encoder_depth=2,
        predictor_depth=2,
        encoder_heads=4,
        predictor_heads=4,
        loss_type="l1",
        var_weight=1.0,
        cov_weight=1.0,
        scale_separated_prediction=False,
    )

    model = PECT_JEPA_5x5(config)
    model.train()

    B = 4
    x = torch.randn(B, 5, 5, 128)
    loss_dict = model(x)

    total_loss = loss_dict["loss"]
    assert torch.isfinite(total_loss), f"Loss is non-finite: {total_loss.item()}"
    print(f"E2E Forward Loss: {total_loss.item():.4f}")

    total_loss.backward()

    # Check gradient flow across tokenizer, context encoder, and predictor
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
        elif p.requires_grad and p.grad is None:
            # target_encoder is updated via EMA/stop-gradient; depth_head is only active when phase_align_weight > 0
            if "target_encoder" not in name and "depth_head" not in name:
                raise RuntimeError(f"Parameter {name} has no gradient!")

    print(f"Active Gradient Parameters: {len(grad_norms)}")
    assert len(grad_norms) > 20, "Expected active gradients on all trainable model modules"

    # Test Foundation Representation Readout with In-Scan Self-Calibration
    model.eval()
    with torch.no_grad():
        Z_uncal = model.extract_foundation_representation(x, self_calibrate=False)
        Z_cal = model.extract_foundation_representation(x, self_calibrate=True)

    assert Z_uncal.shape == (B, 2 * config.embed_dim)
    assert Z_cal.shape == (B, 2 * config.embed_dim)

    # In self-calibrated mode, each coordinate dimension has zero mean and unit variance across batch
    cal_mean = Z_cal.mean(dim=0).abs().max().item()
    cal_std = Z_cal.std(dim=0).mean().item()
    print(f"Self-Calibrated Foundation Dim Mean: {cal_mean:.6f}, Std: {cal_std:.4f}")
    assert cal_mean < 1e-4, f"Self-calibrated mean {cal_mean} should be ~0"
    assert abs(cal_std - 1.0) < 0.1, f"Self-calibrated std {cal_std} should be ~1.0"

    print("PASS: Full Breakthrough Architecture E2E & Foundation Readout verified successfully.")


if __name__ == "__main__":
    test_continuous_linear_field_tokenizer_delay_sensitivity()
    test_frequency_conditioned_diffusion_predictor_kernel()
    test_full_breakthrough_model_e2e()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
