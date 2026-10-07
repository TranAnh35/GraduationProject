"""
Empirical Verification & Unit Tests for PECT-JEPA v2-A Architecture.
Verifies:
1. Model forward pass on tensor [B, 5, 5, 128]
2. Factorization into z_inv [B, 25, 48] and z_meas [B, 25, 16]
3. RelationalJEPALoss5x5 computation (prediction loss, relational loss, orthogonality, VICReg)
4. Extraction of invariant features for downstream evaluation
5. Numerical stability and backward gradient flow (no NaN / Inf)
"""

import sys
import torch
import torch.nn as nn

ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.factorized_jepa_5x5 import PECT_JEPA_v2
from src.PECT_JEPA.spatiotemporal_5x5.losses.relational_jepa_loss import RelationalJEPALoss5x5


def test_v2_architecture():
    print("="*80)
    print("RUNNING PECT-JEPA v2-A ARCHITECTURAL VERIFICATION SUITE")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = Spatiotemporal5x5Config()
    config.tokenizer_type = "continuous_linear_field"
    config.predictor_type = "freq_conditioned_diffusion"
    config.embed_dim = 64
    config.in_channels = 128
    config.grid_size = 5

    # 1. Instantiate Model
    print("1. Instantiating PECT_JEPA_v2...")
    model = PECT_JEPA_v2(config, d_inv=48, d_meas=16).to(device)
    model.train()
    print(f"   Model initialized successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Simulate Forward Pass
    B = 8
    x_A = torch.randn(B, 5, 5, 128, device=device)
    x_B = torch.randn(B, 5, 5, 128, device=device)
    file_ids = torch.randint(0, 3, (B,), device=device)

    print("\n2. Simulating standard batch forward pass...")
    loss_dict_base = model(x_A, file_ids=file_ids)
    assert "loss" in loss_dict_base, "Missing loss key in base output"
    print(f"   Base JEPA Loss: {loss_dict_base['loss'].item():.4f}")

    # 3. Factorization and Relational Loss
    print("\n3. Testing Token Factorization and Relational Consistency...")
    tokens_A, pos_A = model.tokenizer(x_A)
    H_A = model.context_encoder(tokens_A, pos_A)
    z_inv_A, z_meas_A = model.factorize_tokens(H_A)

    tokens_B, pos_B = model.tokenizer(x_B)
    H_B = model.context_encoder(tokens_B, pos_B)
    z_inv_B, z_meas_B = model.factorize_tokens(H_B)

    assert z_inv_A.shape == (B, 25, 48), f"Unexpected z_inv shape: {z_inv_A.shape}"
    assert z_meas_A.shape == (B, 25, 16), f"Unexpected z_meas shape: {z_meas_A.shape}"
    print(f"   Token shapes verified: z_inv={list(z_inv_A.shape)}, z_meas={list(z_meas_A.shape)}")

    loss_fn = RelationalJEPALoss5x5(
        relational_weight=1.0,
        orthogonality_weight=0.5,
        var_weight=1.0,
        cov_weight=1.0,
    ).to(device)

    loss_dict_v2 = loss_fn.forward_factorized(
        base_jepa_loss_dict=loss_dict_base,
        z_inv_A=z_inv_A,
        z_meas_A=z_meas_A,
        z_inv_B=z_inv_B,
        file_ids=file_ids,
    )

    print(f"   Total v2 Loss:         {loss_dict_v2['loss'].item():.4f}")
    print(f"   - Relational Loss:     {loss_dict_v2['loss_rel'].item():.4f}")
    print(f"   - Orthogonality Loss:  {loss_dict_v2['loss_orth'].item():.4f}")
    print(f"   - Invariant Var Loss:  {loss_dict_v2['loss_var_inv'].item():.4f}")
    print(f"   - Invariant Cov Loss:  {loss_dict_v2['loss_cov_inv'].item():.4f}")

    # 4. Backward Pass and Gradient Check
    print("\n4. Verifying Backward Gradient Flow...")
    loss_dict_v2["loss"].backward()
    has_nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                has_nan_grad = True
                print(f"   ERROR: NaN/Inf gradient in {name}")
    assert not has_nan_grad, "Detected NaN or Inf gradient!"
    print("   Backward pass successful: all gradients are finite and stable.")

    # 5. Downstream Invariant Feature Extractor Check
    print("\n5. Testing Invariant Feature Extraction for Downstream Evaluation...")
    model.eval()
    with torch.no_grad():
        z_eval = model.extract_invariant_features(x_A)
        assert z_eval.shape == (B, 48), f"Unexpected extract shape: {z_eval.shape}"
        print(f"   Extracted invariant center probe shape: {list(z_eval.shape)} [B, d_inv=48]")

    print("\n" + "="*80)
    print("ALL TESTS PASSED SUCCESSFULLY! PECT-JEPA v2-A IS STABLE AND VERIFIED.")
    print("="*80)


if __name__ == "__main__":
    test_v2_architecture()
