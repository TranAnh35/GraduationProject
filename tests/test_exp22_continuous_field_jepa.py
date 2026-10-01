"""
Unit and Integration Tests for EXP-22:
Continuous Waveform-Agnostic 25-Token Field Topology with Data-Driven
Neural Field Operator Predictor and Latent Subspace Perturbation Decomposition.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import torch.nn as nn

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import get_exp22_config_5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import ContinuousFieldTokenizer5x5, build_tokenizer_5x5
from src.PECT_JEPA.spatiotemporal_5x5.masking.cluster_mask import build_masker_5x5, ContiguousClusterMasker5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.context_encoder import ContextEncoder5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.predictor import NeuralFieldSubspacePredictor5x5, build_predictor_5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.losses.jepa_loss import JEPALoss5x5
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite


def test_continuous_field_tokenizer_shapes():
    """Verify ContinuousFieldTokenizer produces exactly 25 tokens without temporal/depth slicing."""
    B, H, W, C = 4, 5, 5, 128
    D = 64
    x = torch.randn(B, H, W, C)
    tokenizer = ContinuousFieldTokenizer5x5(in_channels=C, embed_dim=D, grid_size=5)

    tokens, pos = tokenizer(x)
    assert tokens.shape == (B, 25, D), f"Expected [4, 25, 64], got {tokens.shape}"
    assert pos.shape == (B, 25, D), f"Expected [4, 25, 64], got {pos.shape}"
    assert torch.isfinite(tokens).all(), "Tokens contain NaN or Inf"


def test_contiguous_cluster_masker_for_25_tokens():
    """Verify masker factory creates ContiguousClusterMasker5x5 for continuous_field."""
    cfg = get_exp22_config_5x5()
    masker = build_masker_5x5(cfg)
    assert isinstance(masker, ContiguousClusterMasker5x5)

    B = 8
    ctx_idx, tgt_idx, mask_bool = masker.sample_mask(B)
    assert ctx_idx.shape[0] == B
    assert tgt_idx.shape[0] == B
    assert mask_bool.shape == (B, 25)

    # Context + Target must span disjoint subsets of 0..24
    for b in range(B):
        ctx_set = set(ctx_idx[b].tolist())
        tgt_set = set(tgt_idx[b].tolist())
        assert len(ctx_set.intersection(tgt_set)) == 0, "Context and Target overlap!"
        assert len(ctx_set) + len(tgt_set) == 25, f"Expected 25 total tokens, got {len(ctx_set) + len(tgt_set)}"


def test_neural_field_subspace_predictor():
    """Verify NeuralFieldSubspacePredictor outputs H_pred, delta_pred, and H_base."""
    B, N_ctx, N_tgt, D = 4, 16, 9, 64
    H_ctx = torch.randn(B, N_ctx, D)
    target_pos = torch.randn(B, N_tgt, D)
    ctx_indices = torch.arange(N_ctx).unsqueeze(0).expand(B, -1)
    tgt_indices = (torch.arange(N_tgt) + N_ctx).unsqueeze(0).expand(B, -1)

    predictor = NeuralFieldSubspacePredictor5x5(embed_dim=D, depth=2, num_heads=4)
    H_pred, delta_pred, H_base = predictor(
        H_context=H_ctx,
        target_pos=target_pos,
        context_indices=ctx_indices,
        target_indices=tgt_indices,
        return_residual=True,
    )

    assert H_pred.shape == (B, N_tgt, D)
    assert delta_pred.shape == (B, N_tgt, D)
    assert H_base.shape == (B, N_tgt, D)
    assert torch.allclose(H_pred, H_base + delta_pred, atol=1e-5), "H_pred != H_base + delta_pred"


def test_jepa_5x5_end_to_end_exp22():
    """Verify full end-to-end forward pass and backward gradient flow in EXP-22 model."""
    cfg = get_exp22_config_5x5()
    model = PECT_JEPA_5x5(cfg)

    B, H, W, C = 4, 5, 5, 128
    x = torch.randn(B, H, W, C, requires_grad=False)
    file_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = model(x, file_ids=file_ids)
    assert "loss" in out
    assert "loss_pred" in out
    assert "loss_pert" in out
    assert "loss_var" in out
    assert "loss_cov" in out

    loss = out["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0

    loss.backward()

    # Check gradients
    has_grad = False
    for p in model.tokenizer.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0.0:
            has_grad = True
            break
    assert has_grad, "Tokenizer has zero gradients!"


def test_universal_flaw_sizing_benchmark():
    """Verify DownstreamBenchmarkSuite.benchmark_universal_flaw_sizing logic."""
    suite = DownstreamBenchmarkSuite()

    # Create dummy 50x50 inspection grid with 2 circular defects
    gt_mask = np.zeros((50, 50), dtype=np.uint8)
    yy, xx = np.ogrid[:50, :50]
    # Defect 1: r=5mm at (15, 15)
    gt_mask[(yy - 15)**2 + (xx - 15)**2 <= 25] = 1
    # Defect 2: r=7mm at (35, 35)
    gt_mask[(yy - 35)**2 + (xx - 35)**2 <= 49] = 1

    # Predicted mask with slight noise/offset
    pred_mask = np.zeros((50, 50), dtype=np.uint8)
    pred_mask[(yy - 15)**2 + (xx - 16)**2 <= 23] = 1
    pred_mask[(yy - 34)**2 + (xx - 35)**2 <= 45] = 1

    sizing_metrics = suite.benchmark_universal_flaw_sizing(pred_mask, gt_mask, pixel_pitch_mm=1.0)
    assert "diameter_sizing" in sizing_metrics
    assert "area_sizing" in sizing_metrics
    assert sizing_metrics["num_matched_flaws"] == 2
    assert sizing_metrics["diameter_sizing"]["mae_mm"] < 1.0


if __name__ == "__main__":
    test_continuous_field_tokenizer_shapes()
    print("test_continuous_field_tokenizer_shapes PASSED")
    test_contiguous_cluster_masker_for_25_tokens()
    print("test_contiguous_cluster_masker_for_25_tokens PASSED")
    test_neural_field_subspace_predictor()
    print("test_neural_field_subspace_predictor PASSED")
    test_jepa_5x5_end_to_end_exp22()
    print("test_jepa_5x5_end_to_end_exp22 PASSED")
    test_universal_flaw_sizing_benchmark()
    print("test_universal_flaw_sizing_benchmark PASSED")
    print("ALL EXP-22 UNIT TESTS PASSED!")
