"""
Unit tests for Defect Segmentation IoU and Anisotropic Diffusion Predictor (EXP-20).
"""

import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.visualizations import plot_defect_contours_and_iou


def test_benchmark_segmentation_iou():
    """Verify IoU and Dice metric calculation on known binary masks."""
    suite = DownstreamBenchmarkSuite()

    # Ground truth: 10x10 grid with 2x2 defect at center
    gt_mask = np.zeros((10, 10), dtype=np.int8)
    gt_mask[4:6, 4:6] = 1  # 4 defect pixels

    # Prob map: perfect prediction
    prob_map = np.zeros((10, 10), dtype=np.float32)
    prob_map[4:6, 4:6] = 0.9

    res = suite.benchmark_segmentation_iou(prob_map, gt_mask)
    assert res["iou"] == 1.0, f"Expected IoU 1.0, got {res['iou']}"
    assert res["dice"] == 1.0, f"Expected Dice 1.0, got {res['dice']}"
    assert res["optimal_threshold"] <= 0.9

    # Partial overlap (2 pixels overlap out of 4 true and 4 pred, union=6)
    prob_partial = np.zeros((10, 10), dtype=np.float32)
    prob_partial[4:6, 5:7] = 0.9
    res_part = suite.benchmark_segmentation_iou(prob_partial, gt_mask)
    expected_iou = 2.0 / 6.0  # ~0.3333
    assert abs(res_part["iou"] - expected_iou) < 0.01, f"Expected ~0.333, got {res_part['iou']}"


def test_plot_defect_contours_and_iou():
    """Verify contour plotting and metric export."""
    gt_mask = np.zeros((30, 30), dtype=np.int8)
    gt_mask[10:20, 10:20] = 1

    prob_map = np.zeros((30, 30), dtype=np.float32)
    prob_map[11:19, 11:19] = 0.85

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_contour_iou.png")
        res = plot_defect_contours_and_iou(prob_map, gt_mask, save_path=save_path, title="Unit Test Contour")
        assert os.path.isfile(save_path), "Contour image file was not created"
        assert res["iou"] > 0.0
        assert res["dice"] > 0.0


def test_anisotropic_predictor_forward():
    """Verify AnisotropicDiffusionPredictor5x5 initialization, forward pass, and thickness bound."""
    from src.PECT_JEPA.spatiotemporal_5x5.models.predictor import AnisotropicDiffusionPredictor5x5

    embed_dim = 64
    pred = AnisotropicDiffusionPredictor5x5(
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        d_scale_min=1.0,
    )

    B = 2
    N_ctx = 26
    N_tgt = 24

    H_ctx = torch.randn(B, N_ctx, embed_dim)
    target_pos = torch.randn(B, N_tgt, embed_dim)
    ctx_idx = torch.randint(0, 50, (B, N_ctx))
    tgt_idx = torch.randint(0, 50, (B, N_tgt))

    H_pred = pred(
        H_context=H_ctx,
        target_pos=target_pos,
        context_indices=ctx_idx,
        target_indices=tgt_idx,
    )

    assert H_pred.shape == (B, N_tgt, embed_dim), f"Expected shape {(B, N_tgt, embed_dim)}, got {H_pred.shape}"

    # Verify physical depth scale is strictly >= 1.0 mm
    import torch.nn.functional as F
    d_scale = F.softplus(pred.raw_d_scale) + pred.d_scale_min
    assert d_scale.item() >= 1.0, f"d_scale must be >= 1.0 mm, got {d_scale.item()}"

    # Verify gradient computation
    loss = H_pred.sum()
    loss.backward()
    assert pred.raw_alpha_x.grad is not None, "raw_alpha_x must receive gradient"
    assert pred.raw_alpha_y.grad is not None, "raw_alpha_y must receive gradient"
    assert pred.raw_d_scale.grad is not None, "raw_d_scale must receive gradient"


def test_flaw_size_sizing():
    """Verify continuous flaw size / diameter map generation and hurdle regression."""
    from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import GroundTruthManager

    gt_mgr = GroundTruthManager()
    cor_size = gt_mgr.generate_size_map("corrosion", mode="diameter")
    riv_size = gt_mgr.generate_size_map("rivet", mode="diameter")
    mix_size = gt_mgr.generate_size_map("mixed", mode="diameter")

    assert cor_size.shape == (270, 270)
    assert riv_size.shape == (270, 270)
    assert mix_size.shape == (270, 270)

    assert np.max(cor_size) == 10.0
    assert np.max(riv_size) == 7.0
    assert np.max(mix_size) == 14.0

    # Test hurdle regression on simulated features
    suite = DownstreamBenchmarkSuite(n_splits=3)
    np.random.seed(42)
    fake_feats = np.random.randn(270, 270, 16).astype(np.float32)
    # inject correlation
    fake_feats[:, :, 0] += cor_size * 0.5
    res = suite.benchmark_hurdle_depth_regression(fake_feats[::5, ::5], cor_size[::5, ::5])
    assert "compound_hurdle" in res
    assert "conditional_defect_sizing" in res


if __name__ == "__main__":
    test_benchmark_segmentation_iou()
    print("test_benchmark_segmentation_iou: PASSED")
    test_plot_defect_contours_and_iou()
    print("test_plot_defect_contours_and_iou: PASSED")
    test_anisotropic_predictor_forward()
    print("test_anisotropic_predictor_forward: PASSED")
    test_flaw_size_sizing()
    print("test_flaw_size_sizing: PASSED")
