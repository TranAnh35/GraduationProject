"""
Test suite for EXP-13: Multi-Scale Concentric Star (Octagram) Topology PECT-JEPA.
"""

import sys
import os
import time
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import (
    get_spatial_topology_offsets,
    compute_physical_distance_matrix,
)
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.data.dataset import PECT5x5Dataset
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import extract_full_cscan_map


def test_concentric_star_geometry():
    print("  Testing Concentric Star geometry...")
    offsets = get_spatial_topology_offsets("concentric_star", star_radii=(1, 3, 7))
    assert offsets.shape == (25, 2), f"Expected (25, 2), got {offsets.shape}"
    assert (offsets[0] == [0, 0]).all(), f"Center must be (0, 0), got {offsets[0]}"

    dists = np.sqrt(np.sum(offsets.astype(np.float32) ** 2, axis=-1))
    # Point 0: dist 0
    assert dists[0] == 0.0
    # Ring 1: dist in [1.0, 1.42]
    assert np.all((dists[1:9] >= 0.99) & (dists[1:9] <= 1.42))
    # Ring 2: dist in [3.0, 4.25]
    assert np.all((dists[9:17] >= 2.99) & (dists[9:17] <= 4.25))
    # Ring 3: dist in [7.0, 7.08]
    assert np.all((dists[17:25] >= 6.99) & (dists[17:25] <= 7.08))

    D = compute_physical_distance_matrix(offsets, step_mm=1.0)
    assert D.shape == (25, 25)
    assert torch.allclose(torch.diag(D), torch.zeros(25))
    assert torch.allclose(D, D.T)
    print("    [Pass] 25 star offsets and physical distance matrix verified.")


def test_dataset_topology_extraction():
    print("  Testing Dataset topology extraction...")
    # Synthetic 2D grid [100, 100, 128]
    dummy_cscan = np.random.randn(100, 100, 128).astype(np.float32)
    offsets = get_spatial_topology_offsets("concentric_star", star_radii=(1, 3, 7))
    max_off = int(np.max(np.abs(offsets)))
    padded = np.pad(dummy_cscan, ((max_off, max_off), (max_off, max_off), (0, 0)), mode="edge")

    # Center (50, 50)
    cr, cc = 50 + max_off, 50 + max_off
    sample_r = cr + offsets[:, 0]
    sample_c = cc + offsets[:, 1]
    patch = padded[sample_r, sample_c, :].reshape(5, 5, 128)
    assert patch.shape == (5, 5, 128)
    assert np.allclose(patch[0, 0, :], padded[cr, cc, :])
    print("    [Pass] Dataset patch extraction verified.")


def test_full_exp13_pipeline():
    print("  Testing Full EXP-13 JEPA Pipeline...")
    cfg = Spatiotemporal5x5Config(
        tokenizer_type="spatio_spectral",
        cst_mask_mode="surface_to_depth",
        spatial_topology="concentric_star",
        star_radii=(1, 3, 7),
        predictor_type="residual_diffusion",
        use_target_ema=False,
        embed_dim=64,
        encoder_depth=2,
        predictor_depth=2,
    )
    model = PECT_JEPA_5x5(cfg)

    x = torch.randn(4, 5, 5, 128, requires_grad=True)
    loss_dict = model(x)

    assert "loss" in loss_dict
    assert "H_pred" in loss_dict
    assert "H_tgt" in loss_dict
    assert loss_dict["H_pred"].shape[-1] == 64

    # Test backward pass
    loss = loss_dict["loss"]
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("    [Pass] Forward/backward pass clean, loss =", float(loss.item()))


def test_fast_vectorized_extractor():
    print("  Testing Fast Vectorized C-Scan Extractor...")
    cfg = Spatiotemporal5x5Config(
        tokenizer_type="spatio_spectral",
        spatial_topology="concentric_star",
        star_radii=(1, 3, 7),
        embed_dim=64,
    )
    model = PECT_JEPA_5x5(cfg)

    # 40x40 C-scan grid
    dummy_cscan = np.random.randn(40, 40, 128).astype(np.float32)
    t0 = time.time()
    feat_map = extract_full_cscan_map(
        model=model,
        full_cscan_3d=dummy_cscan,
        batch_size=1024,
        device="cpu",
    )
    t1 = time.time()
    assert feat_map.shape == (40, 40, 64)
    print(f"    [Pass] Vectorized feature map shape {feat_map.shape} in {(t1 - t0)*1000:.1f}ms")


if __name__ == "__main__":
    print("Running EXP-13 Concentric Star Pipeline Tests...")
    test_concentric_star_geometry()
    test_dataset_topology_extraction()
    test_full_exp13_pipeline()
    test_fast_vectorized_extractor()
    print("\nALL EXP-13 PIPELINE TESTS PASSED 100%!")
