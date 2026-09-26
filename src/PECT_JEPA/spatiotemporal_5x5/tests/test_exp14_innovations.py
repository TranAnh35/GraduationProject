"""
Comprehensive Unit & Physical Verification Test for EXP-14 Innovations:
1. Continuous Radial Distance Attention Bias in ContextEncoder.
2. Harmonic Radial Phase Curvature in SpatioSpectralTokenizer.
3. Unified Latent Feature Extraction ([H_ctx; Delta_H]) in PECT_JEPA_5x5.
4. Two-Stage Hurdle Depth Regression Benchmark in DownstreamBenchmarks.
"""

import torch
import numpy as np

from ..configs.config import Spatiotemporal5x5Config
from ..models.attention import RadialAttentionBias, TransformerBlock
from ..models.context_encoder import ContextEncoder5x5
from ..models.tokenizer_5x5 import SpatioSpectralTokenizer5x5, build_tokenizer_5x5
from ..models.jepa_5x5 import PECT_JEPA_5x5
from ..evaluation.downstream_benchmarks import DownstreamBenchmarkSuite


def test_radial_attention_bias():
    num_heads = 4
    bias_mod = RadialAttentionBias(num_heads=num_heads, init_gamma=0.05, init_alpha=0.1)
    
    # 2D distance matrix: [100, 100]
    dist_2d = torch.rand(100, 100) * 10.0
    bias_2d = bias_mod(dist_2d)
    assert bias_2d.shape == (1, num_heads, 100, 100), f"Expected [1, 4, 100, 100], got {bias_2d.shape}"
    # Green's decay must be strictly non-positive (attenuates attention with distance)
    assert (bias_2d <= 0.0).all(), "Attention bias must be non-positive"

    # 3D batched distance matrix: [B, 64, 64]
    B, N_ctx = 2, 64
    dist_3d = torch.rand(B, N_ctx, N_ctx) * 10.0
    bias_3d = bias_mod(dist_3d)
    assert bias_3d.shape == (B, num_heads, N_ctx, N_ctx), f"Expected [{B}, {num_heads}, {N_ctx}, {N_ctx}], got {bias_3d.shape}"
    assert (bias_3d <= 0.0).all()


def test_context_encoder_radial_bias():
    B = 2
    N_ctx = 64
    D = 64
    encoder = ContextEncoder5x5(
        embed_dim=D,
        depth=2,
        num_heads=4,
        use_radial_attention_bias=True,
        spatial_topology="concentric_star",
        star_radii=(1, 3, 7),
    )
    tokens = torch.randn(B, N_ctx, D)
    pos = torch.randn(B, N_ctx, D)
    ctx_idx = torch.randint(0, 100, (B, N_ctx))

    out = encoder(tokens, pos, context_indices=ctx_idx)
    assert out.shape == (B, N_ctx, D), f"Expected [{B}, {N_ctx}, {D}], got {out.shape}"
    assert not torch.isnan(out).any(), "ContextEncoder produced NaNs"


def test_phase_curvature_tokenizer():
    B = 2
    in_channels = 128
    embed_dim = 64
    tokenizer = SpatioSpectralTokenizer5x5(
        in_channels=in_channels,
        embed_dim=embed_dim,
        grid_size=5,
        num_scales=4,
        use_phase_curvature=True,
        spatial_topology="concentric_star",
    )
    x = torch.randn(B, 5, 5, in_channels)
    tokens, pos = tokenizer(x)
    assert tokens.shape == (B, 100, embed_dim), f"Expected [{B}, 100, {embed_dim}], got {tokens.shape}"
    assert pos.shape == (B, 100, embed_dim)
    assert not torch.isnan(tokens).any(), "Tokenizer produced NaNs"


def test_jepa_unified_feature_extraction():
    cfg = Spatiotemporal5x5Config(
        in_channels=128,
        embed_dim=64,
        encoder_depth=2,
        predictor_depth=2,
        use_radial_attention_bias=True,
        use_phase_curvature=True,
        feature_extraction_mode="unified",
        spatial_topology="concentric_star",
    )
    model = PECT_JEPA_5x5(cfg)
    model.eval()

    B = 3
    x = torch.randn(B, 5, 5, 128)
    # 1. Standard center feature: [B, D]
    z_center = model.extract_center_feature(x)
    assert z_center.shape == (B, cfg.embed_dim), f"Expected [{B}, {cfg.embed_dim}], got {z_center.shape}"

    # 2. Unified feature: [B, 2 * D]
    z_unified = model.extract_unified_features(x)
    assert z_unified.shape == (B, 2 * cfg.embed_dim), f"Expected [{B}, {2 * cfg.embed_dim}], got {z_unified.shape}"
    assert not torch.isnan(z_unified).any(), "extract_unified_features produced NaNs"

    # 3. Dynamic dispatch
    z_dispatch = model.extract_features(x)
    assert z_dispatch.shape == (B, 2 * cfg.embed_dim), f"Expected [{B}, {2 * cfg.embed_dim}], got {z_dispatch.shape}"


def test_hurdle_depth_regression_benchmark():
    benchmarks = DownstreamBenchmarkSuite(n_splits=3, random_state=42)
    N = 300
    D = 128
    # 90% sound metal (depth 0.0), 10% defects (depth 0.2 - 1.0)
    features = np.random.randn(N, D).astype(np.float32)
    depth_map = np.zeros(N, dtype=np.float32)
    def_indices = np.random.choice(N, size=30, replace=False)
    depth_map[def_indices] = np.random.uniform(0.2, 1.0, size=30).astype(np.float32)

    res = benchmarks.benchmark_hurdle_depth_regression(features, depth_map)
    assert "gate" in res, "Missing gate results"
    assert "compound_hurdle" in res, "Missing compound hurdle results"
    assert "conditional_defect_sizing" in res, "Missing conditional sizing results"
    assert "standard_unclamped" in res, "Missing standard unclamped results"
    assert 0.0 <= res["gate"]["auc_roc"] <= 1.0
