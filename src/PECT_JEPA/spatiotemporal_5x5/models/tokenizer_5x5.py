"""
Spatial Grid Tokenizer for 5x5 PECT-JEPA.

Linearly projects the C-channel temporal waveform vector at each spatial coordinate (i, j)
into an embedding token of dimension D, combined with a 2D spatial positional embedding.
"""

import math
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn


def build_2d_sinusoidal_pos_embedding(grid_size: int, embed_dim: int) -> torch.Tensor:
    """
    2D Sinusoidal Positional Embedding for a grid_size x grid_size grid.
    Returns: [1, grid_size * grid_size, embed_dim] float32 tensor.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal pos embedding"
    half_d = embed_dim // 2

    # Generate coordinates for (x, y)
    coords_y = torch.arange(grid_size, dtype=torch.float32)
    coords_x = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
    grid_y = grid_y.flatten()  # [N]
    grid_x = grid_x.flatten()  # [N]

    dim_t = torch.arange(half_d // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (2.0 * dim_t / half_d))

    out_x = torch.einsum("m,d->md", grid_x, omega)
    out_y = torch.einsum("m,d->md", grid_y, omega)

    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)  # [N, half_d]
    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)  # [N, half_d]

    pos_2d = torch.cat([pos_x, pos_y], dim=-1).unsqueeze(0)  # [1, N, embed_dim]
    return pos_2d


class SpatialGridTokenizer5x5(nn.Module):
    """
    Unified Tokenizer for 5x5 C-scan grid:
    Input: [B, 5, 5, C] -> Output: tokens [B, 25, D], pos [B, 25, D]
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 128,
        grid_size: int = 5,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size  # 25
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 5, 5, C] tensor (or [5, 5, C])

        Returns:
            tokens:     [B, 25, D]
            pos_expand: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B, self.num_tokens, C)
        tokens = self.drop(self.norm(self.proj(x_flat)))  # [B, 25, D]
        pos = self.pos_embed.expand(B, -1, -1)           # [B, 25, D]
        return tokens, pos


class DualDomainGridTokenizer5x5(nn.Module):
    """
    Spatiotemporal-Spectral Dual-Domain Tokenizer for 5x5 PECT-JEPA.

    Transforms the C-channel temporal waveform vector at each spatial coordinate (i, j)
    into a fused dual-domain token of dimension D (default 128):
      1. Time Domain Branch: Linear projection from in_channels (128) to D // 2 (64).
      2. Spectral Domain Branch: torch.fft.rfft extracts frequency bins (k=1..num_freq_bins).
         Computes normalized phase angle phi_k = torch.angle(X_k) / pi and log-magnitude log(1 + |X_k|),
         projected via Linear to D // 2 (64).
         - Phase lag delta_phi is physically linear in defect depth d and strictly invariant to lift-off distance h.
      3. Dual-Domain Fusion: Concatenates [z_time, z_freq] to dimension D (128),
         followed by LayerNorm and Dropout.
      4. 2D Spatial Positional Embedding: Added to produce tokens ready for Transformer Encoder.

    Input: [B, 5, 5, C] -> Output: tokens [B, 25, D], pos [B, 25, D]
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 128,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        spectral_features: str = "phase_and_mag",
        phase_snr_tapering: bool = True,
        phase_noise_floor: float = 0.05,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size  # 25
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)
        self.spectral_features = spectral_features
        self.phase_snr_tapering = phase_snr_tapering
        self.phase_noise_floor = max(1e-6, float(phase_noise_floor))

        # Dimensions for time and frequency branches
        self.time_embed_dim = embed_dim // 2
        self.freq_embed_dim = embed_dim - self.time_embed_dim

        # 1. Time branch projection
        self.time_proj = nn.Linear(in_channels, self.time_embed_dim)

        # 2. Spectral branch projection
        if spectral_features == "phase_only":
            spectral_dim = self.num_freq_bins
        elif spectral_features == "phase_and_mag":
            spectral_dim = self.num_freq_bins * 2
        else:
            raise ValueError(f"Unknown spectral_features: {spectral_features}")

        self.freq_proj = nn.Linear(spectral_dim, self.freq_embed_dim)

        # 3. Fusion Norm & Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 4. Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 5, 5, C] tensor (or [5, 5, C])

        Returns:
            tokens:     [B, 25, D]
            pos_expand: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B, self.num_tokens, C)

        # Time branch
        z_time = self.time_proj(x_flat)  # [B, 25, D_time]

        # Spectral branch via torch.fft.rfft along temporal dimension
        # Run in float32 for high numerical precision
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B, 25, C//2 + 1] complex
        X_sub = X_fft[..., 1:self.num_freq_bins + 1]  # Exclude DC bin 0

        # Phase angle normalized by pi -> [-1, 1]
        phase = torch.angle(X_sub) / torch.pi
        mag_linear = torch.abs(X_sub)
        mag = torch.log1p(mag_linear)

        # Magnitude-weighted phase tapering: suppresses phase fluctuations in low-energy bins
        if self.phase_snr_tapering:
            snr_weight = torch.tanh(mag_linear / self.phase_noise_floor)
            phase = phase * snr_weight

        if self.spectral_features == "phase_only":
            spectral_feat = phase.to(x_flat.dtype)
        else:
            spectral_feat = torch.cat([phase, mag], dim=-1).to(x_flat.dtype)

        z_freq = self.freq_proj(spectral_feat)  # [B, 25, D_freq]

        # Dual-domain fusion
        z_fused = torch.cat([z_time, z_freq], dim=-1)  # [B, 25, D]
        tokens = self.drop(self.norm(z_fused))
        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class DualDomainAttentionTokenizer5x5(nn.Module):
    """
    Physics-Grounded Dual-Domain Attention Tokenizer for 5x5 PECT-JEPA.

    Transforms the C-channel waveform at each spatial coordinate (i, j)
    into a unified physical token of dimension D via Cross-Domain Multi-Head Attention:
      1. Time Domain Branch:
         - Linear projection: in_channels (128) -> D
         - Branch-wise LayerNorm: guarantees independent variance = 1.0
         - Adds learnable Domain Embedding for temporal domain
      2. Spectral Domain Branch:
         - torch.fft.rfft extracts harmonic frequency bins (k=1..num_freq_bins).
         - Computes normalized phase angle phi_k = angle(X_k)/pi (lift-off invariant)
           and log-magnitude ln(1 + |X_k|).
         - Linear projection: spectral_dim -> D
         - Branch-wise LayerNorm: guarantees independent variance = 1.0
         - Adds learnable Domain Embedding for spectral domain
      3. Cross-Domain Multi-Head Attention Fusion:
         - Stacks [z_time, z_freq] as 2 physical tokens per spatial location: [B*25, 2, D]
         - Multi-Head Self-Attention allows data-dependent cross-routing:
           * Temporal token queries spectral phase to reject lift-off artifacts.
           * Spectral token queries temporal peak dynamics (t_p, t_z).
      4. Spatial Synthesis & Residual Highway:
         - Projects flattened tokens [2*D] -> D.
         - Adds residual shortcut from z_time to preserve raw transient dynamics.
         - Final LayerNorm + Dropout.
      5. 2D Spatial Positional Embedding:
         - Added across 25 spatial grid coordinates.

    Input: [B, 5, 5, C] -> Output: tokens [B, 25, D], pos [B, 25, D]
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        num_heads: int = 4,
        spectral_features: str = "phase_and_mag",
        phase_snr_tapering: bool = True,
        phase_noise_floor: float = 0.05,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size  # 25
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)
        self.spectral_features = spectral_features
        self.phase_snr_tapering = phase_snr_tapering
        self.phase_noise_floor = max(1e-6, float(phase_noise_floor))

        # 1. Temporal branch
        self.time_proj = nn.Linear(in_channels, embed_dim)
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Spectral branch
        if spectral_features == "phase_only":
            spectral_dim = self.num_freq_bins
        elif spectral_features == "phase_and_mag":
            spectral_dim = self.num_freq_bins * 2
        else:
            raise ValueError(f"Unknown spectral_features: {spectral_features}")

        self.freq_proj = nn.Linear(spectral_dim, embed_dim)
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Domain Embeddings
        self.domain_time = nn.Parameter(torch.zeros(1, embed_dim))
        self.domain_freq = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.domain_time, std=0.02)
        nn.init.trunc_normal_(self.domain_freq, std=0.02)

        # 4. Cross-domain Multi-Head Attention Fusion
        self.cross_domain_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 5. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_tokens, C)

        # 1. Time branch with independent LayerNorm
        z_time = self.ln_time(self.time_proj(x_flat)) + self.domain_time  # [B*25, D]

        # 2. Spectral branch with independent LayerNorm
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)[:, 1:self.num_freq_bins + 1]  # Exclude DC
        phase = torch.angle(X_fft) / torch.pi
        mag_linear = torch.abs(X_fft)
        mag = torch.log1p(mag_linear)

        if self.phase_snr_tapering:
            snr_weight = torch.tanh(mag_linear / self.phase_noise_floor)
            phase = phase * snr_weight

        if self.spectral_features == "phase_only":
            spectral_feat = phase.to(x_flat.dtype)
        else:
            spectral_feat = torch.cat([phase, mag], dim=-1).to(x_flat.dtype)

        z_freq = self.ln_freq(self.freq_proj(spectral_feat)) + self.domain_freq  # [B*25, D]

        # 3. Stack into 2 tokens per spatial pixel: [B*25, 2, D]
        tokens_pair = torch.stack([z_time, z_freq], dim=1)  # [B*25, 2, D]
        attn_out, _ = self.cross_domain_attn(
            query=tokens_pair,
            key=tokens_pair,
            value=tokens_pair
        )
        tokens_fused = self.attn_norm(tokens_pair + attn_out)  # [B*25, 2, D]

        # 4. Synthesize to single spatial token with Residual Shortcut
        fused_flat = tokens_fused.reshape(B * self.num_tokens, 2 * self.embed_dim)
        token_spatial = self.fuse_proj(fused_flat) + z_time  # [B*25, D]
        tokens = self.drop(self.norm_out(token_spatial)).reshape(B, self.num_tokens, self.embed_dim)

        # 5. Positional Embedding
        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class DualScaleDiffusionTokenizer5x5(nn.Module):
    """
    Dual-Scale Spatiotemporal-Diffusion Tokenizer for 5x5 PECT-JEPA.

    Transforms the C-channel waveform at each spatial coordinate (i, j)
    into TWO distinct physical tokens of dimension D:
      1. Token Shallow: High-frequency / Surface diffusion regime (skin depth delta small).
         Captures surface interaction, lift-off distance h, and surface defect reflections.
      2. Token Deep: Low-frequency / Bulk diffusion regime (skin depth delta large).
         Captures deep eddy current penetration, wall thickness, and deep defect reflections.

    Key Features:
      - Uses global nn.Linear(in_channels, embed_dim) for temporal projection (preserving global dynamics).
      - Uses Adaptive Energy-Weighted Spectral Embedding to completely eliminate empty noise bins.
      - Total tokens = grid_size * grid_size * 2 = 25 * 2 = 50 tokens.
      - Positional embedding combines 2D spatial coordinate pos_embed with learnable scale embeddings
        (scale_shallow and scale_deep).

    Input: [B, 5, 5, C] -> Output: tokens [B, 50, D], pos [B, 50, D]
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial * 2    # 50
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)

        # 1. Global Temporal Projection (nn.Linear without restrictive inductive bias)
        self.time_proj = nn.Linear(in_channels, embed_dim)
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Spectral Projections for Deep (low freq) and Shallow (high freq)
        self.split_bin = min(4, max(1, self.num_freq_bins // 2))
        self.proj_deep = nn.Linear(2, embed_dim)
        self.proj_shallow = nn.Linear(2, embed_dim)
        self.ln_deep = nn.LayerNorm(embed_dim)
        self.ln_shallow = nn.LayerNorm(embed_dim)

        # 3. Learnable Scale / Diffusion Regime Embeddings
        self.scale_shallow = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_deep = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.scale_shallow, std=0.02)
        nn.init.trunc_normal_(self.scale_deep, std=0.02)

        # 4. Spatial Positional Embedding (2D)
        if pos_embed_type == "learnable_2d":
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial, embed_dim))
            nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("spatial_pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    @staticmethod
    def compute_energy_weighted_phase(x: torch.Tensor, num_bins: int = 14) -> torch.Tensor:
        """
        Computes the global Energy-Weighted Spectral Phase (Phi_EWP) for each pixel.
        Phi_EWP = sum_k w_k * phi_k where w_k = |X_k|^2 / sum(|X_m|^2)
        Returns: [B, 25] in [-1, 1]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        x_flat = x.reshape(B * H * W, C).float()
        X_fft = torch.fft.rfft(x_flat, dim=-1)[:, 1:num_bins + 1]
        mags = torch.abs(X_fft)
        phases = torch.angle(X_fft) / torch.pi
        pwr = mags ** 2
        weights = pwr / (torch.sum(pwr, dim=-1, keepdim=True) + 1e-12)
        ew_phase = torch.sum(weights * phases, dim=-1)
        return ew_phase.reshape(B, H * W).to(x.dtype)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_spatial, C)

        # 1. Temporal branch
        z_time = self.ln_time(self.time_proj(x_flat))  # [B*25, D]

        # 2. Spectral branch via FFT
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)[:, 1:self.num_freq_bins + 1]  # [B*25, K]
        mags = torch.abs(X_fft)  # [B*25, K]
        phases = torch.angle(X_fft) / torch.pi  # [B*25, K] in [-1, 1]
        pwr = mags ** 2  # [B*25, K]

        # Deep bins: 0 .. split_bin
        pwr_deep = pwr[:, :self.split_bin]
        w_deep = pwr_deep / (torch.sum(pwr_deep, dim=-1, keepdim=True) + 1e-12)
        feat_deep = torch.sum(
            w_deep.unsqueeze(-1) * torch.stack([phases[:, :self.split_bin], torch.log1p(mags[:, :self.split_bin])], dim=-1),
            dim=1
        ).to(x_flat.dtype)  # [B*25, 2]
        z_freq_deep = self.proj_deep(feat_deep)  # [B*25, D]

        # Shallow bins: split_bin .. K
        pwr_shallow = pwr[:, self.split_bin:]
        w_shallow = pwr_shallow / (torch.sum(pwr_shallow, dim=-1, keepdim=True) + 1e-12)
        feat_shallow = torch.sum(
            w_shallow.unsqueeze(-1) * torch.stack([phases[:, self.split_bin:], torch.log1p(mags[:, self.split_bin:])], dim=-1),
            dim=1
        ).to(x_flat.dtype)  # [B*25, 2]
        z_freq_shallow = self.proj_shallow(feat_shallow)  # [B*25, D]

        # 3. Fuse into Token Shallow and Token Deep
        t_shallow = self.ln_shallow(z_time + z_freq_shallow)  # [B*25, D]
        t_deep = self.ln_deep(z_time + z_freq_deep)           # [B*25, D]

        t_shallow = t_shallow.reshape(B, self.num_spatial, self.embed_dim)
        t_deep = t_deep.reshape(B, self.num_spatial, self.embed_dim)

        # 4. Interleave into 50 tokens: [shallow_0, deep_0, shallow_1, deep_1, ...]
        tokens = torch.stack([t_shallow, t_deep], dim=2).reshape(B, self.num_tokens, self.embed_dim)
        tokens = self.drop(tokens)

        # 5. Positional Embeddings: Spatial Pos + Scale Embedding
        sp_pos = self.spatial_pos_embed.expand(B, -1, -1)  # [B, 25, D]
        pos_shallow = sp_pos + self.scale_shallow          # [B, 25, D]
        pos_deep = sp_pos + self.scale_deep                # [B, 25, D]
        pos = torch.stack([pos_shallow, pos_deep], dim=2).reshape(B, self.num_tokens, self.embed_dim)
        return tokens, pos


class UncrushedDiffusionTokenizer5x5(nn.Module):
    """
    Uncrushed Harmonic Dispersion Dual-Scale Diffusion Tokenizer for 5x5 PECT-JEPA.
    
    Resolves the 2D Spectral Crushing Bottleneck of EXP-17:
    Instead of summing over harmonics into a single 2D scalar pair [phase_avg, log_mag_avg],
    this tokenizer preserves the FULL harmonic dispersion vectors:
      - Deep Diffusion Regime: low frequencies [0 .. split_bin] (uncrushed phase & log-magnitude).
      - Shallow Diffusion Regime: high frequencies [split_bin .. K] (uncrushed phase & log-magnitude).
    
    Transforms the C-channel waveform at each of the 25 spatial coordinates into TWO
    physically distinct tokens of dimension D:
      Total tokens = 25 probes * 2 modes = 50 tokens.
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
        use_snr_tapering: bool = True,
        phase_noise_floor: float = 0.02,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial * 2    # 50
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)
        self.use_snr_tapering = use_snr_tapering
        self.phase_noise_floor = phase_noise_floor

        # 1. Multi-scale 1D Temporal Conv Filterbank for transient dynamics
        d_sub = embed_dim // 3
        self.conv_short = nn.Conv1d(1, d_sub, kernel_size=5, stride=2, padding=2)
        self.conv_med = nn.Conv1d(1, d_sub, kernel_size=15, stride=2, padding=7)
        self.conv_long = nn.Conv1d(1, embed_dim - 2 * d_sub, kernel_size=31, stride=2, padding=15)
        self.act_time = nn.GELU()
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Uncrushed Spectral Dispersion Branch
        self.split_bin = min(4, max(1, self.num_freq_bins // 2))  # typically 4
        self.deep_dim = self.split_bin * 2                        # 4 bins * 2 = 8 dims
        self.shallow_dim = (self.num_freq_bins - self.split_bin) * 2 # 10 bins * 2 = 20 dims

        self.proj_deep = nn.Linear(self.deep_dim, embed_dim)
        self.proj_shallow = nn.Linear(self.shallow_dim, embed_dim)
        self.ln_deep = nn.LayerNorm(embed_dim)
        self.ln_shallow = nn.LayerNorm(embed_dim)

        # 3. Physics-Gated Dual-Domain Fusion
        self.gate_shallow = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.gate_deep = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.fuse_shallow = nn.Linear(embed_dim, embed_dim)
        self.fuse_deep = nn.Linear(embed_dim, embed_dim)
        self.norm_shallow = nn.LayerNorm(embed_dim)
        self.norm_deep = nn.LayerNorm(embed_dim)

        # 4. Learnable Scale / Diffusion Regime Embeddings
        self.scale_shallow = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_deep = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.scale_shallow, std=0.02)
        nn.init.trunc_normal_(self.scale_deep, std=0.02)

        # 5. Spatial Positional Embedding (2D)
        if pos_embed_type == "learnable_2d":
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial, embed_dim))
            nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("spatial_pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_spatial, C)

        # 1. Multi-scale Temporal Features
        x_1d = x_flat.unsqueeze(1)  # [B*25, 1, C]
        h_s = self.conv_short(x_1d)
        h_m = self.conv_med(x_1d)
        h_l = self.conv_long(x_1d)
        h_cat = self.act_time(torch.cat([h_s, h_m, h_l], dim=1))  # [B*25, D, C//2]
        z_time = self.ln_time(self.time_pool(h_cat).squeeze(-1))   # [B*25, D]

        # 2. Uncrushed Spectral Dispersion Features
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)[:, 1:self.num_freq_bins + 1]  # [B*25, K]
        X_abs = torch.abs(X_fft)
        mags = torch.log1p(X_abs)
        phases_raw = torch.angle(X_fft) / torch.pi  # Normalized to [-1, 1]

        if getattr(self, "use_snr_tapering", True):
            # Dynamic magnitude-aware SNR gate: suppresses undefined phase noise in low-energy frequency bins
            peak_mag = torch.amax(X_abs, dim=-1, keepdim=True).clamp(min=1e-6)
            snr_gate = torch.tanh(X_abs / (getattr(self, "phase_noise_floor", 0.02) * peak_mag))
            phases = phases_raw * snr_gate
        else:
            phases = phases_raw

        # Deep bins: 0 .. split_bin (low-frequency penetration)
        deep_phase = phases[:, :self.split_bin]
        deep_mag = mags[:, :self.split_bin]
        feat_deep = torch.cat([deep_phase, deep_mag], dim=-1).to(x_flat.dtype)  # [B*25, 2 * split_bin]
        z_freq_deep = self.ln_deep(self.proj_deep(feat_deep))                   # [B*25, D]

        # Shallow bins: split_bin .. K (high-frequency surface)
        shallow_phase = phases[:, self.split_bin:]
        shallow_mag = mags[:, self.split_bin:]
        feat_shallow = torch.cat([shallow_phase, shallow_mag], dim=-1).to(x_flat.dtype)  # [B*25, 2 * (K - split_bin)]
        z_freq_shallow = self.ln_shallow(self.proj_shallow(feat_shallow))               # [B*25, D]

        # 3. Physics-Gated Dual-Domain Fusion
        g_s = self.gate_shallow(torch.cat([z_time, z_freq_shallow], dim=-1))
        t_shallow = self.norm_shallow(self.fuse_shallow(g_s * z_time + (1.0 - g_s) * z_freq_shallow))
        t_shallow = t_shallow.reshape(B, self.num_spatial, self.embed_dim)

        g_d = self.gate_deep(torch.cat([z_time, z_freq_deep], dim=-1))
        t_deep = self.norm_deep(self.fuse_deep(g_d * z_time + (1.0 - g_d) * z_freq_deep))
        t_deep = t_deep.reshape(B, self.num_spatial, self.embed_dim)

        # 4. Interleave into 50 tokens: [shallow_0, deep_0, shallow_1, deep_1, ...]
        tokens = torch.stack([t_shallow, t_deep], dim=2).reshape(B, self.num_tokens, self.embed_dim)
        tokens = self.drop(tokens)

        # 5. Positional Embeddings: Spatial Pos + Scale Embedding
        sp_pos = self.spatial_pos_embed.expand(B, -1, -1)  # [B, 25, D]
        pos_shallow = sp_pos + self.scale_shallow          # [B, 25, D]
        pos_deep = sp_pos + self.scale_deep                # [B, 25, D]
        pos = torch.stack([pos_shallow, pos_deep], dim=2).reshape(B, self.num_tokens, self.embed_dim)

        return tokens, pos


class ContinuousSTFTokenizer5x5(nn.Module):
    """
    Continuous Spatiotemporal Filterbank Tokenizer for Multi-Waveform PECT-JEPA.
    
    Transforms the C-channel temporal A-scan at each spatial coordinate (i, j)
    into a rich spatiotemporal-spectral token of dimension D:
      1. Multi-Scale 1D Temporal Convolutions: 3 parallel causal kernels (k=5, 15, 31, stride=2)
         extracting sharp Square rising edges, Gaussian wave packets, and slow diffusive exponential tails.
      2. Full-Spectrum Harmonic Dispersion: Preserves all K frequency bins (normalized phase and log-magnitude)
         without destructive scalar averaging.
      3. Data-Driven Gated Dual-Domain Fusion: Learnable gating balances temporal transients and spectral phase
         dynamically with zero external metadata.
      4. 2D Spatial Positional Embedding: Added across 25 spatial grid coordinates.
    
    Input: [B, 5, 5, C] -> Output: tokens [B, 25, D], pos [B, 25, D]
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size  # 25 spatial tokens
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)

        # 1. Multi-scale 1D Temporal Conv Filterbank
        d_sub = embed_dim // 3
        self.conv_short = nn.Conv1d(1, d_sub, kernel_size=5, stride=2, padding=2)
        self.conv_med = nn.Conv1d(1, d_sub, kernel_size=15, stride=2, padding=7)
        self.conv_long = nn.Conv1d(1, embed_dim - 2 * d_sub, kernel_size=31, stride=2, padding=15)
        self.act_time = nn.GELU()
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Full-Spectrum Harmonic Dispersion
        # 14 bins * 2 (phase and log-magnitude) = 28 features
        self.spectral_dim = self.num_freq_bins * 2
        self.freq_proj = nn.Linear(self.spectral_dim, embed_dim)
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Data-Driven Gated Dual-Domain Fusion
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 4. 2D Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_tokens, C)

        # 1. Temporal Branch via Multi-Scale 1D Convolutions
        x_1d = x_flat.unsqueeze(1)  # [B*25, 1, C]
        h_s = self.conv_short(x_1d)
        h_m = self.conv_med(x_1d)
        h_l = self.conv_long(x_1d)
        h_cat = self.act_time(torch.cat([h_s, h_m, h_l], dim=1))  # [B*25, D, C//2]
        z_time = self.ln_time(self.time_pool(h_cat).squeeze(-1))   # [B*25, D]

        # 2. Spectral Branch via Full Harmonic Dispersion
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)[:, 1:self.num_freq_bins + 1]  # Exclude DC
        phase = torch.angle(X_fft) / torch.pi  # Normalized to [-1, 1]
        mag = torch.log1p(torch.abs(X_fft))
        spectral_feat = torch.cat([phase, mag], dim=-1).to(x_flat.dtype)  # [B*25, 2K]
        z_freq = self.ln_freq(self.freq_proj(spectral_feat))             # [B*25, D]

        # 3. Data-Driven Gated Fusion (Zero Metadata)
        g = self.gate(torch.cat([z_time, z_freq], dim=-1))               # [B*25, D] in [0, 1]
        z_fused = g * z_time + (1.0 - g) * z_freq
        tokens = self.drop(self.norm_out(self.fuse_proj(z_fused))).reshape(B, self.num_tokens, self.embed_dim)

        # 4. Positional Embedding
        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class SpatiotemporalPatchTokenizer5x5(nn.Module):
    """
    Spatiotemporal Patch Tokenizer for PECT-JEPA.

    Transforms the C-channel temporal A-scan at each spatial coordinate (i, j)
    into T_s chronological diffusion stage tokens of dimension D:
      - 25 spatial grid probes * T_s temporal stages (default T_s = 4 -> 100 tokens).
      - Each token z_{(s, tau)} captures the local electromagnetic field state
        at probe location s in diffusion time window tau.
      - Disentangled 2D spatial + 1D temporal positional embeddings.
      - Waveform-agnostic: operates natively across Square, Gaussian, and Chirp
        without arbitrary static frequency cuts or metadata injection.

    Input:  [B, 5, 5, C] -> Output: tokens [B, 100, D], pos [B, 100, D]
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_temporal_stages: int = 4,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_temporal_stages = num_temporal_stages  # 4
        self.num_tokens = self.num_spatial * num_temporal_stages  # 100
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.chunk_size = in_channels // num_temporal_stages  # 32

        # Linear projection per temporal chunk
        self.chunk_proj = nn.Linear(self.chunk_size, embed_dim)
        self.ln_chunk = nn.LayerNorm(embed_dim)

        # 2D Spatial Positional Embedding
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial, 1, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # 1D Temporal Positional Embedding (diffusion stage)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, num_temporal_stages, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        # Reshape to [B, 25, num_temporal_stages, chunk_size]
        x_chunks = x.reshape(B, self.num_spatial, self.num_temporal_stages, self.chunk_size)

        # Project each chunk to embed_dim
        h = self.ln_chunk(self.chunk_proj(x_chunks))  # [B, 25, 4, D]

        # Combine spatial and temporal positional embeddings
        pos = (self.spatial_pos_embed + self.temporal_pos_embed).reshape(1, self.num_tokens, self.embed_dim).expand(B, -1, -1)

        # Flatten tokens to [B, 100, D]
        tokens = self.drop(h.reshape(B, self.num_tokens, self.embed_dim))

        return tokens, pos


class SpatioSpectralTokenizer5x5(nn.Module):
    """
    Physics-Grounded Dual-Domain Spatio-Spectral Skin-Depth Tokenizer for 5x5 PECT-JEPA (EXP-12).

    Transforms the C-channel waveform at each spatial coordinate (i, j) into 4 physical
    skin-depth scale tokens via analytic subband filtering and physics-gated dual-domain fusion:
      - 25 spatial probes x 4 skin-depth scales = 100 physical tokens.

    Skin-Depth Scale Decomposition (Maxwell-Fourier diffusion delta ~ 1/sqrt(f)):
      - Scale 0 (Deepest Subsurface / Back-Wall): Bins 0-2  (DC to ~156 Hz)
      - Scale 1 (Mid-Deep Diffusion):             Bins 3-6  (~234 to ~468 Hz)
      - Scale 2 (Mid-Shallow Diffusion):          Bins 7-14 (~546 to ~1093 Hz)
      - Scale 3 (Near-Surface / Lift-Off):         Bins 15-64 (~1171 to 5000 Hz)

    At each scale k:
      1. Time Domain Branch:
         - Waveform x_k(t) = irfft(X(f) * W_k(f), n=in_channels)
         - 1D Temporal Linear projection: in_channels (128) -> D // 2
         - Independent LayerNorm: ensures unit variance
      2. Spectral Domain Branch:
         - Fourier Phase theta_k = angle(X_k)/pi (Dodd-Deeds lift-off invariant)
         - Log-magnitude ln(1 + |X_k|)
         - Linear projection -> D // 2
         - Independent LayerNorm
      3. Physics-Gated Dual-Domain Fusion:
         - Phase-based gating: Gate_k = sigmoid(Linear(z_freq_k))
         - Gated temporal features: z_time_gated = z_time_k * Gate_k
         - Fused projection with Residual Highway:
           token(i, j, k) = LayerNorm(Linear([z_time_gated, z_freq_k]) + z_time_k)
      4. 3D Positional Embedding:
         - E_pos(i, j, k) = E_spatial(i, j) + E_scale(k)

    Output: tokens [B, 100, D], pos [B, 100, D]
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_scales: int = 4,
        phase_snr_tapering: bool = True,
        phase_noise_floor: float = 0.05,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
        use_phase_curvature: bool = False,
        spatial_topology: str = "concentric_star",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_scales = num_scales  # 4
        self.num_tokens = self.num_spatial * num_scales  # 100
        self.phase_snr_tapering = phase_snr_tapering
        self.phase_noise_floor = max(1e-6, float(phase_noise_floor))
        self.use_phase_curvature = use_phase_curvature
        self.spatial_topology = spatial_topology

        num_fft_bins = in_channels // 2 + 1  # 65 for in_channels=128

        # Define 4 non-overlapping frequency bands covering all 65 bins
        # Scale 0: 0..2 (3 bins) -> Deepest penetration (lowest freq)
        # Scale 1: 3..6 (4 bins) -> Mid-deep
        # Scale 2: 7..14 (8 bins) -> Mid-shallow
        # Scale 3: 15..64 (50 bins) -> Near-surface / lift-off (highest freq)
        self.band_slices = [
            (0, 3),
            (3, 7),
            (7, 15),
            (15, num_fft_bins),
        ]
        self.band_sizes = [end - start for start, end in self.band_slices]

        # Register partition windows W_k(f) on buffer [4, 65]
        windows = torch.zeros(num_scales, num_fft_bins)
        for k, (s, e) in enumerate(self.band_slices):
            windows[k, s:e] = 1.0
        self.register_buffer("windows", windows, persistent=False)

        half_dim = embed_dim // 2

        # 1. Temporal branches per scale
        self.time_proj = nn.ModuleList([
            nn.Linear(in_channels, half_dim) for _ in range(num_scales)
        ])
        self.ln_time = nn.ModuleList([
            nn.LayerNorm(half_dim) for _ in range(num_scales)
        ])

        # 2. Spectral branches per scale (phase + log-mag: 2 * M_k)
        self.freq_proj = nn.ModuleList([
            nn.Linear(2 * size, half_dim) for size in self.band_sizes
        ])
        self.ln_freq = nn.ModuleList([
            nn.LayerNorm(half_dim) for _ in range(num_scales)
        ])

        if use_phase_curvature:
            self.curv_proj = nn.ModuleList([
                nn.Linear(size, half_dim) for size in self.band_sizes
            ])
            self.curv_ln = nn.ModuleList([
                nn.LayerNorm(half_dim) for _ in range(num_scales)
            ])

        # 3. Physics-Gated Dual-Domain Fusion per scale
        self.gate_proj = nn.ModuleList([
            nn.Linear(half_dim, half_dim) for _ in range(num_scales)
        ])
        self.fuse_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_scales)
        ])
        self.time_res = nn.ModuleList([
            nn.Linear(half_dim, embed_dim) for _ in range(num_scales)
        ])
        self.norm_out = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_scales)
        ])

        # Domain embeddings
        self.domain_time = nn.Parameter(torch.zeros(1, 1, half_dim))
        self.domain_freq = nn.Parameter(torch.zeros(1, 1, half_dim))
        nn.init.trunc_normal_(self.domain_time, std=0.02)
        nn.init.trunc_normal_(self.domain_freq, std=0.02)

        # 4. Separable 3D Positional Embedding: E_spatial(25) + E_scale(4)
        self.pos_spatial = nn.Parameter(torch.zeros(1, self.num_spatial, 1, embed_dim))
        self.pos_scale = nn.Parameter(torch.zeros(1, 1, self.num_scales, embed_dim))
        nn.init.trunc_normal_(self.pos_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_scale, std=0.02)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_sp = x.reshape(B, self.num_spatial, C)
        x_fp32 = x_sp.float()

        # Compute full complex FFT: [B, 25, 65]
        X_fft = torch.fft.rfft(x_fp32, dim=-1)
        self._last_fft = X_fft

        # Vectorized analytic temporal subbands across all scales in 1 single CUDA kernel:
        # [4, 1, 1, 65] * [1, B, 25, 65] -> [4, B, 25, 65] -> irfft -> [4, B, 25, in_channels]
        windows_4d = self.windows.view(self.num_scales, 1, 1, -1).to(x_fp32.device)
        X_subbands_fft = X_fft.unsqueeze(0) * windows_4d
        x_subbands = torch.fft.irfft(X_subbands_fft, n=self.in_channels, dim=-1).to(x.dtype)

        scale_tokens_list = []
        for k in range(self.num_scales):
            s_idx, e_idx = self.band_slices[k]

            # 1. Analytic temporal subband
            x_k = x_subbands[k]

            # Temporal feature extraction
            z_time = self.ln_time[k](self.time_proj[k](x_k)) + self.domain_time  # [B, 25, D//2]

            # 2. Spectral feature extraction (Fourier phase + log-magnitude for band k)
            X_band = X_fft[:, :, s_idx:e_idx]  # [B, 25, M_k]
            phase_k = torch.angle(X_band) / torch.pi
            mag_linear_k = torch.abs(X_band)
            mag_k = torch.log1p(mag_linear_k)

            if self.phase_snr_tapering:
                snr_weight = torch.tanh(mag_linear_k / self.phase_noise_floor)
                phase_k = phase_k * snr_weight

            spectral_feat_k = torch.cat([phase_k, mag_k], dim=-1).to(x.dtype)  # [B, 25, 2*M_k]
            z_freq = self.ln_freq[k](self.freq_proj[k](spectral_feat_k)) + self.domain_freq  # [B, 25, D//2]

            # Harmonic Radial Phase Curvature: kappa_theta = d^2 theta / dr^2
            if self.use_phase_curvature and self.spatial_topology in ("concentric_star", "star", "octagram"):
                # Probes 1..8: Ring 1 (1mm), 9..16: Ring 2 (3mm), 17..24: Ring 3 (7mm)
                r1_p = phase_k[:, 1:9, :].mean(dim=1)
                r2_p = phase_k[:, 9:17, :].mean(dim=1)
                r3_p = phase_k[:, 17:25, :].mean(dim=1)
                # Radial phase differences: dr12=2mm, dr23=4mm
                d12 = (r2_p - r1_p) / 2.0
                d23 = (r3_p - r2_p) / 4.0
                curv_k = (d23 - d12) / 3.0  # [B, M_k]
                z_curv = self.curv_ln[k](self.curv_proj[k](curv_k)).unsqueeze(1)  # [B, 1, D//2]
                z_freq = z_freq + z_curv

            # 3. Physics-Gated Dual-Domain Fusion
            gate = torch.sigmoid(self.gate_proj[k](z_freq))  # [B, 25, D//2]
            z_time_gated = z_time * gate

            z_cat = torch.cat([z_time_gated, z_freq], dim=-1)  # [B, 25, D]
            z_fused = self.fuse_proj[k](z_cat) + self.time_res[k](z_time)  # [B, 25, D]
            token_k = self.norm_out[k](z_fused)  # [B, 25, D]

            scale_tokens_list.append(token_k)

        # Stack into [B, 25, 4, D] -> reshape to [B, 100, D]
        # Order: token for probe s, scale k is at index s*4 + k
        tokens_grid = torch.stack(scale_tokens_list, dim=2)  # [B, 25, 4, D]
        tokens = self.drop(tokens_grid.reshape(B, self.num_tokens, self.embed_dim))

        # 4. Positional Embedding: [B, 25, 4, D] -> [B, 100, D]
        pos = (self.pos_spatial + self.pos_scale).reshape(1, self.num_tokens, self.embed_dim).expand(B, -1, -1)

        return tokens, pos


class ContinuousFieldTokenizer5x5(nn.Module):
    """
    Continuous Dual-Domain Field Tokenizer for 5x5 PECT-JEPA (EXP-22).
    
    Waveform-Agnostic, Continuous 25-Token Architecture:
      - Strictly 1 continuous token per spatial probe on the Concentric Star or 5x5 grid (25 tokens total).
      - Zero temporal slicing (tau_0..tau_3) and zero depth thresholding (shallow/deep at 1.0mm).
      - Multi-scale 1D Conv filterbank for continuous transient dynamics (arrival time t_p, rise slope).
      - Full uncrushed 14-harmonic Fourier dispersion (phase & log-magnitude).
      - Dodd-Deeds lift-off invariance via Fourier phase with magnitude-weighted SNR tapering.
      - Physics-gated cross-domain dynamic fusion with residual highway from time branch.
      - 2D spatial positional embedding.
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
        use_snr_tapering: bool = True,
        phase_noise_floor: float = 0.02,
        temporal_ac_coupling: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial        # Exactly 25 tokens
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)
        self.use_snr_tapering = use_snr_tapering
        self.phase_noise_floor = phase_noise_floor
        self.temporal_ac_coupling = temporal_ac_coupling

        # 1. Multi-scale 1D Temporal Conv Filterbank for transient dynamics
        d_sub = embed_dim // 3
        self.conv_short = nn.Conv1d(1, d_sub, kernel_size=5, stride=2, padding=2)
        self.conv_med = nn.Conv1d(1, d_sub, kernel_size=15, stride=2, padding=7)
        self.conv_long = nn.Conv1d(1, embed_dim - 2 * d_sub, kernel_size=31, stride=2, padding=15)
        self.act_time = nn.GELU()
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Uncrushed Spectral Dispersion Branch (14 frequency bins: phase + log-mag)
        self.spectral_dim = self.num_freq_bins * 2
        self.proj_freq = nn.Linear(self.spectral_dim, embed_dim)
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Physics-Gated Dual-Domain Dynamic Fusion
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_tokens, C)
        if self.temporal_ac_coupling:
            x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)

        # 1. Multi-scale Temporal Features
        x_1d = x_flat.unsqueeze(1)  # [B*25, 1, C]
        h_short = self.time_pool(self.act_time(self.conv_short(x_1d))).squeeze(-1)
        h_med = self.time_pool(self.act_time(self.conv_med(x_1d))).squeeze(-1)
        h_long = self.time_pool(self.act_time(self.conv_long(x_1d))).squeeze(-1)
        h_time = torch.cat([h_short, h_med, h_long], dim=-1)  # [B*25, D]
        z_time = self.ln_time(h_time)

        # 2. Uncrushed Spectral Dispersion
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)
        self._last_fft = X_fft
        X_sub = X_fft[:, 1:self.num_freq_bins + 1]  # Exclude DC

        phase = torch.angle(X_sub) / torch.pi
        mag_linear = torch.abs(X_sub)
        mag = torch.log1p(mag_linear)

        if self.use_snr_tapering:
            snr_weight = torch.tanh(mag_linear / self.phase_noise_floor)
            phase = phase * snr_weight

        spectral_feat = torch.cat([phase, mag], dim=-1).to(x.dtype)  # [B*25, 2*num_freq_bins]
        z_freq = self.ln_freq(self.proj_freq(spectral_feat))        # [B*25, D]

        # 3. Physics-Gated Dual-Domain Fusion
        gate = self.gate_proj(torch.cat([z_time, z_freq], dim=-1))   # [B*25, D]
        z_fused = self.fuse_proj(z_time * gate + z_freq * (1.0 - gate)) + z_time  # [B*25, D]
        tokens = self.drop(self.norm_out(z_fused)).reshape(B, self.num_tokens, self.embed_dim)

        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos



class ContinuousLinearFieldTokenizer5x5(nn.Module):
    """
    Continuous Linear Dual-Domain Field Tokenizer for 5x5 PECT-JEPA (EXP-28).

    Waveform-Agnostic, Continuous 25-Token Architecture:
      - Strictly 1 continuous token per spatial probe on the Concentric Star or 5x5 grid (25 tokens total).
      - Zero temporal slicing (tau_0..tau_3) and zero temporal pooling (AdaptiveAvgPool1d(1) eliminated).
      - Continuous 1D Learnable Projection: directly projects the full in_channels waveform vector
        into latent space via an MLP with LayerNorm and GELU, preserving peak arrival delay t_p
        and LOI point with full end-to-end gradient sensitivity.
      - Full uncrushed 14-harmonic Fourier dispersion (phase & log-magnitude).
      - Dodd-Deeds lift-off invariance via Fourier phase with magnitude-weighted SNR tanh tapering.
      - Direct orthogonal dual-domain projection (NO zero-sum convex gating), preserving carrier
        and dispersion at full rank.
      - 2D spatial positional embedding.
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
        use_snr_tapering: bool = True,
        phase_noise_floor: float = 0.02,
        temporal_ac_coupling: bool = False,
        adaptive_phase_floor: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial        # Exactly 25 tokens
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)
        self.use_snr_tapering = use_snr_tapering
        self.phase_noise_floor = phase_noise_floor
        self.temporal_ac_coupling = temporal_ac_coupling
        self.adaptive_phase_floor = adaptive_phase_floor

        # 1. Continuous 1D Temporal Projection (NO temporal pooling!)
        self.time_proj = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Uncrushed Spectral Dispersion Branch (14 frequency bins: phase + log-mag)
        self.spectral_dim = self.num_freq_bins * 2
        self.proj_freq = nn.Sequential(
            nn.Linear(self.spectral_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Direct Orthogonal Dual-Domain Fusion (No zero-sum convex gate)
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_tokens, C)
        if self.temporal_ac_coupling:
            x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)

        # 1. Continuous Temporal Projection preserving peak arrival t_p
        z_time = self.ln_time(self.time_proj(x_flat))  # [B*25, D]

        # 2. Uncrushed Spectral Dispersion
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)
        self._last_fft = X_fft
        X_sub = X_fft[:, 1:self.num_freq_bins + 1]  # Exclude DC

        phase = torch.angle(X_sub) / torch.pi
        mag_linear = torch.abs(X_sub)
        mag = torch.log1p(mag_linear)

        if self.use_snr_tapering:
            if getattr(self, "adaptive_phase_floor", False):
                # Scale noise floor adaptively by mean harmonic magnitude per probe to prevent Gaussian starvation
                local_floor = self.phase_noise_floor * (mag_linear.mean(dim=-1, keepdim=True) + 1e-6)
                snr_weight = torch.tanh(mag_linear / (local_floor + 1e-8))
            else:
                snr_weight = torch.tanh(mag_linear / self.phase_noise_floor)
            phase = phase * snr_weight

        spectral_feat = torch.cat([phase, mag], dim=-1).to(x.dtype)  # [B*25, 2*num_freq_bins]
        z_freq = self.ln_freq(self.proj_freq(spectral_feat))        # [B*25, D]

        # 3. Direct Orthogonal Dual-Domain Fusion with Residual Highway
        z_cat = torch.cat([z_time, z_freq], dim=-1)                 # [B*25, 2*D]
        z_fused = self.fuse_proj(z_cat) + z_time                     # [B*25, D]
        tokens = self.drop(self.norm_out(z_fused)).reshape(B, self.num_tokens, self.embed_dim)

        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class EnergyAdaptiveDualDomainTokenizer5x5(nn.Module):
    """
    Energy-Adaptive Dual-Domain Tokenizer for 5x5 PECT-JEPA (Stage 2 Re-foundation, EXP-35).

    Resolves the 2 core failure modes identified in ContinuousLinearFieldTokenizer5x5:
    1. Spectral Out-of-Band Noise Ingestion: Bins with near-zero excitation power (e.g. 9-12 out of 14 bins
       in Square and Gaussian pulses) previously injected random uniform phase noise into z_freq.
       Solution: Energy Saliency Gating s(f) = |X(f)|^2 / sum(|X|^2) smoothly suppresses inactive bins to 0
       while preserving Dodd-Deeds lift-off invariant phase theta(f) in active excitation bands.
       Continuous spectral moments (centroid f_c, spread sigma_f) are added to capture dispersion bandwidth.
    2. Time-Branch Dominance (+ z_time bias): The legacy fusion added an asymmetric + z_time skip, starving
       z_freq gradients (2.91x ratio) and forcing the latent space into waveform-locked chronological time coordinates.
       Solution: Symmetric Balanced Residual Fusion: z_fused = W_fuse [z_time, z_freq] + 0.5 * (z_time + z_freq),
       achieving a 1.07:1 gradient balance between physical time dynamics and spectral dispersion.
    3. Waveform-Agnostic Shape Normalization: Waveform instance normalization removes pulse morphology memorization,
       while concatenating 3 physical transient invariants: peak arrival time t_p, peak-to-peak voltage V_pp,
       and transient energy integral E_time.
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial        # Exactly 25 tokens
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)

        # 1. Temporal Branch: Instance-Normalized Waveform + Physical Transient Invariants (t_p, V_pp, E_time)
        self.time_in_dim = in_channels + 3
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Spectral Branch: Energy-Saliency Weighted Phase + Log-Mag + Moments
        # Features: [s * theta (K), s * mag (K), s (K), f_centroid (1), f_spread (1)] = 3*K + 2
        self.spectral_dim = self.num_freq_bins * 3 + 2
        self.proj_freq = nn.Sequential(
            nn.Linear(self.spectral_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Symmetric Dual-Domain Balanced Highway Fusion
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 5, 5, C] tensor (or [5, 5, C])
        Returns:
            tokens:     [B, 25, D]
            pos_expand: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B * self.num_tokens, C)

        # --- A. Time Domain Feature Extraction ---
        # 1. Physical transient anchors:
        # Peak arrival index normalized to [0, 1]
        t_p = torch.argmax(torch.abs(x_flat), dim=-1, keepdim=True).float() / float(C)
        v_max, _ = torch.max(x_flat, dim=-1, keepdim=True)
        v_min, _ = torch.min(x_flat, dim=-1, keepdim=True)
        v_pp = v_max - v_min
        e_time = torch.mean(torch.abs(x_flat), dim=-1, keepdim=True)

        # 2. Instance-normalized waveform (shape-invariant transient dynamic):
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        x_std = x_flat.std(dim=-1, keepdim=True)
        x_norm = (x_flat - x_mean) / (x_std + 1e-6)

        phi_time = torch.cat([x_norm, t_p, v_pp, e_time], dim=-1).to(x.dtype)
        z_time = self.ln_time(self.time_proj(phi_time))  # [B*25, D]

        # --- B. Frequency Domain Feature Extraction ---
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)
        self._last_fft = X_fft
        X_sub = X_fft[:, 1:self.num_freq_bins + 1]  # Exclude DC, [B*25, K]

        # Relative Power Spectral Density:
        P = torch.abs(X_sub) ** 2
        P_tot = P.sum(dim=-1, keepdim=True) + 1e-8
        s = P / P_tot  # [B*25, K] in [0, 1]

        # Dodd-Deeds Lift-off Invariant Phase:
        theta = torch.angle(X_sub) / torch.pi  # [-1, 1]
        mag = torch.log1p(torch.abs(X_sub))

        # Energy-Saliency Gating:
        theta_gated = s * theta
        mag_gated = s * mag

        # Continuous Spectral Moments:
        bin_indices = torch.arange(1, self.num_freq_bins + 1, device=x.device, dtype=torch.float32).unsqueeze(0)
        f_centroid = (bin_indices * s).sum(dim=-1, keepdim=True) / float(self.num_freq_bins)
        var_f = ((bin_indices / float(self.num_freq_bins) - f_centroid) ** 2 * s).sum(dim=-1, keepdim=True)
        f_spread = torch.sqrt(torch.clamp(var_f, min=0.0) + 1e-8)

        phi_freq = torch.cat([theta_gated, mag_gated, s, f_centroid, f_spread], dim=-1).to(x.dtype)
        z_freq = self.ln_freq(self.proj_freq(phi_freq))  # [B*25, D]

        # --- C. Symmetric Balanced Dual-Domain Fusion ---
        z_cat = torch.cat([z_time, z_freq], dim=-1)
        # Balanced residual highway: 50% time + 50% freq (NO asymmetric + z_time bias)
        z_fused = self.fuse_proj(z_cat) + 0.5 * (z_time + z_freq)
        tokens = self.drop(self.norm_out(z_fused)).reshape(B, self.num_tokens, self.embed_dim)

        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class ImpedanceDeconvolutionTokenizer5x5(nn.Module):
    """
    Self-Calibrated Impedance Deconvolution Tokenizer for 5x5 PECT-JEPA (EXP-38).

    Eliminates sensor transfer function T(f) and excitation waveform I(f) variations
    by computing relative spectral impedance deconvolution against the local spatial
    sound-metal reference carrier in both the Fourier spectral domain and transient time domain:

    1. Spectral Impedance Deconvolution:
       V_p(f) = I(f) * T_sensor(f) * Z_p(f)
       V_ref(f) = I(f) * T_sensor(f) * Z_sound(f)
       Deconvolution: V_p(f) / V_ref(f) = Z_p(f) / Z_sound(f)
       - Relative Phase Shift: Delta theta_p(f) = arg(V_p(f) * conj(V_ref(f))) / pi
         100% independent of arg(I(f)) and arg(T_sensor(f)).
       - Relative Amplitude Modulation: R_p(f) = (|V_p(f)| - |V_ref(f)|) / (|V_ref(f)| + eps)
         100% independent of |I(f)| and |T_sensor(f)|.

    2. Relative Transient Dynamics:
       Delta x_p(t) = (x_p(t) - x_ref(t)) / (||x_ref|| + eps).

    3. Waveform-Agnostic Energy Saliency Gating:
       s(f) = |V_ref(f)|^2 / sum(|V_ref|^2) suppresses inactive out-of-band spectral noise.

    4. Symmetric Balanced Dual-Domain Fusion:
       z_fused = W_fuse [z_time, z_freq] + 0.5 * (z_time + z_freq).
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial        # Exactly 25 tokens
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)

        # 1. Temporal Branch: Local Reference-Normalized Perturbation + Invariant Transient Features
        self.time_in_dim = in_channels + 3
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Spectral Branch: Deconvolved Relative Phase + Relative Mag + Saliency + Moments
        self.spectral_dim = self.num_freq_bins * 3 + 2
        self.proj_freq = nn.Sequential(
            nn.Linear(self.spectral_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Symmetric Dual-Domain Balanced Highway Fusion
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 5, 5, C] tensor (or [5, 5, C])
        Returns:
            tokens:     [B, 25, D]
            pos_expand: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_grid = x.reshape(B, self.num_spatial, C)

        # Local sound-metal spatial median across the 25 probes
        x_ref = torch.median(x_grid, dim=1, keepdim=True).values  # [B, 1, C]
        e_ref = torch.sqrt(torch.mean(x_ref ** 2, dim=-1, keepdim=True) + 1e-6)  # [B, 1, 1]

        # --- A. Time Domain: Normalized Transient Perturbation ---
        delta_x = (x_grid - x_ref) / e_ref  # [B, 25, C]
        t_p = torch.argmax(torch.abs(x_grid), dim=-1, keepdim=True).float() / float(C)  # [B, 25, 1]
        v_pp = (x_grid.max(dim=-1, keepdim=True).values - x_grid.min(dim=-1, keepdim=True).values) / e_ref  # [B, 25, 1]
        e_pert = torch.mean(torch.abs(x_grid - x_ref), dim=-1, keepdim=True) / e_ref  # [B, 25, 1]

        phi_time = torch.cat([delta_x, t_p, v_pp, e_pert], dim=-1).to(x.dtype)
        z_time = self.ln_time(self.time_proj(phi_time.reshape(B * self.num_spatial, -1)))  # [B*25, D]

        # --- B. Frequency Domain: Dodd-Deeds Relative Impedance Deconvolution ---
        x_fp32 = x_grid.float()
        x_ref_fp32 = x_ref.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)      # [B, 25, C//2 + 1]
        X_ref = torch.fft.rfft(x_ref_fp32, dim=-1)  # [B, 1, C//2 + 1]

        self._last_fft = X_fft.reshape(B * self.num_spatial, -1)

        X_sub = X_fft[:, :, 1:self.num_freq_bins + 1]          # [B, 25, K]
        X_ref_sub = X_ref[:, :, 1:self.num_freq_bins + 1]      # [B, 1, K]

        # Relative Phase Shift via conjugate multiplication (cancels I(f) and T_sensor(f)):
        prod = X_sub * torch.conj(X_ref_sub)
        delta_theta = torch.angle(prod) / torch.pi  # [B, 25, K] in [-1, 1]
        self._last_phase = delta_theta.reshape(B * self.num_spatial, -1)

        # Relative Amplitude Variation (cancels |I(f)| and |T_sensor(f)|):
        mag_probe = torch.abs(X_sub)
        mag_ref = torch.abs(X_ref_sub)
        rel_mag = torch.clamp((mag_probe - mag_ref) / (mag_ref + 1e-6), min=-2.0, max=5.0)  # [B, 25, K]

        # Reference Energy Saliency Gating:
        P_ref = mag_ref ** 2
        s = P_ref / (P_ref.sum(dim=-1, keepdim=True) + 1e-8)  # [B, 1, K]

        delta_theta_gated = s * delta_theta  # [B, 25, K]
        rel_mag_gated = s * rel_mag          # [B, 25, K]

        # Continuous Spectral Moments:
        bin_indices = torch.arange(1, self.num_freq_bins + 1, device=x.device, dtype=torch.float32).reshape(1, 1, -1)
        f_centroid = (bin_indices * s).sum(dim=-1, keepdim=True) / float(self.num_freq_bins)  # [B, 1, 1]
        var_f = ((bin_indices / float(self.num_freq_bins) - f_centroid) ** 2 * s).sum(dim=-1, keepdim=True)
        f_spread = torch.sqrt(torch.clamp(var_f, min=0.0) + 1e-8)  # [B, 1, 1]

        s_exp = s.expand(B, self.num_spatial, self.num_freq_bins)
        fc_exp = f_centroid.expand(B, self.num_spatial, 1)
        fs_exp = f_spread.expand(B, self.num_spatial, 1)

        phi_freq = torch.cat([delta_theta_gated, rel_mag_gated, s_exp, fc_exp, fs_exp], dim=-1).to(x.dtype)
        z_freq = self.ln_freq(self.proj_freq(phi_freq.reshape(B * self.num_spatial, -1)))  # [B*25, D]

        # --- C. Symmetric Balanced Dual-Domain Fusion ---
        z_cat = torch.cat([z_time, z_freq], dim=-1)
        z_fused = self.fuse_proj(z_cat) + 0.5 * (z_time + z_freq)
        tokens = self.drop(self.norm_out(z_fused)).reshape(B, self.num_tokens, self.embed_dim)

        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class EnergyStabilizedDeconvTokenizer5x5(nn.Module):
    """
    Energy-Stabilized Dodd-Deeds Impedance Deconvolution Tokenizer for PECT-JEPA (EXP-40).

    Physical & Algorithmic Invariants:
      1. Tikhonov Frequency Regularization Floor:
         Bounds relative surface impedance deconvolution Delta Z(f) against low-energy
         spectral tails outside the excitation bandwidth (especially for 500-1500Hz Chirp sweeps),
         using an adaptive 5% peak excitation floor:
           eps_f = 0.05 * max_k |X_ref(k)|
           Delta Z(f) = (X_probe(f) - X_ref(f)) / sqrt(|X_ref(f)|^2 + eps_f^2)
      2. Dual-Polarity Dispersion Representation:
         Encodes BOTH signed phase delay Delta theta(f) and sign-invariant phase magnitude |Delta theta(f)|,
         as well as signed relative magnitude Delta Z(f) and relative perturbation magnitude |Delta Z(f)|.
         This guarantees that coil self-resonance polarity flips do not invert downstream anomaly detection.
      3. Dual-Stream Transient Perturbation:
         Encodes normalized signed transient perturbation Delta x(t) and absolute transient envelope |Delta x(t)|,
         preserving both fine directional derivatives for depth sizing and invariant perturbation energy for detection.
      4. Reference Energy Saliency Gating:
         Harmonic weighting s(f) concentrates gradient updates on active excitation frequencies.
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
        tikhonov_gamma: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size
        self.num_tokens = self.num_spatial
        self.num_freq_bins = num_freq_bins
        self.tikhonov_gamma = tikhonov_gamma

        # 1. Temporal Branch: Signed + Absolute Transient Perturbation + Invariant Features
        self.time_in_dim = in_channels * 2 + 3
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Spectral Branch: Signed + Absolute Phase, Signed + Absolute Mag, Saliency, Moments
        # 5 * K (delta_theta, abs_delta_theta, rel_mag, abs_rel_mag, s) + 2 moments (centroid, spread)
        self.spectral_dim = self.num_freq_bins * 5 + 2
        self.proj_freq = nn.Sequential(
            nn.Linear(self.spectral_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Symmetric Dual-Domain Balanced Highway Fusion
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_grid = x.reshape(B, self.num_spatial, C)

        # Local sound-metal spatial reference (median across 25 probes)
        x_ref = torch.median(x_grid, dim=1, keepdim=True).values  # [B, 1, C]
        e_ref = torch.sqrt(torch.mean(x_ref ** 2, dim=-1, keepdim=True) + 1e-6)  # [B, 1, 1]

        # --- A. Time Domain: Dual-Stream Transient Perturbation ---
        diff_x = (x_grid - x_ref) / e_ref       # [B, 25, C] (signed)
        abs_diff_x = torch.abs(diff_x)          # [B, 25, C] (sign-invariant)
        t_p = torch.argmax(torch.abs(x_grid), dim=-1, keepdim=True).float() / float(C)  # [B, 25, 1]
        v_pp = (x_grid.max(dim=-1, keepdim=True).values - x_grid.min(dim=-1, keepdim=True).values) / e_ref  # [B, 25, 1]
        e_pert = torch.mean(abs_diff_x, dim=-1, keepdim=True)  # [B, 25, 1]

        phi_time = torch.cat([diff_x, abs_diff_x, t_p, v_pp, e_pert], dim=-1).to(x.dtype)
        z_time = self.ln_time(self.time_proj(phi_time.reshape(B * self.num_spatial, -1)))  # [B*25, D]

        # --- B. Frequency Domain: Tikhonov Dodd-Deeds Deconvolution ---
        x_fp32 = x_grid.float()
        x_ref_fp32 = x_ref.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)      # [B, 25, C//2 + 1]
        X_ref = torch.fft.rfft(x_ref_fp32, dim=-1)  # [B, 1, C//2 + 1]

        self._last_fft = X_fft.reshape(B * self.num_spatial, -1)

        X_sub = X_fft[:, :, 1:self.num_freq_bins + 1]          # [B, 25, K]
        X_ref_sub = X_ref[:, :, 1:self.num_freq_bins + 1]      # [B, 1, K]

        # Tikhonov regularization floor based on peak excitation energy:
        mag_ref = torch.abs(X_ref_sub)                         # [B, 1, K]
        peak_ref = torch.amax(mag_ref, dim=-1, keepdim=True).clamp(min=1e-6)  # [B, 1, 1]
        eps_tikhonov = self.tikhonov_gamma * peak_ref          # [B, 1, 1]
        denom = torch.sqrt(mag_ref ** 2 + eps_tikhonov ** 2)   # [B, 1, K]

        # 1. Relative Phase Shift (cancels I(f) and T_sensor(f)):
        prod = X_sub * torch.conj(X_ref_sub)
        delta_theta = torch.angle(prod) / torch.pi             # [B, 25, K] in [-1, 1] (signed)
        abs_delta_theta = torch.abs(delta_theta)               # [B, 25, K] in [0, 1] (sign-invariant)
        self._last_phase = delta_theta.reshape(B * self.num_spatial, -1)

        # 2. Relative Amplitude Variation (with Tikhonov denominator):
        mag_probe = torch.abs(X_sub)                           # [B, 25, K]
        rel_mag = torch.clamp((mag_probe - mag_ref) / denom, min=-3.0, max=5.0)  # [B, 25, K] (signed)
        abs_rel_mag = torch.abs(rel_mag)                       # [B, 25, K] (sign-invariant)

        # 3. Energy Saliency Gating:
        P_ref = mag_ref ** 2
        s = P_ref / (P_ref.sum(dim=-1, keepdim=True) + 1e-8)   # [B, 1, K]

        delta_theta_gated = s * delta_theta                    # [B, 25, K]
        abs_theta_gated = s * abs_delta_theta                  # [B, 25, K]
        rel_mag_gated = s * rel_mag                            # [B, 25, K]
        abs_mag_gated = s * abs_rel_mag                        # [B, 25, K]

        # 4. Continuous Spectral Moments:
        bin_indices = torch.arange(1, self.num_freq_bins + 1, device=x.device, dtype=torch.float32).reshape(1, 1, -1)
        f_centroid = (bin_indices * s).sum(dim=-1, keepdim=True) / float(self.num_freq_bins)  # [B, 1, 1]
        var_f = ((bin_indices / float(self.num_freq_bins) - f_centroid) ** 2 * s).sum(dim=-1, keepdim=True)
        f_spread = torch.sqrt(torch.clamp(var_f, min=0.0) + 1e-8)  # [B, 1, 1]

        s_exp = s.expand(B, self.num_spatial, self.num_freq_bins)
        fc_exp = f_centroid.expand(B, self.num_spatial, 1)
        fs_exp = f_spread.expand(B, self.num_spatial, 1)

        phi_freq = torch.cat([delta_theta_gated, abs_theta_gated, rel_mag_gated, abs_mag_gated, s_exp, fc_exp, fs_exp], dim=-1).to(x.dtype)
        z_freq = self.ln_freq(self.proj_freq(phi_freq.reshape(B * self.num_spatial, -1)))  # [B*25, D]

        # --- C. Symmetric Balanced Dual-Domain Fusion ---
        z_cat = torch.cat([z_time, z_freq], dim=-1)
        z_fused = self.fuse_proj(z_cat) + 0.5 * (z_time + z_freq)
        tokens = self.drop(self.norm_out(z_fused)).reshape(B, self.num_tokens, self.embed_dim)

        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


class AutonomousDualDomainTokenizer5x5(nn.Module):
    """
    Autonomous Dual-Domain Tokenizer for 5x5 PECT-JEPA (Phase 1 Pre-registration).

    Physical & Algorithmic Invariants:
      1. Strictly Autonomous Probe Tokenization (Zero Local Differential Probing):
         Processes all 25 spatial probes independently without subtracting the local 25-probe median.
         Completely eliminates flat corrosion blindness on 40x40mm flaws where all probes sit on defect.
         Fully adheres to RULE 3 (Strict Prohibition of Software Differential Probing).
      2. Dual-Domain Dynamics:
         - Time Domain: Instance-normalized shape-invariant transient dynamic x_norm,
           absolute envelope |x_norm|, and 3 physical transient invariants (t_p, V_pp, E_time).
         - Frequency Domain: Uncrushed Fourier harmonic decomposition (14 bins),
           Dodd-Deeds lift-off invariant phase theta(f), dual-polarity phase |theta(f)|,
           energy saliency gating s(f) concentrating updates on active excitation frequencies,
           and continuous spectral moments (f_centroid, f_spread).
      3. Symmetric Balanced Dual-Domain Fusion:
         Balanced residual highway: z_fused = W_fuse [z_time, z_freq] + 0.5 * (z_time + z_freq),
         guaranteeing 1:1 gradient flow between transient dynamics and spectral dispersion.
      4. Decoupled Scan-Level Calibration Support:
         Optionally accepts an external Scan-Level Sound-Metal Reference (x_sound) from M1 calibration
         for global Dodd-Deeds impedance deconvolution, strictly decoupling calibration from patch tokenization.
    """
    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 64,
        grid_size: int = 5,
        num_freq_bins: int = 14,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0,
        tikhonov_gamma: float = 0.05,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size  # 25
        self.num_tokens = self.num_spatial        # Exactly 25 tokens
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_freq_bins = min(num_freq_bins, in_channels // 2)
        self.tikhonov_gamma = float(tikhonov_gamma)

        # 1. Temporal Branch: Instance-Normalized Waveform + Absolute Envelope + Physical Transient Invariants (t_p, V_pp, E_time)
        # Dimensions: in_channels (x_norm) + in_channels (|x_norm|) + 3 (t_p, V_pp, E_time)
        self.time_in_dim = in_channels * 2 + 3
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_time = nn.LayerNorm(embed_dim)

        # 2. Spectral Branch: Dual-Polarity Gated Phase + Gated Log-Mag + Saliency + Moments
        # Features: [s * theta (K), s * |theta| (K), s * log_mag (K), s (K), f_centroid (1), f_spread (1)] = 4*K + 2
        self.spectral_dim = self.num_freq_bins * 4 + 2
        self.proj_freq = nn.Sequential(
            nn.Linear(self.spectral_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln_freq = nn.LayerNorm(embed_dim)

        # 3. Symmetric Dual-Domain Balanced Highway Fusion
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Spatial Positional Embedding
        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scan_ref_x: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, 5, 5, C] tensor (or [5, 5, C])
            scan_ref_x: Optional [C] or [1, C] or [B, C] global scan sound-metal reference waveform.
                        When None, operates in pure autonomous single-probe mode.
        Returns:
            tokens: [B, 25, D]
            pos:    [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_grid = x.reshape(B, self.num_spatial, C)  # [B, 25, C]
        x_flat = x_grid.reshape(B * self.num_spatial, C)  # [B*25, C]

        # --- A. Time Domain Feature Extraction (100% Autonomous, No Local Spatial Subtraction) ---
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        x_std = x_flat.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm = (x_flat - x_mean) / x_std
        abs_x_norm = torch.abs(x_norm)

        # Physical transient invariants:
        t_p = torch.argmax(torch.abs(x_flat), dim=-1, keepdim=True).float() / float(C)  # [B*25, 1]
        v_max, _ = torch.max(x_flat, dim=-1, keepdim=True)
        v_min, _ = torch.min(x_flat, dim=-1, keepdim=True)
        v_pp = v_max - v_min                                                             # [B*25, 1]
        e_time = torch.mean(torch.abs(x_flat), dim=-1, keepdim=True)                    # [B*25, 1]

        phi_time = torch.cat([x_norm, abs_x_norm, t_p, v_pp, e_time], dim=-1).to(x.dtype)
        z_time = self.ln_time(self.time_proj(phi_time))  # [B*25, D]

        # --- B. Frequency Domain Feature Extraction (Uncrushed Harmonic Decomposition) ---
        x_fp32 = x_flat.float()
        X_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B*25, C//2 + 1]
        self._last_fft = X_fft

        X_sub = X_fft[:, 1:self.num_freq_bins + 1]  # [B*25, K], exclude DC
        mag = torch.abs(X_sub)                      # [B*25, K]
        P = mag ** 2
        P_tot = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        s = P / P_tot                               # Energy saliency gating [B*25, K] in [0, 1]

        # Dodd-Deeds Lift-off Invariant Phase:
        if scan_ref_x is not None:
            # Calibrated Mode (M1): Deconvolution against global scan-level sound reference
            ref_flat = scan_ref_x.reshape(-1, C).float()
            X_ref = torch.fft.rfft(ref_flat, dim=-1)[:, 1:self.num_freq_bins + 1]
            prod = X_sub * torch.conj(X_ref)
            theta = torch.angle(prod) / torch.pi
        else:
            # Pure Autonomous Mode: Direct Dodd-Deeds Fourier Phase
            theta = torch.angle(X_sub) / torch.pi   # [B*25, K] in [-1, 1]

        abs_theta = torch.abs(theta)                # [B*25, K] in [0, 1]
        self._last_phase = theta

        # Energy-Saliency Gating & Invariant Projection:
        theta_gated = s * theta                     # Signed phase delay
        abs_theta_gated = s * abs_theta             # Polarity-invariant phase magnitude
        mag_gated = s * torch.log1p(mag)            # Gated log-magnitude

        # Continuous Spectral Moments:
        bin_indices = torch.arange(1, self.num_freq_bins + 1, device=x.device, dtype=torch.float32).unsqueeze(0)
        f_centroid = (bin_indices * s).sum(dim=-1, keepdim=True) / float(self.num_freq_bins)
        var_f = ((bin_indices / float(self.num_freq_bins) - f_centroid) ** 2 * s).sum(dim=-1, keepdim=True)
        f_spread = torch.sqrt(torch.clamp(var_f, min=0.0) + 1e-8)

        phi_freq = torch.cat([theta_gated, abs_theta_gated, mag_gated, s, f_centroid, f_spread], dim=-1).to(x.dtype)
        z_freq = self.ln_freq(self.proj_freq(phi_freq))  # [B*25, D]

        # --- C. Symmetric Balanced Dual-Domain Highway Fusion ---
        z_cat = torch.cat([z_time, z_freq], dim=-1)
        z_fused = self.fuse_proj(z_cat) + 0.5 * (z_time + z_freq)
        tokens = self.drop(self.norm_out(z_fused)).reshape(B, self.num_tokens, self.embed_dim)

        pos = self.pos_embed.expand(B, -1, -1)
        return tokens, pos


def build_tokenizer_5x5(config) -> nn.Module:
    """
    Factory function to construct tokenizer based on config.
    """
    tokenizer_type = getattr(config, "tokenizer_type", "continuous_linear_field")
    if tokenizer_type in ("autonomous_dual_domain", "autonomous_deconv", "autonomous_field", "autonomous_5x5"):
        return AutonomousDualDomainTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
            tikhonov_gamma=getattr(config, "tikhonov_gamma", 0.05),
        )
    elif tokenizer_type in ("energy_stabilized_deconv", "stabilized_deconv", "stabilized_impedance_deconv"):
        return EnergyStabilizedDeconvTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
            tikhonov_gamma=getattr(config, "tikhonov_gamma", 0.05),
        )
    elif tokenizer_type in ("impedance_deconvolution", "impedance_deconv", "dodd_deeds_deconv"):
        return ImpedanceDeconvolutionTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type in ("energy_adaptive_dual_domain", "energy_adaptive_field", "adaptive_dual_domain"):
        return EnergyAdaptiveDualDomainTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type in ("continuous_linear_field", "continuous_linear", "linear_field"):
        return ContinuousLinearFieldTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
            use_snr_tapering=getattr(config, "phase_snr_tapering", True),
            phase_noise_floor=getattr(config, "phase_noise_floor", 0.02),
            temporal_ac_coupling=getattr(config, "temporal_ac_coupling", False),
            adaptive_phase_floor=getattr(config, "adaptive_phase_floor", False),
        )
    elif tokenizer_type in ("continuous_field", "waveform_agnostic_field", "continuous_dual_domain"):
        return ContinuousFieldTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
            use_snr_tapering=getattr(config, "phase_snr_tapering", True),
            phase_noise_floor=getattr(config, "phase_noise_floor", 0.02),
            temporal_ac_coupling=getattr(config, "temporal_ac_coupling", False),
        )
    elif tokenizer_type in ("spatio_spectral", "skin_depth"):
        return SpatioSpectralTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_scales=getattr(config, "num_scales", 4),
            phase_snr_tapering=getattr(config, "phase_snr_tapering", True),
            phase_noise_floor=getattr(config, "phase_noise_floor", 0.05),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
            use_phase_curvature=getattr(config, "use_phase_curvature", False),
            spatial_topology=getattr(config, "spatial_topology", "concentric_star"),
        )
    elif tokenizer_type in ("spatiotemporal_patch", "st_patch", "cst_patch", "auto", "default"):
        return SpatiotemporalPatchTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_temporal_stages=getattr(config, "num_temporal_stages", 4),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type in ("continuous_stf", "continuous_filterbank"):
        return ContinuousSTFTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type in ("uncrushed_diffusion", "continuous_diffusion", "snr_tapered_diffusion"):
        return UncrushedDiffusionTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
            use_snr_tapering=getattr(config, "use_snr_tapering", True),
            phase_noise_floor=getattr(config, "phase_noise_floor", 0.02),
        )
    elif tokenizer_type in ("dual_scale_diffusion", "dual_scale"):
        return DualScaleDiffusionTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type == "dual_domain_attention":
        return DualDomainAttentionTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            num_heads=getattr(config, "tokenizer_heads", 4),
            spectral_features=getattr(config, "spectral_features", "phase_and_mag"),
            phase_snr_tapering=getattr(config, "phase_snr_tapering", True),
            phase_noise_floor=getattr(config, "phase_noise_floor", 0.05),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type == "dual_domain":
        return DualDomainGridTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            spectral_features=getattr(config, "spectral_features", "phase_and_mag"),
            phase_snr_tapering=getattr(config, "phase_snr_tapering", True),
            phase_noise_floor=getattr(config, "phase_noise_floor", 0.05),
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    elif tokenizer_type in ("time_only", "spatial_grid", "standard"):
        return SpatialGridTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


