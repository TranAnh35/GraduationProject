"""
Spatial Grid Tokenizer for 5x5 PECT-JEPA.

Linearly projects the C-channel temporal waveform vector at each spatial coordinate (i, j)
into an embedding token of dimension D, combined with a 2D spatial positional embedding.
"""

import math
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


def build_tokenizer_5x5(config) -> nn.Module:
    """
    Factory function to construct tokenizer based on config.
    """
    tokenizer_type = getattr(config, "tokenizer_type", "dual_domain")
    if tokenizer_type == "dual_domain":
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
    elif tokenizer_type == "time_only":
        return SpatialGridTokenizer5x5(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            grid_size=config.grid_size,
            pos_embed_type=config.pos_embed_type,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
