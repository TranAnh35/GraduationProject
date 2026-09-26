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


def build_tokenizer_5x5(config) -> nn.Module:
    """
    Factory function to construct tokenizer based on config.
    """
    tokenizer_type = getattr(config, "tokenizer_type", "spatio_spectral")
    if tokenizer_type in ("spatio_spectral", "skin_depth"):
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


