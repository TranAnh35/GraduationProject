"""
Physical Waveform-Agnostic Dual-Domain Tokenizer for Cross-Condition PECT-JEPA.

Transforms the temporal waveform vector at each spatial coordinate into a fused
dual-domain token:
1. Time Domain: Continuous 1D projection from raw temporal samples to embed_dim // 2.
2. Spectral Domain: Uncrushed Fourier harmonic decomposition (k=1..num_harmonics)
   extracting lift-off invariant normalized phase angle and log-magnitude, projected
   to embed_dim // 2.
3. Positional Embedding: Concentric star 25-probe positional embedding.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..configs.config import CrossConditionConfig
from ..data.topologies import ConcentricStarTopology


class DualDomainTokenizer(nn.Module):
    """
    Continuous Dual-Domain Tokenizer for 25-probe concentric star topology.
    Input: [B, 5, 5, C] -> Output: tokens [B, 25, D], pos [B, 25, D]
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config

        self.num_tokens = 25
        self.temporal_samples = config.temporal_samples
        self.embed_dim = config.embed_dim
        self.num_harmonics = min(config.num_harmonics, config.temporal_samples // 2)

        self.time_dim = config.embed_dim // 2
        self.freq_dim = config.embed_dim - self.time_dim

        # 1. Time branch projection
        self.time_proj = nn.Linear(self.temporal_samples, self.time_dim)

        # 2. Spectral branch projection (phase + log-mag for each harmonic)
        spectral_input_dim = self.num_harmonics * 2
        self.freq_proj = nn.Linear(spectral_input_dim, self.freq_dim)

        # 3. Layer normalization and dropout
        self.norm = nn.LayerNorm(config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        # 4. Learnable 2D positional embedding for 25 concentric star probes
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 5, 5, C] tensor representing 5x5 C-scan patch.
        Returns:
            tokens: [B, 25, D] fused dual-domain token sequence.
            pos:    [B, 25, D] spatial positional embeddings.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == 5 and W == 5, f"Expected 5x5 patch, got {H}x{W}"
        assert C == self.temporal_samples, f"Expected {self.temporal_samples} channels, got {C}"

        x_flat = x.reshape(B, self.num_tokens, C)

        # 1. Time domain branch
        z_time = self.time_proj(x_flat)  # [B, 25, time_dim]

        # 2. Spectral domain branch (torch.fft.rfft along time dimension)
        x_fp32 = x_flat.float()
        x_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B, 25, C//2 + 1] complex
        x_harmonics = x_fft[..., 1:self.num_harmonics + 1]  # Exclude DC offset bin 0

        # Normalized Dodd-Deeds phase angle [-1, 1] & log-magnitude
        phase = torch.angle(x_harmonics) / torch.pi
        mag_linear = torch.abs(x_harmonics)
        mag = torch.log1p(mag_linear)

        spectral_feat = torch.cat([phase, mag], dim=-1).to(x_flat.dtype)
        z_freq = self.freq_proj(spectral_feat)  # [B, 25, freq_dim]

        # 3. Dual-domain fusion
        z_fused = torch.cat([z_time, z_freq], dim=-1)  # [B, 25, D]
        tokens = self.drop(self.norm(z_fused))
        pos = self.pos_embed.expand(B, -1, -1)

        return tokens, pos
