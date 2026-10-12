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
    Supports:
    - 'energy_adaptive': Instance-normalized temporal signal + physical transient scalars
      (peak arrival t_p, V_pp, energy) + energy-gated Dodd-Deeds phase and spectral moments.
    - 'dual_domain': Standard baseline dual-domain projection.
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
        self.tokenizer_type = getattr(config, "tokenizer_type", "energy_adaptive")

        self.time_dim = config.embed_dim // 2
        self.freq_dim = config.embed_dim - self.time_dim

        # 1. Time branch projection
        if self.tokenizer_type == "energy_adaptive":
            # x_norm [C] + peak_time [1] + V_pp [1] + energy [1] = C + 3
            self.time_proj = nn.Linear(self.temporal_samples + 3, self.time_dim)
        elif self.tokenizer_type == "waveform_invariant":
            # x_shape [C] + log_energy [1] + t_peak [1] = C + 2
            self.time_proj = nn.Linear(self.temporal_samples + 2, self.time_dim)
        else:
            self.time_proj = nn.Linear(self.temporal_samples, self.time_dim)

        # 2. Spectral branch projection
        if self.tokenizer_type == "energy_adaptive":
            # phase [K] + log-mag [K] + phase_gated [K] + centroid [1] + spread [1] = 3*K + 2
            spectral_input_dim = self.num_harmonics * 3 + 2
        elif self.tokenizer_type == "waveform_invariant":
            # phase_coherent [K] + saliency [K] = 2*K
            spectral_input_dim = self.num_harmonics * 2
        else:
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

        if self.tokenizer_type == "energy_adaptive":
            # --- 1. Time domain branch with instance normalization & transient physical anchors ---
            mean_t = x_flat.mean(dim=-1, keepdim=True)
            std_t = x_flat.std(dim=-1, keepdim=True) + 1e-6
            x_norm = (x_flat - mean_t) / std_t

            # Transient physical anchors (waveform-agnostic)
            t_peak = torch.argmax(torch.abs(x_norm), dim=-1, keepdim=True).float() / float(C)
            v_max = x_flat.max(dim=-1, keepdim=True).values
            v_min = x_flat.min(dim=-1, keepdim=True).values
            v_pp = torch.log1p(torch.clamp(v_max - v_min, min=0.0))
            energy_t = torch.log1p(torch.mean(x_flat ** 2, dim=-1, keepdim=True))

            time_feat = torch.cat([x_norm, t_peak, v_pp, energy_t], dim=-1)
            z_time = self.time_proj(time_feat)  # [B, 25, time_dim]

            # --- 2. Spectral domain branch with energy-weighted Dodd-Deeds gating ---
            x_fp32 = x_flat.float()
            x_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B, 25, C//2 + 1] complex
            x_harmonics = x_fft[..., 1:self.num_harmonics + 1]  # Exclude DC offset bin 0

            phase = torch.angle(x_harmonics) / torch.pi  # [-1, 1]
            mag_linear = torch.abs(x_harmonics)
            mag = torch.log1p(mag_linear)

            power = mag_linear ** 2
            saliency = power / (torch.sum(power, dim=-1, keepdim=True) + 1e-8)
            phase_gated = saliency * phase

            # Spectral centroid and bandwidth moments
            k_idx = torch.linspace(0.1, 1.0, self.num_harmonics, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            f_c = torch.sum(k_idx * saliency, dim=-1, keepdim=True)
            f_spread = torch.sqrt(torch.sum(((k_idx - f_c) ** 2) * saliency, dim=-1, keepdim=True) + 1e-8)

            spectral_feat = torch.cat([phase, mag, phase_gated, f_c, f_spread], dim=-1).to(x_flat.dtype)
            z_freq = self.freq_proj(spectral_feat)  # [B, 25, freq_dim]

        elif self.tokenizer_type == "waveform_invariant":
            # --- 1. Scale-Invariant Time Domain Branch ---
            # L2 normalized waveform shape: strictly unit energy across all waveforms
            norm_l2 = torch.norm(x_flat, p=2, dim=-1, keepdim=True) + 1e-6
            x_shape = x_flat / norm_l2  # [B, 25, C] - removes waveform amplitude scaling
            log_energy = torch.log1p(norm_l2)  # [B, 25, 1] - compact energy scalar
            t_peak = torch.argmax(torch.abs(x_shape), dim=-1, keepdim=True).float() / float(C)  # [B, 25, 1]
            time_feat = torch.cat([x_shape, log_energy, t_peak], dim=-1)  # [B, 25, C + 2]
            z_time = self.time_proj(time_feat)  # [B, 25, time_dim]

            # --- 2. Spectral Coherence Phase Branch (Dodd-Deeds Invariant) ---
            x_fp32 = x_flat.float()
            x_fft = torch.fft.rfft(x_fp32, dim=-1)
            x_harmonics = x_fft[..., 1:self.num_harmonics + 1]  # [B, 25, K]
            phase = torch.angle(x_harmonics) / torch.pi  # [-1, 1]
            mag_linear = torch.abs(x_harmonics)
            power = mag_linear ** 2
            saliency = power / (torch.sum(power, dim=-1, keepdim=True) + 1e-8)  # Relative power distribution
            phase_coherent = saliency * phase  # Only prominent frequencies contribute phase!
            spectral_feat = torch.cat([phase_coherent, saliency], dim=-1).to(x_flat.dtype)  # [B, 25, 2 * K]
            z_freq = self.freq_proj(spectral_feat)  # [B, 25, freq_dim]

        else:
            # Baseline dual_domain branch
            z_time = self.time_proj(x_flat)  # [B, 25, time_dim]

            x_fp32 = x_flat.float()
            x_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B, 25, C//2 + 1] complex
            x_harmonics = x_fft[..., 1:self.num_harmonics + 1]  # Exclude DC offset bin 0

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

