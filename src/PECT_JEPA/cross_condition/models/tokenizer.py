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
from ..data.topologies import ConcentricStarTopology, get_spatial_topology_offsets


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
        self.patch_energy_norm = bool(getattr(config, "patch_energy_norm", False)) or (
            self.tokenizer_type == "patch_energy_adaptive"
        )
        self.median_patch_energy = bool(getattr(config, "median_patch_energy", False))
        self.clean_phasor_token = bool(getattr(config, "clean_phasor_token", False))
        self.relative_energy_anchor = bool(getattr(config, "relative_energy_anchor", False))
        self.patch_polarity_align = bool(getattr(config, "patch_polarity_align", False))
        self.bi_directional_relative_energy = bool(getattr(config, "bi_directional_relative_energy", False))

        self.time_dim = config.embed_dim // 2
        self.freq_dim = config.embed_dim - self.time_dim

        # 1. Time branch projection
        if self.clean_phasor_token:
            # x_aligned [C] + t_peak [1] + V_pp [1] + energy [1] + 5 diffusion arrival quantiles [5] = C + 8
            self.time_proj = nn.Linear(self.temporal_samples + 8, self.time_dim)
        elif self.tokenizer_type in ("energy_adaptive", "patch_energy_adaptive"):
            # x_norm [C] + peak_time [1] + V_pp [1] + energy [1]
            # (+ rel_energy [1] if relative_energy_anchor)
            # (+ abs_rel_energy [1] if bi_directional_relative_energy)
            if self.bi_directional_relative_energy:
                extra_time = 5
            elif self.relative_energy_anchor:
                extra_time = 4
            else:
                extra_time = 3
            self.time_proj = nn.Linear(self.temporal_samples + extra_time, self.time_dim)
        elif self.tokenizer_type == "waveform_invariant":
            # x_shape [C] + log_energy [1] + t_peak [1] = C + 2
            self.time_proj = nn.Linear(self.temporal_samples + 2, self.time_dim)
        else:
            self.time_proj = nn.Linear(self.temporal_samples, self.time_dim)

        # 2. Spectral branch projection
        if self.clean_phasor_token:
            # phasor_re [K] + phasor_im [K] + log-mag [K] + saliency [K] + centroid [1] + spread [1] = 4*K + 2
            spectral_input_dim = self.num_harmonics * 4 + 2
        elif self.tokenizer_type in ("energy_adaptive", "patch_energy_adaptive"):
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

        # 4. Spatial positional embedding for 25 concentric star probes:
        # - Option A (continuous_pos_embed=True): Continuous 2D coordinate MLP mapping
        #   normalized physical probe offsets (dr/r_max, dc/r_max) in [-1, 1]^2 to R^D.
        # - Legacy (continuous_pos_embed=False): Discrete [1, 25, D] parameter table.
        self.continuous_pos_embed = bool(getattr(config, "continuous_pos_embed", False))
        # Always precalculate normalized 2D physical coordinates in [-1, 1]^2 for Green's diffusion attention
        offsets = get_spatial_topology_offsets(
            topology=getattr(config, "spatial_topology", "concentric_star"),
            star_radii=tuple(getattr(config, "star_radii", (1, 3, 7))),
            grid_size=getattr(config, "grid_size", 5),
        )
        max_r = float(max(1.0, float(abs(offsets).max())))
        norm_coords = torch.tensor(offsets, dtype=torch.float32).unsqueeze(0) / max_r  # [1, 25, 2]
        self.register_buffer("norm_coords", norm_coords, persistent=False)

        if self.continuous_pos_embed:
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, config.embed_dim),
            )
            nn.init.trunc_normal_(self.pos_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.pos_mlp[0].bias)
            nn.init.trunc_normal_(self.pos_mlp[2].weight, std=0.02)
            nn.init.zeros_(self.pos_mlp[2].bias)
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.pos_mlp = None

        # Precomputed spectral-moment frequency grid (hoisted: avoids rebuilding
        # a linspace tensor on every forward call in the hot training loop)
        k_grid = torch.linspace(0.1, 1.0, self.num_harmonics, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("k_grid", k_grid, persistent=False)
        q_levels = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float32).view(1, 1, 5, 1)
        self.register_buffer("q_levels", q_levels, persistent=False)

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

        if self.clean_phasor_token:
            # --- 1. Polarity-Aligned Time Domain + Diffusion Energy Quantiles ---
            mean_t = x_flat.mean(dim=-1, keepdim=True)
            x_zero = x_flat - mean_t
            std_t = x_zero.std(dim=-1, keepdim=True) + 1e-6  # [B, 25, 1]
            x_norm = x_zero / std_t

            # Canonical peak-polarity alignment: ensures dominant impulse peak is positive (+1)
            # across all sensors (resolving FM18 TMR vs Hall polarity inversion x -> -x).
            idx_peak = torch.argmax(torch.abs(x_norm), dim=-1, keepdim=True)  # [B, 25, 1]
            val_peak = torch.gather(x_norm, -1, idx_peak)
            sign_peak = torch.where(val_peak >= 0, torch.ones_like(val_peak), -torch.ones_like(val_peak))
            x_aligned = x_norm * sign_peak  # [B, 25, C]

            # Decoupled 25-probe median-patch amplitude normalization: removes 56x cross-condition
            # voltage scale while preserving within-patch spatial flaw contrast.
            std_med = torch.median(std_t, dim=1, keepdim=True).values  # [B, 1, 1]
            x_amp = (x_zero / std_med) * sign_peak

            t_peak = idx_peak.to(dtype=x_flat.dtype) / float(C)
            v_max = x_amp.max(dim=-1, keepdim=True).values
            v_min = x_amp.min(dim=-1, keepdim=True).values
            v_pp = torch.log1p(torch.clamp(v_max - v_min, min=0.0))
            energy_t = torch.log1p(torch.mean(x_amp ** 2, dim=-1, keepdim=True))

            # 5 Cumulative Eddy-Current Diffusion Energy Arrival Quantiles (t_10..t_90 in [0, 1]):
            # 100% invariant to gain and polarity; captures subsurface diffusion delay (dt ~ d^2 mu sigma).
            cum_e = torch.cumsum(x_norm ** 2, dim=-1) / float(C)  # [B, 25, C]
            q_lvl = self.q_levels.to(dtype=cum_e.dtype, device=cum_e.device)  # [1, 1, 5, 1]
            t_quantiles = (cum_e.unsqueeze(2) < q_lvl).to(dtype=x_flat.dtype).sum(dim=-1) / float(C)  # [B, 25, 5]

            time_feat = torch.cat([x_aligned, t_peak, v_pp, energy_t, t_quantiles], dim=-1)
            z_time = self.time_proj(time_feat)  # [B, 25, time_dim]

            # --- 2. Branch-Cut-Free Root-Saliency Complex Phasors ---
            # Eliminates FM16 stopband +-pi phase-wrap noise on zero-energy harmonics (s_k=0 -> phasor=0).
            x_fft = torch.fft.rfft(x_amp.float(), dim=-1)  # [B, 25, C//2 + 1] complex
            x_harmonics = x_fft[..., 1:self.num_harmonics + 1]  # Exclude DC offset bin 0
            mag_linear = torch.abs(x_harmonics)
            mag = torch.log1p(mag_linear)

            power = mag_linear ** 2
            total_power = torch.sum(power, dim=-1, keepdim=True) + 1e-8
            saliency = power / total_power
            phasor = x_harmonics / torch.sqrt(total_power)  # sqrt(s_k) * (cos theta_k + i sin theta_k)

            k_idx = self.k_grid.to(dtype=saliency.dtype, device=saliency.device)
            f_c = torch.sum(k_idx * saliency, dim=-1, keepdim=True)
            f_spread = torch.sqrt(torch.sum(((k_idx - f_c) ** 2) * saliency, dim=-1, keepdim=True) + 1e-8)

            spectral_feat = torch.cat(
                [phasor.real, phasor.imag, mag, saliency, f_c, f_spread], dim=-1
            ).to(x_flat.dtype)
            z_freq = self.freq_proj(spectral_feat)  # [B, 25, freq_dim]

        elif self.tokenizer_type in ("energy_adaptive", "patch_energy_adaptive"):
            # --- 1. Time domain branch with instance/patch normalization & transient physical anchors ---
            mean_t = x_flat.mean(dim=-1, keepdim=True)
            if self.patch_energy_norm:
                # Patch-level (25-probe) shared temporal std normalization (dim=(-2, -1)):
                # Removes the 56x cross-waveform / cross-sensor global voltage scale while
                # strictly preserving within-patch spatial amplitude contrast ||x_center||/||x_ring||
                # across all 25 probes (resolving FM11 on Rivet_v2 and Defect Sizing R2).
                x_zero = x_flat - mean_t
                std_patch = x_zero.std(dim=(-2, -1), keepdim=True) + 1e-6  # [B, 1, 1]
                x_norm = x_zero / std_patch

                t_peak = torch.argmax(torch.abs(x_norm), dim=-1, keepdim=True).float() / float(C)
                v_max = x_norm.max(dim=-1, keepdim=True).values
                v_min = x_norm.min(dim=-1, keepdim=True).values
                v_pp = torch.log1p(torch.clamp(v_max - v_min, min=0.0))
                energy_t = torch.log1p(torch.mean(x_norm ** 2, dim=-1, keepdim=True))
                x_fp32 = x_norm.float()
            elif self.median_patch_energy:
                # Decoupled Median-Patch Energy Normalization (fixes FM11 without FM13):
                # 1. Keeps x_norm strictly per-probe unit-variance ((x - mean_t) / std_t) so
                #    the 128 waveform shape samples are 100% immune to outer-ring boundary steps (FM13).
                # 2. Normalizes amplitude/spectral anchors (v_pp, energy_t, mag) by the robust
                #    25-probe median std (50% breakdown point), removing the 56x cross-condition
                #    voltage scale while preserving relative center/ring amplitude contrast (FM11).
                x_zero = x_flat - mean_t
                std_t = x_zero.std(dim=-1, keepdim=True) + 1e-6  # [B, 25, 1]
                x_norm = x_zero / std_t

                std_med = torch.median(std_t, dim=1, keepdim=True).values  # [B, 1, 1]
                x_amp = x_zero / std_med

                t_peak = torch.argmax(torch.abs(x_norm), dim=-1, keepdim=True).float() / float(C)
                v_max = x_amp.max(dim=-1, keepdim=True).values
                v_min = x_amp.min(dim=-1, keepdim=True).values
                v_pp = torch.log1p(torch.clamp(v_max - v_min, min=0.0))
                energy_t = torch.log1p(torch.mean(x_amp ** 2, dim=-1, keepdim=True))
                x_fp32 = x_amp.float()
            else:
                std_t = x_flat.std(dim=-1, keepdim=True) + 1e-6
                x_norm = (x_flat - mean_t) / std_t

                if self.patch_polarity_align:
                    # Determine patch-wide polarity from center probe 0 dominant peak
                    idx_p0 = torch.argmax(torch.abs(x_norm[:, 0:1, :]), dim=-1, keepdim=True)  # [B, 1, 1]
                    val_p0 = torch.gather(x_norm[:, 0:1, :], -1, idx_p0)  # [B, 1, 1]
                    sign_patch = torch.where(val_p0 >= 0, torch.ones_like(val_p0), -torch.ones_like(val_p0))  # [B, 1, 1]
                    x_norm = x_norm * sign_patch
                    x_flat_aligned = x_flat * sign_patch
                else:
                    sign_patch = None
                    x_flat_aligned = x_flat

                # Transient physical anchors (waveform-agnostic)
                t_peak = torch.argmax(torch.abs(x_norm), dim=-1, keepdim=True).float() / float(C)
                v_max = x_flat_aligned.max(dim=-1, keepdim=True).values
                v_min = x_flat_aligned.min(dim=-1, keepdim=True).values
                v_pp = torch.log1p(torch.clamp(v_max - v_min, min=0.0))
                energy_t = torch.log1p(torch.mean(x_flat_aligned ** 2, dim=-1, keepdim=True))
                x_fp32 = x_flat_aligned.float()

                if self.bi_directional_relative_energy:
                    # Bi-directional spatial energy contrast (fixes Rivet_v2 half-space sign inversion in latent space):
                    # Relative probe energy compared to surrounding context probes (probes 1..24)
                    e_probe = std_t ** 2  # [B, 25, 1]
                    e_context = e_probe[:, 1:, :].mean(dim=1, keepdim=True) + 1e-8  # [B, 1, 1]
                    rel_energy = torch.log1p(e_probe) - torch.log1p(e_context)  # [B, 25, 1]
                    abs_rel_energy = torch.abs(rel_energy)  # [B, 25, 1]
                    time_feat = torch.cat([x_norm, t_peak, v_pp, energy_t, rel_energy, abs_rel_energy], dim=-1)
                elif self.relative_energy_anchor:
                    # Decoupled spatial energy contrast (fixes FM11 on Rivet_v2 without FM13/FM14):
                    # Relative probe energy compared to surrounding context probes (probes 1..24)
                    e_probe = std_t ** 2  # [B, 25, 1]
                    e_context = e_probe[:, 1:, :].mean(dim=1, keepdim=True) + 1e-8  # [B, 1, 1]
                    rel_energy = torch.log1p(e_probe) - torch.log1p(e_context)  # [B, 25, 1]
                    time_feat = torch.cat([x_norm, t_peak, v_pp, energy_t, rel_energy], dim=-1)
                else:
                    time_feat = torch.cat([x_norm, t_peak, v_pp, energy_t], dim=-1)

            z_time = self.time_proj(time_feat)  # [B, 25, time_dim]

            # --- 2. Spectral domain branch with energy-weighted Dodd-Deeds gating ---
            x_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B, 25, C//2 + 1] complex
            x_harmonics = x_fft[..., 1:self.num_harmonics + 1]  # Exclude DC offset bin 0

            phase = torch.angle(x_harmonics) / torch.pi  # [-1, 1]
            mag_linear = torch.abs(x_harmonics)
            mag = torch.log1p(mag_linear)

            power = mag_linear ** 2
            saliency = power / (torch.sum(power, dim=-1, keepdim=True) + 1e-8)
            phase_gated = saliency * phase

            # Spectral centroid and bandwidth moments
            k_idx = self.k_grid.to(dtype=x_flat.dtype, device=x_flat.device)
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
        if self.continuous_pos_embed:
            coords = self.norm_coords.to(dtype=tokens.dtype, device=tokens.device)
            pos = self.pos_mlp(coords).expand(B, -1, -1)
        else:
            pos = self.pos_embed.expand(B, -1, -1)

        return tokens, pos

