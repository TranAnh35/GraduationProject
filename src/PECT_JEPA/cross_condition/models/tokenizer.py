"""
Physical Waveform-Agnostic Dual-Domain Tokenizer for Cross-Condition PECT-JEPA.

Single canonical tokenizer (exp70 lineage). Each spatial coordinate's temporal
waveform is mapped to a fused dual-domain token:

1. Time domain (embed_dim // 2):
   - per-probe unit-variance waveform shape x_norm (128 samples),
   - patch-consistent polarity alignment (center probe 0 dominant peak),
   - physical anchors t_peak, log V_pp, log energy,
   - bi-directional spatial energy contrast
       Delta E_local = log(1 + E_probe) - log(1 + E_context_ring)
       |Delta E_local|,
     where E_probe is the per-probe temporal variance and E_context_ring the mean
     over the other 24 probes of the same patch,
   - optional global background anchor
       Delta E_global = log(1 + E_probe) - log(1 + E_bg_ref),
     with E_bg_ref the median energy of the scan's four defect-free corner
     patches. Delta E_local loses contrast inside wide defects (the context ring
     sits inside the defect); the corner anchor keeps it.
2. Spectral domain (embed_dim - embed_dim // 2): energy-weighted Dodd-Deeds
   harmonics k = 1..K (phase, log-magnitude, saliency-gated phase, centroid,
   spectral spread), with the DC bin removed.
3. Positional embedding: continuous 2D coordinate MLP (dr, dc) -> R^D.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from ..configs.config import CrossConditionConfig
from ..data.topologies import get_spatial_topology_offsets


class DualDomainTokenizer(nn.Module):
    """
    Continuous Dual-Domain Tokenizer for the 25-probe concentric star topology.
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
        self.global_bg_energy_anchor = bool(getattr(config, "global_bg_energy_anchor", False))

        self.time_dim = config.embed_dim // 2
        self.freq_dim = config.embed_dim - self.time_dim

        # 1. Time branch: x_norm [C] + t_peak + V_pp + energy + Delta E_local
        #    + |Delta E_local| (+ Delta E_global)
        time_input_dim = self.temporal_samples + 5 + int(self.global_bg_energy_anchor)
        self.time_proj = nn.Linear(time_input_dim, self.time_dim)

        # 2. Spectral branch: phase [K] + log-mag [K] + phase_gated [K] + centroid + spread
        self.freq_proj = nn.Linear(self.num_harmonics * 3 + 2, self.freq_dim)

        # 3. Layer normalization and dropout
        self.norm = nn.LayerNorm(config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        # 4. Continuous 2D positional embedding for the 25 star probes.
        offsets = get_spatial_topology_offsets(
            topology=getattr(config, "spatial_topology", "concentric_star"),
            star_radii=tuple(getattr(config, "star_radii", (2, 4, 7))),
            grid_size=getattr(config, "grid_size", 5),
        )
        max_r = float(max(1.0, float(abs(offsets).max())))
        self.register_buffer(
            "norm_coords",
            torch.tensor(offsets, dtype=torch.float32).unsqueeze(0) / max_r,
            persistent=False,
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        nn.init.trunc_normal_(self.pos_mlp[0].weight, std=0.02)
        nn.init.zeros_(self.pos_mlp[0].bias)
        nn.init.trunc_normal_(self.pos_mlp[2].weight, std=0.02)
        nn.init.zeros_(self.pos_mlp[2].bias)

        # Precomputed spectral-moment frequency grid (hoisted out of the hot loop)
        k_grid = torch.linspace(0.1, 1.0, self.num_harmonics, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("k_grid", k_grid, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        bg_ref_energy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 5, 5, C] raw 5x5 C-scan patch (C temporal samples).
            bg_ref_energy: optional [B], [B, 1] or [B, 1, 1] per-sample sound-metal
                background energy (per-probe temporal variance units). Only consumed
                when global_bg_energy_anchor is enabled; when absent, the patch
                median probe energy is used as the anchor so train/inference stay
                consistent for callers that do not have a full scan.
        Returns:
            tokens: [B, 25, D] fused dual-domain tokens.
            pos:    [B, 25, D] spatial positional embeddings.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == 5 and W == 5, f"Expected 5x5 patch, got {H}x{W}"
        assert C == self.temporal_samples, f"Expected {self.temporal_samples} channels, got {C}"

        x_flat = x.reshape(B, self.num_tokens, C)
        mean_t = x_flat.mean(dim=-1, keepdim=True)
        std_t = x_flat.std(dim=-1, keepdim=True) + 1e-6          # [B, 25, 1]
        x_norm = (x_flat - mean_t) / std_t

        # --- 1. Patch-consistent polarity alignment -------------------------
        # The dominant peak of the center probe defines the sign for the whole
        # patch, resolving TMR/Hall polarity inversion without per-probe
        # checkerboard flips or root-saliency spectral starvation.
        idx_p0 = torch.argmax(torch.abs(x_norm[:, 0:1, :]), dim=-1, keepdim=True)
        val_p0 = torch.gather(x_norm[:, 0:1, :], -1, idx_p0)
        sign_patch = torch.where(val_p0 >= 0, torch.ones_like(val_p0), -torch.ones_like(val_p0))
        x_norm = x_norm * sign_patch
        x_aligned = x_flat * sign_patch

        # --- 2. Transient anchors and spatial energy contrast ----------------
        t_peak = torch.argmax(torch.abs(x_norm), dim=-1, keepdim=True).float() / float(C)
        v_max = x_aligned.max(dim=-1, keepdim=True).values
        v_min = x_aligned.min(dim=-1, keepdim=True).values
        v_pp = torch.log1p(torch.clamp(v_max - v_min, min=0.0))
        energy_t = torch.log1p(torch.mean(x_aligned ** 2, dim=-1, keepdim=True))

        e_probe = std_t ** 2                                      # [B, 25, 1]
        e_context = e_probe[:, 1:, :].mean(dim=1, keepdim=True) + 1e-8
        rel_energy = torch.log1p(e_probe) - torch.log1p(e_context)  # [B, 25, 1]

        time_feats = [x_norm, t_peak, v_pp, energy_t, rel_energy, torch.abs(rel_energy)]
        if self.global_bg_energy_anchor:
            if bg_ref_energy is None:
                e_bg_ref = torch.median(e_probe, dim=1, keepdim=True).values
            else:
                e_bg_ref = bg_ref_energy.reshape(B, 1, 1).to(dtype=e_probe.dtype, device=e_probe.device)
            rel_energy_global = torch.log1p(e_probe) - torch.log1p(e_bg_ref)
            time_feats.append(rel_energy_global)
        z_time = self.time_proj(torch.cat(time_feats, dim=-1))     # [B, 25, time_dim]

        # --- 3. Spectral domain: energy-weighted Dodd-Deeds harmonics --------
        x_fp32 = x_aligned.float()
        x_fft = torch.fft.rfft(x_fp32, dim=-1)                    # [B, 25, C//2 + 1]
        x_harmonics = x_fft[..., 1:self.num_harmonics + 1]        # drop DC bin 0

        phase = torch.angle(x_harmonics) / torch.pi               # [-1, 1]
        mag_linear = torch.abs(x_harmonics)
        mag = torch.log1p(mag_linear)

        power = mag_linear ** 2
        saliency = power / (torch.sum(power, dim=-1, keepdim=True) + 1e-8)
        phase_gated = saliency * phase

        k_idx = self.k_grid.to(dtype=saliency.dtype, device=saliency.device)
        f_c = torch.sum(k_idx * saliency, dim=-1, keepdim=True)
        f_spread = torch.sqrt(torch.sum(((k_idx - f_c) ** 2) * saliency, dim=-1, keepdim=True) + 1e-8)

        spectral_feat = torch.cat([phase, mag, phase_gated, f_c, f_spread], dim=-1).to(x_flat.dtype)
        z_freq = self.freq_proj(spectral_feat)                    # [B, 25, freq_dim]

        # --- 4. Dual-domain fusion ------------------------------------------
        tokens = self.drop(self.norm(torch.cat([z_time, z_freq], dim=-1)))
        coords = self.norm_coords.to(dtype=tokens.dtype, device=tokens.device)
        pos = self.pos_mlp(coords).expand(B, -1, -1)
        return tokens, pos
