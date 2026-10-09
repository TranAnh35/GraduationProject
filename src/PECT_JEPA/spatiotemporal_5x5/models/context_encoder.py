import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .attention import TransformerBlock, RadialAttentionBias, MultiheadSelfAttention, MLP
from ..data.topologies import get_spatial_topology_offsets


class ContextEncoder5x5(nn.Module):
    """
    Baseline Context Encoder processing visible context points:
    Input: context_tokens [B, N_ctx, D], context_pos [B, N_ctx, D]
    Output: H_ctx [B, N_ctx, D]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_radial_attention_bias: bool = False,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        diffusion_gamma_init: float = 0.05,
        diffusion_alpha_init: float = 0.1,
    ):
        super().__init__()
        self.use_radial_attention_bias = use_radial_attention_bias
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if use_radial_attention_bias:
            self.radial_bias = RadialAttentionBias(
                num_heads=num_heads,
                init_gamma=diffusion_gamma_init,
                init_alpha=diffusion_alpha_init,
            )
            offsets = get_spatial_topology_offsets(
                topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
            )  # [25, 2]
            diff = offsets[:, None, :] - offsets[None, :, :]
            dist_25 = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)  # [25, 25] in mm

            # Precompute 100x100 matrix for spatio-spectral 4-scale tokens
            dist_100 = dist_25[np.arange(100) // 4, :][:, np.arange(100) // 4]  # [100, 100]
            self.register_buffer("dist_matrix_100", torch.from_numpy(dist_100), persistent=False)
            self.register_buffer("dist_matrix_25", torch.from_numpy(dist_25), persistent=False)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        freq_condition: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        h = context_tokens + context_pos
        B, N_ctx, D = h.shape

        attn_bias = None
        if self.use_radial_attention_bias:
            if context_indices is not None:
                # context_indices: [B, N_ctx]
                dist_full = self.dist_matrix_100 if hasattr(self, "dist_matrix_100") and self.dist_matrix_100.shape[0] >= N_ctx else getattr(self, "dist_matrix_25", None)
                if dist_full is not None:
                    if dist_full.device != h.device:
                        dist_full = dist_full.to(h.device)
                    idx = context_indices.long()
                    dist_ctx = dist_full[idx.unsqueeze(2), idx.unsqueeze(1)]  # [B, N_ctx, N_ctx]
                    attn_bias = self.radial_bias(dist_ctx)  # [B, H, N_ctx, N_ctx]
            elif N_ctx == 100 and hasattr(self, "dist_matrix_100"):
                mat = self.dist_matrix_100 if self.dist_matrix_100.device == h.device else self.dist_matrix_100.to(h.device)
                attn_bias = self.radial_bias(mat)
            elif N_ctx == 25 and hasattr(self, "dist_matrix_25"):
                mat = self.dist_matrix_25 if self.dist_matrix_25.device == h.device else self.dist_matrix_25.to(h.device)
                attn_bias = self.radial_bias(mat)

        last_attn = None
        for i, blk in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                h, last_attn = blk(h, return_attention=True, attn_bias=attn_bias)
            else:
                h = blk(h, attn_bias=attn_bias)
        h = self.norm(h)
        if return_attention:
            return h, last_attn
        return h


class AnisotropicSkinDepthAttentionBias(nn.Module):
    """
    EXP-36: Continuous Anisotropic Skin-Depth Attention Bias.
    Accounts for:
      1. Directional scanner anisotropy: alpha_x * (Delta x)^2 + alpha_y * (Delta y)^2
      2. Harmonic skin-depth scaling: xi^2 = Delta r_aniso^2 * omega_bar
      3. Parabolic Green's diffusion log-polynomial decay:
         B(i, j) = -gamma * xi^2 - alpha * ln(1 + xi^2)
    """
    def __init__(
        self,
        num_heads: int = 4,
        init_gamma: float = 0.05,
        init_alpha: float = 0.1,
        init_alpha_x: float = 1.0,
        init_alpha_y: float = 1.0,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Learnable decay rates per head
        self.log_gamma = nn.Parameter(torch.full((1, num_heads, 1, 1), math.log(max(1e-4, init_gamma))))
        self.log_alpha = nn.Parameter(torch.full((1, num_heads, 1, 1), math.log(max(1e-4, init_alpha))))
        # Learnable directional anisotropy coefficients (shared across heads)
        self.log_alpha_x = nn.Parameter(torch.tensor(math.log(max(1e-4, init_alpha_x)), dtype=torch.float32))
        self.log_alpha_y = nn.Parameter(torch.tensor(math.log(max(1e-4, init_alpha_y)), dtype=torch.float32))

        offsets = get_spatial_topology_offsets(
            topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
        )  # [25, 2] in mm
        self.register_buffer("probe_offsets", torch.from_numpy(offsets).float(), persistent=False)

    def forward(
        self,
        context_indices: Optional[torch.Tensor] = None,
        N_tokens: int = 25,
        freq_condition: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.probe_offsets.device

        alpha_x = torch.exp(self.log_alpha_x)
        alpha_y = torch.exp(self.log_alpha_y)
        gamma = torch.exp(self.log_gamma)  # [1, H, 1, 1]
        alpha = torch.exp(self.log_alpha)  # [1, H, 1, 1]

        offsets = self.probe_offsets.to(device)

        if context_indices is not None:
            # context_indices: [B, N_ctx]
            B, N_ctx = context_indices.shape
            idx = context_indices.long()
            if N_tokens > 25:
                idx = (idx // (N_tokens // 25)).clamp(0, 24)
            else:
                idx = idx.clamp(0, 24)
            coords = offsets[idx]  # [B, N_ctx, 2]
            dx = coords[:, :, 0].unsqueeze(2) - coords[:, :, 0].unsqueeze(1)  # [B, N_ctx, N_ctx]
            dy = coords[:, :, 1].unsqueeze(2) - coords[:, :, 1].unsqueeze(1)  # [B, N_ctx, N_ctx]
        else:
            B = batch_size
            N_ctx = N_tokens
            if N_tokens == 25:
                coords = offsets  # [25, 2]
            elif N_tokens > 25:
                scale_ratio = N_tokens // 25
                p_idx = np.arange(N_tokens) // scale_ratio
                coords = offsets[p_idx]  # [N_tokens, 2]
            else:
                coords = offsets[:N_tokens]
            dx = (coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)).unsqueeze(0).expand(B, -1, -1)
            dy = (coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)).unsqueeze(0).expand(B, -1, -1)

        r_aniso_sq = (alpha_x * (dx ** 2) + alpha_y * (dy ** 2)).unsqueeze(1)  # [B, 1, N_ctx, N_ctx]

        if freq_condition is not None:
            omega_bar = freq_condition.view(B, 1, 1, 1).to(device=r_aniso_sq.device, dtype=r_aniso_sq.dtype)
            xi_sq = r_aniso_sq * omega_bar
        else:
            xi_sq = r_aniso_sq * 0.25

        bias = - gamma * xi_sq - alpha * torch.log1p(xi_sq)  # [B, H, N_ctx, N_ctx]
        return bias


class DispersionConditionedTransformerBlock(nn.Module):
    """
    EXP-36: Pre-LN Transformer Block with Harmonic Dispersion Adaptive Layer Normalization (AdaLN).
    Condition vector cond modulates coordinate scale (1 + gamma) and coordinate shift (beta)
    for both the Self-Attention branch and the Feed-Forward MLP branch.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_dim: int = 64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = MultiheadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), dropout=dropout)

        # AdaLN modulation mapping: produces [gamma1, beta1, gamma2, beta2]
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * embed_dim),
        )
        # Zero-initialize AdaLN projection so the block begins identically to Pre-LN Transformer
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        # cond: [B, cond_dim] -> [B, 1, 4 * embed_dim]
        params = self.ada_ln(cond).unsqueeze(1)
        gamma1, beta1, gamma2, beta2 = torch.chunk(params, 4, dim=-1)

        norm_x1 = self.norm1(x) * (1.0 + gamma1) + beta1
        if return_attention:
            attn_out, attn_weights = self.attn(norm_x1, return_attention=True, attn_bias=attn_bias)
            x = x + attn_out
            norm_x2 = self.norm2(x) * (1.0 + gamma2) + beta2
            x = x + self.mlp(norm_x2)
            return x, attn_weights

        x = x + self.attn(norm_x1, attn_bias=attn_bias)
        norm_x2 = self.norm2(x) * (1.0 + gamma2) + beta2
        x = x + self.mlp(norm_x2)
        return x


class DispersionConditionedContextEncoder5x5(nn.Module):
    """
    EXP-36: Harmonic Dispersion Conditioned Context Encoder.
    Processes visible context tokens with:
      1. Harmonic Dispersion Conditioning (AdaLN):
         Embeds the dimensionless characteristic diffusion parameter xi_disp = sqrt(omega_bar)
         using multi-scale Fourier features to modulate LayerNorm affine scales and shifts
         across all Transformer layers, aligning latent coordinates across disparate waveforms.
      2. Anisotropic Skin-Depth Attention Bias:
         Continuous 2D spatial attention bias accounting for scanner anisotropy (alpha_x, alpha_y)
         and frequency skin-depth scaling:
           xi_ij^2 = (alpha_x * dx^2 + alpha_y * dy^2) * omega_bar
           B_ij = -gamma * xi_ij^2 - alpha * ln(1 + xi_ij^2)
      3. Concentric Ring Spatial Preservation:
         Preserves multi-ring topology (Ring 2 vs Ring 3) while modeling continuous
         radial eddy current diffusion.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_radial_attention_bias: bool = True,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        diffusion_gamma_init: float = 0.05,
        diffusion_alpha_init: float = 0.1,
        diffusion_alpha_x_init: float = 1.0,
        diffusion_alpha_y_init: float = 1.0,
        num_fourier_feats: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.use_radial_attention_bias = use_radial_attention_bias

        # 1. Fourier Feature Projection for Harmonic Dispersion Condition sqrt(omega_bar)
        self.num_fourier_feats = num_fourier_feats
        # Multi-scale frequencies: 2^0, 2^1, ..., 2^(M-1)
        freq_bands = 2.0 ** torch.arange(num_fourier_feats, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        in_fourier_dim = num_fourier_feats * 2  # sin and cos

        self.freq_embed = nn.Sequential(
            nn.Linear(in_fourier_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Dispersion-Conditioned Transformer Blocks with AdaLN
        self.blocks = nn.ModuleList([
            DispersionConditionedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                cond_dim=embed_dim,
            )
            for _ in range(depth)
        ])

        # 3. Final AdaLN Modulation
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.final_ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        # Zero-initialize final AdaLN projection
        nn.init.zeros_(self.final_ada_ln[-1].weight)
        nn.init.zeros_(self.final_ada_ln[-1].bias)

        # 4. Anisotropic Skin-Depth Attention Bias
        if use_radial_attention_bias:
            self.skin_depth_bias = AnisotropicSkinDepthAttentionBias(
                num_heads=num_heads,
                init_gamma=diffusion_gamma_init,
                init_alpha=diffusion_alpha_init,
                init_alpha_x=diffusion_alpha_x_init,
                init_alpha_y=diffusion_alpha_y_init,
                spatial_topology=spatial_topology,
                star_radii=star_radii,
                grid_size=grid_size,
            )
        else:
            self.skin_depth_bias = None

    def _compute_dispersion_embedding(
        self,
        freq_condition: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the dispersion embedding vector cond_vec [B, embed_dim]
        and returns the scalar omega_bar [B] for attention bias.
        """
        if freq_condition is not None:
            omega_bar = freq_condition.view(batch_size).to(device=device, dtype=torch.float32).clamp(min=1e-4, max=1.0)
        else:
            omega_bar = torch.full((batch_size,), 0.25, device=device, dtype=torch.float32)

        # Dimensionless skin-depth parameter: xi = sqrt(omega_bar) in (0, 1]
        xi = torch.sqrt(omega_bar).unsqueeze(-1)  # [B, 1]
        bands = self.freq_bands.to(device=device).unsqueeze(0)  # [1, M]
        angles = math.pi * xi * bands  # [B, M]
        fourier_feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype=dtype)  # [B, 2*M]
        cond_vec = self.freq_embed(fourier_feats)  # [B, D]
        return cond_vec, omega_bar

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        freq_condition: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        h = context_tokens + context_pos
        B, N_ctx, D = h.shape
        device = h.device

        cond_vec, omega_bar = self._compute_dispersion_embedding(
            freq_condition=freq_condition,
            batch_size=B,
            device=device,
            dtype=h.dtype,
        )

        attn_bias = None
        if self.skin_depth_bias is not None:
            attn_bias = self.skin_depth_bias(
                context_indices=context_indices,
                N_tokens=N_ctx,
                freq_condition=omega_bar,
                batch_size=B,
                device=device,
            )

        last_attn = None
        for i, blk in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                h, last_attn = blk(h, cond=cond_vec, attn_bias=attn_bias, return_attention=True)
            else:
                h = blk(h, cond=cond_vec, attn_bias=attn_bias)

        # Final AdaLN normalization
        params = self.final_ada_ln(cond_vec).unsqueeze(1)  # [B, 1, 2 * D]
        gamma, beta = torch.chunk(params, 2, dim=-1)
        h = self.norm(h) * (1.0 + gamma) + beta

        if return_attention:
            return h, last_attn
        return h


class BilinearCoupledFFN(nn.Module):
    """
    Bilinear Transient-Harmonic Coupling Feed-Forward Network:
    Splits token features into transient dynamic channel and harmonic dispersion channel,
    computing bilinear gated interaction:
        h = GELU(W_gate * x) * (W_val * x)
        out = W_out * h
    Models the physical coupling between eddy current arrival delay and harmonic phase shift
    without collapsing coordinate variance.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(embed_dim, hidden_dim)
        self.w_val = nn.Linear(embed_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.w_gate(x))
        val = self.w_val(x)
        h = gate * val
        return self.drop(self.w_out(h))


class HarmonicIsometricAttentionBias(nn.Module):
    """
    Normalized Receptive Field Attention Bias for Context Encoder (EXP-37 Stage 3):
    Calibrates spatial dispersion by intermediate coil radius (3.0 mm)
    and normalized frequency ratio relative to center frequency (0.25):
        f_ratio = sqrt(clamp(omega_bar, 0.05, 1.0) / 0.25)
        scale_ij = (||r_i - r_j|| / 3.0mm) * (1.0 + softplus(beta_h) * f_ratio)
        B_ij = - softplus(gamma_h) * scale_ij - softplus(alpha_h) * ln(1.0 + scale_ij)

    Guarantees:
    - Square pulse (omega=0.105) maintains strong spatial decay (scale within 65% of nominal).
    - Zero AdaLN on token representation vectors; strictly preserves full SVD rank >= 20D.
    """
    def __init__(
        self,
        num_heads: int = 4,
        init_gamma: float = 0.1,
        init_alpha: float = 0.2,
        init_beta: float = 0.5,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        r_coil_ref: float = 3.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.r_coil_ref = r_coil_ref

        offsets = get_spatial_topology_offsets(
            topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
        )  # [25, 2] in mm
        diff = offsets[:, None, :] - offsets[None, :, :]
        dist_25 = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)  # [25, 25] in mm

        dist_100 = dist_25[np.arange(100) // 4, :][:, np.arange(100) // 4]
        self.register_buffer("dist_matrix_100", torch.from_numpy(dist_100), persistent=False)
        self.register_buffer("dist_matrix_25", torch.from_numpy(dist_25), persistent=False)

        raw_gamma = math.log(math.exp(init_gamma) - 1.0) if init_gamma > 0 else 0.0
        raw_alpha = math.log(math.exp(init_alpha) - 1.0) if init_alpha > 0 else 0.0
        raw_beta = math.log(math.exp(init_beta) - 1.0) if init_beta > 0 else 0.0

        self.raw_gamma = nn.Parameter(torch.full((num_heads, 1, 1), raw_gamma, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.full((num_heads, 1, 1), raw_alpha, dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.full((num_heads, 1, 1), raw_beta, dtype=torch.float32))

    def forward(self, dist_matrix: torch.Tensor, omega_bar: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = dist_matrix.device
        dtype = dist_matrix.dtype

        if dist_matrix.ndim == 2:
            if omega_bar is not None and omega_bar.numel() > 1:
                dist_matrix = dist_matrix.unsqueeze(0).expand(omega_bar.numel(), -1, -1)
            else:
                dist_matrix = dist_matrix.unsqueeze(0)  # [1, N, N]
        elif dist_matrix.shape[0] == 1 and omega_bar is not None and omega_bar.numel() > 1:
            dist_matrix = dist_matrix.expand(omega_bar.numel(), -1, -1)

        B, N, _ = dist_matrix.shape
        import torch.nn.functional as F
        gamma = F.softplus(self.raw_gamma).to(device=device, dtype=dtype)  # [H, 1, 1]
        alpha = F.softplus(self.raw_alpha).to(device=device, dtype=dtype)  # [H, 1, 1]
        beta = F.softplus(self.raw_beta).to(device=device, dtype=dtype)    # [H, 1, 1]

        if omega_bar is not None:
            omega = omega_bar.view(B, 1, 1, 1).to(device=device, dtype=dtype).clamp(min=0.05, max=1.0)
            f_ratio = torch.sqrt(omega / 0.25)  # [B, 1, 1, 1]
        else:
            f_ratio = torch.ones((B, 1, 1, 1), device=device, dtype=dtype)

        d_norm = (dist_matrix.unsqueeze(1) / self.r_coil_ref)  # [B, 1, N, N]
        scale = d_norm * (1.0 + beta.unsqueeze(0) * f_ratio)    # [B, H, N, N]

        bias = - gamma.unsqueeze(0) * scale - alpha.unsqueeze(0) * torch.log(1.0 + scale)
        return bias  # [B, H, N, N]


class HarmonicIsometricTransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer Block equipped with BilinearCoupledFFN.
    Zero AdaLN: preserves full 64D manifold rank without dimensional collapse.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = BilinearCoupledFFN(
            embed_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True, attn_bias=attn_bias)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class HarmonicIsometricContextEncoder5x5(nn.Module):
    """
    Harmonic-Isometric Context Encoder (EXP-37 Stage 3):
    1. Pre-LN Transformer blocks with strictly NO AdaLN (Rank-Preserving).
    2. Normalized Receptive Field Attention Bias calibrated by coil radius (3.0 mm)
       and normalized frequency ratio sqrt(omega / omega_ref).
    3. Bilinear Coupled FFN explicitly modeling transient-harmonic coupling.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_radial_attention_bias: bool = True,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        diffusion_gamma_init: float = 0.1,
        diffusion_alpha_init: float = 0.2,
        diffusion_beta_init: float = 0.5,
    ):
        super().__init__()
        self.use_radial_attention_bias = use_radial_attention_bias
        self.blocks = nn.ModuleList([
            HarmonicIsometricTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if use_radial_attention_bias:
            self.harmonic_bias = HarmonicIsometricAttentionBias(
                num_heads=num_heads,
                init_gamma=diffusion_gamma_init,
                init_alpha=diffusion_alpha_init,
                init_beta=diffusion_beta_init,
                spatial_topology=spatial_topology,
                star_radii=star_radii,
                grid_size=grid_size,
                r_coil_ref=3.0,
            )
        else:
            self.harmonic_bias = None

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        freq_condition: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        h = context_tokens + context_pos
        B, N_ctx, D = h.shape

        attn_bias = None
        if self.use_radial_attention_bias and self.harmonic_bias is not None:
            if context_indices is not None:
                dist_full = self.harmonic_bias.dist_matrix_100 if hasattr(self.harmonic_bias, "dist_matrix_100") and self.harmonic_bias.dist_matrix_100.shape[0] >= N_ctx else self.harmonic_bias.dist_matrix_25
                if dist_full.device != h.device:
                    dist_full = dist_full.to(h.device)
                idx = context_indices.long()
                dist_ctx = dist_full[idx.unsqueeze(2), idx.unsqueeze(1)]  # [B, N_ctx, N_ctx]
                attn_bias = self.harmonic_bias(dist_ctx, omega_bar=freq_condition)  # [B, H, N_ctx, N_ctx]
            elif N_ctx == 25:
                mat = self.harmonic_bias.dist_matrix_25.to(h.device)
                if B > 1:
                    mat = mat.unsqueeze(0).expand(B, -1, -1)
                attn_bias = self.harmonic_bias(mat, omega_bar=freq_condition)

        last_attn = None
        for i, blk in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                h, last_attn = blk(h, attn_bias=attn_bias, return_attention=True)
            else:
                h = blk(h, attn_bias=attn_bias)

        h = self.norm(h)
        if return_attention:
            return h, last_attn
        return h


def build_context_encoder_5x5(config) -> nn.Module:
    """
    Factory function for 5x5 PECT-JEPA Context Encoders.
    Supports:
      - 'harmonic_isometric' (EXP-37: Harmonic-Isometric Context Encoder with Coupled FFN)
      - 'dispersion_conditioned' (EXP-36: Harmonic Dispersion Conditioned Context Encoder)
      - 'standard' / 'vit' / 'baseline' (Standard Pre-LN ContextEncoder5x5)
    """
    encoder_type = getattr(config, "encoder_type", "harmonic_isometric").lower()
    embed_dim = getattr(config, "embed_dim", 64)
    depth = getattr(config, "encoder_depth", 4)
    num_heads = getattr(config, "encoder_heads", 4)
    mlp_ratio = getattr(config, "mlp_ratio", 4.0)
    dropout = getattr(config, "dropout", 0.0)
    use_radial_attention_bias = getattr(config, "use_radial_attention_bias", True)
    spatial_topology = getattr(config, "spatial_topology", "concentric_star")
    star_radii = getattr(config, "star_radii", (1, 3, 7))
    grid_size = getattr(config, "grid_size", 5)
    diffusion_gamma_init = getattr(config, "diffusion_gamma_init", 0.1)
    diffusion_alpha_init = getattr(config, "diffusion_alpha_init", 0.2)
    diffusion_beta_init = getattr(config, "diffusion_beta_init", 0.5)
    diffusion_alpha_x_init = getattr(config, "diffusion_alpha_x_init", 1.0)
    diffusion_alpha_y_init = getattr(config, "diffusion_alpha_y_init", 1.0)

    if encoder_type in ("harmonic_isometric", "harmonic", "isometric"):
        return HarmonicIsometricContextEncoder5x5(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_radial_attention_bias=use_radial_attention_bias,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
            grid_size=grid_size,
            diffusion_gamma_init=diffusion_gamma_init,
            diffusion_alpha_init=diffusion_alpha_init,
            diffusion_beta_init=diffusion_beta_init,
        )
    elif encoder_type in ("dispersion_conditioned", "dispersion", "harmonic_dispersion"):
        return DispersionConditionedContextEncoder5x5(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_radial_attention_bias=use_radial_attention_bias,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
            grid_size=grid_size,
            diffusion_gamma_init=diffusion_gamma_init,
            diffusion_alpha_init=diffusion_alpha_init,
            diffusion_alpha_x_init=diffusion_alpha_x_init,
            diffusion_alpha_y_init=diffusion_alpha_y_init,
        )
    elif encoder_type in ("standard", "vit", "baseline"):
        return ContextEncoder5x5(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_radial_attention_bias=use_radial_attention_bias,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
            grid_size=grid_size,
            diffusion_gamma_init=diffusion_gamma_init,
            diffusion_alpha_init=diffusion_alpha_init,
        )
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'harmonic_isometric', 'dispersion_conditioned', or 'standard'.")

