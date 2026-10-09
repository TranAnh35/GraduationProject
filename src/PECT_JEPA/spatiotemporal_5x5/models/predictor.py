"""
Predictor for 5x5 PECT-JEPA.
Predicts target latent representations from target queries (mask_token + target_pos)
and context key/value representations (H_context). Zero information leakage.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from .attention import MultiheadSelfAttention, MultiheadCrossAttention, MLP
from ..data.topologies import get_spatial_topology_offsets


class PredictorBlock(nn.Module):
    """
    Predictor Transformer Block:
    1. Self-Attention among Target Queries
    2. Cross-Attention from Target Queries to H_context
    3. Feed-Forward MLP
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(
        self,
        target_queries: torch.Tensor,
        H_context: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention
        q = target_queries + self.self_attn(self.norm_self(target_queries))
        # 2. Cross-Attention to Context with optional Helmholtz diffusion attention bias
        q = q + self.cross_attn(
            query=self.norm_cross_q(q),
            key_value=self.norm_cross_kv(H_context),
            attn_bias=attn_bias,
        )
        # 3. MLP
        q = q + self.mlp(self.norm_mlp(q))
        return q


class DiffusionOperatorEmbedding(nn.Module):
    """
    Continuous Fourier Feature Embedding for Eddy Current Diffusion Operator:
    Encodes harmonic frequency omega_k / skin depth delta(omega_k) = sqrt(2 / (omega * mu * sigma))
    into an action condition vector q_diff in R^D.
    """
    def __init__(self, embed_dim: int = 64, num_freq_bands: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        # 2 * num_freq_bands features (sin and cos)
        self.mlp = nn.Sequential(
            nn.Linear(num_freq_bands * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Learnable fallback base operator embedding
        self.default_op = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.default_op, std=0.02)

    def forward(self, freq_val: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            freq_val: [B] or [B, 1] normalized frequencies in range (0, 1]
        Returns:
            q_diff: [B, 1, embed_dim]
        """
        if freq_val is None:
            return self.default_op

        if freq_val.ndim == 1:
            freq_val = freq_val.unsqueeze(1)  # [B, 1]

        # Multi-scale Fourier features: 2^m * pi * freq
        device = freq_val.device
        scales = 2.0 ** torch.arange(self.num_freq_bands, device=device, dtype=torch.float32)  # [M]
        args = freq_val.float() * scales.unsqueeze(0) * torch.pi  # [B, M]

        fourier_feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*M]
        q_diff = self.mlp(fourier_feats).unsqueeze(1) + self.default_op  # [B, 1, embed_dim]
        return q_diff


class Predictor5x5(nn.Module):
    """
    Standard Predictor Network:
    Input: H_context [B, N_ctx, D], target_pos [B, N_tgt, D]
    Output: H_pred [B, N_tgt, D]
    """
    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        diffusion_operator: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, N_tgt, D = target_pos.shape
        queries = self.mask_token.expand(B, N_tgt, -1) + target_pos
        if diffusion_operator is not None:
            if diffusion_operator.ndim == 2:
                diffusion_operator = diffusion_operator.unsqueeze(1)
            queries = queries + diffusion_operator

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        return self.norm(q)


class OperatorDiffusionPredictor5x5(Predictor5x5):
    """
    Physics Neural Operator Predictor for PECT-JEPA:
    1. Conditions target queries on continuous Fourier features of the characteristic
       electromagnetic frequency omega_bar derived from the excitation and eddy current response:
           q_diff(omega_bar) in R^D
    2. Modulates Cross-Attention via Green's Function Diffusion Attention Bias:
           M_{ij}^{diff} = - gamma * dist_{ij} * sqrt(omega_bar) - beta * |s_i - s_j|
       governed by the fundamental solution of the Helmholtz diffusion equation:
           nabla^2 B = j * omega * mu * sigma * B  ==> G(r) ~ exp(-r / delta(omega))
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_freq_bins: int = 14,
        gamma_init: float = 1.0,
        beta_init: float = 0.5,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.num_freq_bins = num_freq_bins
        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.op_embedding = DiffusionOperatorEmbedding(embed_dim=embed_dim)

        # Learnable physical coupling parameters (strictly non-negative via softplus)
        raw_gamma = math.log(math.exp(gamma_init) - 1.0) if gamma_init > 0 else 0.0
        raw_beta = math.log(math.exp(beta_init) - 1.0) if beta_init > 0 else 0.0
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma, dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.tensor(raw_beta, dtype=torch.float32))

        self._register_distance_tables()

    def _register_distance_tables(self):
        if self.spatial_topology in ("concentric_star", "star", "octagram", "star_25"):
            offsets = get_spatial_topology_offsets(
                topology=self.spatial_topology,
                star_radii=self.star_radii
            )  # [25, 2] in physical mm
        else:
            offsets = np.array([(float(k // 5), float(k % 5)) for k in range(25)], dtype=np.float32)

        # 50 tokens: 25 spatial points * 2 diffusion scales (shallow=0, deep=1)
        coords_50 = []
        scales_50 = []
        for k in range(50):
            sp = k // 2
            coords_50.append((float(offsets[sp, 0]), float(offsets[sp, 1])))
            scales_50.append(float(k % 2))

        coords_50_t = torch.tensor(coords_50, dtype=torch.float32)  # [50, 2] in mm
        scales_50_t = torch.tensor(scales_50, dtype=torch.float32)  # [50]

        diff_50 = coords_50_t.unsqueeze(1) - coords_50_t.unsqueeze(0)  # [50, 50, 2]
        dist_50 = torch.norm(diff_50, p=2, dim=-1)  # [50, 50] physical distance in mm
        scale_diff_50 = torch.abs(scales_50_t.unsqueeze(1) - scales_50_t.unsqueeze(0))  # [50, 50]

        self.register_buffer("dist_table_50", dist_50, persistent=False)
        self.register_buffer("scale_diff_table_50", scale_diff_50, persistent=False)

        # 25 tokens: 25 spatial points * 1 scale
        coords_25 = [(float(offsets[k, 0]), float(offsets[k, 1])) for k in range(25)]
        coords_25_t = torch.tensor(coords_25, dtype=torch.float32)  # [25, 2] in mm
        diff_25 = coords_25_t.unsqueeze(1) - coords_25_t.unsqueeze(0)
        dist_25 = torch.norm(diff_25, p=2, dim=-1)
        scale_diff_25 = torch.zeros(25, 25, dtype=torch.float32)

        self.register_buffer("dist_table_25", dist_25, persistent=False)
        self.register_buffer("scale_diff_table_25", scale_diff_25, persistent=False)

        # 100 tokens: 25 spatial points * 4 temporal diffusion stages
        coords_100 = []
        scales_100 = []
        for k in range(100):
            sp = k // 4
            coords_100.append((float(offsets[sp, 0]), float(offsets[sp, 1])))
            scales_100.append(float(k % 4))

        coords_100_t = torch.tensor(coords_100, dtype=torch.float32)  # [100, 2] in mm
        scales_100_t = torch.tensor(scales_100, dtype=torch.float32)  # [100]

        diff_100 = coords_100_t.unsqueeze(1) - coords_100_t.unsqueeze(0)  # [100, 100, 2]
        dist_100 = torch.norm(diff_100, p=2, dim=-1)  # [100, 100]
        scale_diff_100 = torch.abs(scales_100_t.unsqueeze(1) - scales_100_t.unsqueeze(0))  # [100, 100]

        self.register_buffer("dist_table_100", dist_100, persistent=False)
        self.register_buffer("scale_diff_table_100", scale_diff_100, persistent=False)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            H_context: [B, N_ctx, D]
            target_pos: [B, N_tgt, D]
            context_indices: optional [B, N_ctx] token indices into the 100-token, 50-token, or 25-token grid
            target_indices: optional [B, N_tgt] token indices into the grid
            freq_condition: [B] normalized characteristic frequency in (0, 1]
            diffusion_operator: optional explicit [B, 1, D] condition
        """
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Operator Query Condition
        if diffusion_operator is None:
            if freq_condition is not None:
                if freq_condition.dtype in (torch.int32, torch.int64):
                    freq_norm = freq_condition.float() / float(self.num_freq_bins)
                else:
                    freq_norm = freq_condition.float()
                op_cond = self.op_embedding(freq_norm)
            else:
                op_cond = self.op_embedding.default_op
        else:
            op_cond = diffusion_operator

        # 2. Helmholtz Green's Function Diffusion Cross-Attention Bias
        attn_bias = None
        if context_indices is not None and target_indices is not None:
            max_idx = max(int(target_indices.max().item()), int(context_indices.max().item()))
            if max_idx >= 50:
                dist_table = self.dist_table_100
                scale_table = self.scale_diff_table_100
            elif max_idx >= 25:
                dist_table = self.dist_table_50
                scale_table = self.scale_diff_table_50
            else:
                dist_table = self.dist_table_25
                scale_table = self.scale_diff_table_25

            t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
            c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]
            dist = dist_table[t_idx, c_idx]                   # [B, N_tgt, N_ctx]
            scale_diff = scale_table[t_idx, c_idx]           # [B, N_tgt, N_ctx]

            if freq_condition is not None:
                if freq_condition.dtype in (torch.int32, torch.int64):
                    f_val = freq_condition.float() / float(self.num_freq_bins)
                else:
                    f_val = freq_condition.float()
                sqrt_freq = torch.sqrt(f_val.clamp(min=1e-4)).view(B, 1, 1).to(device)
            else:
                sqrt_freq = torch.ones(B, 1, 1, device=device)

            gamma = F.softplus(self.raw_gamma)
            beta = F.softplus(self.raw_beta)

            # M_{ij}^{diff} = - gamma * dist_{ij} * sqrt(omega) - beta * |s_i - s_j| <= 0
            M_diff = - gamma * (dist * sqrt_freq) - beta * scale_diff  # [B, N_tgt, N_ctx]
            attn_bias = M_diff.unsqueeze(1)  # [B, 1, N_tgt, N_ctx]

        return super().forward(H_context, target_pos, diffusion_operator=op_cond, attn_bias=attn_bias)


class ParabolicDiffusionPredictor5x5(Predictor5x5):
    """
    Parabolic Green's Function Diffusion Predictor for CST-Masking (100 tokens: 25 probes x 4 stages).

    1. Context-Conditioned Query Initialization:
       Grounds target queries in the observed excitation field baseline bar{H}_ctx(tau=0)
       rather than an uninformative static mask token:
           q_i = Linear(bar{H}_ctx(tau=0)) + target_pos_i + q_diff(omega_bar)

    2. Parabolic Spatiotemporal Green's Diffusion Attention Bias:
       Exact heat kernel fundamental solution for nabla^2 B = mu * sigma * d(B)/dt:
           G(Delta r, Delta tau) ~ (4 * pi * D * Delta tau)^(-3/2) * exp(- ||Delta r||^2 / (4 * D * Delta tau))
       Attention Log-Bias:
           M_{ij}^{diff} = - gamma(omega_bar) * ||Delta r_{ij}||^2 / Delta tau_{ij} - alpha * log(Delta tau_{ij})
       for Delta tau_{ij} = tau_tgt - tau_ctx > 0, and -10000.0 (strictly causal) otherwise.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_freq_bins: int = 14,
        gamma_init: float = 1.0,
        alpha_init: float = 0.5,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.num_freq_bins = num_freq_bins
        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.op_embedding = DiffusionOperatorEmbedding(embed_dim=embed_dim)

        # Context-conditioned query baseline projection
        self.ctx_init_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_ctx_init = nn.LayerNorm(embed_dim)

        # Learnable physical coupling parameters (strictly non-negative via softplus)
        raw_gamma = math.log(math.exp(gamma_init) - 1.0) if gamma_init > 0 else 0.0
        raw_alpha = math.log(math.exp(alpha_init) - 1.0) if alpha_init > 0 else 0.0
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))

        self._register_parabolic_tables()

    def _register_parabolic_tables(self):
        offsets = get_spatial_topology_offsets(
            topology=self.spatial_topology,
            star_radii=self.star_radii
        )  # [25, 2] in physical mm

        coords_100 = []
        stages_100 = []
        for k in range(100):
            sp = k // 4
            coords_100.append((float(offsets[sp, 0]), float(offsets[sp, 1])))
            stages_100.append(float(k % 4))

        coords_100_t = torch.tensor(coords_100, dtype=torch.float32)  # [100, 2] in mm
        stages_100_t = torch.tensor(stages_100, dtype=torch.float32)  # [100]

        diff_100 = coords_100_t.unsqueeze(1) - coords_100_t.unsqueeze(0)  # [100, 100, 2]
        dist_sq_100 = torch.sum(diff_100 ** 2, dim=-1)  # [100, 100] physical squared distance in mm^2
        delta_tau_100 = stages_100_t.unsqueeze(1) - stages_100_t.unsqueeze(0)  # [100, 100]

        self.register_buffer("dist_sq_table_100", dist_sq_100, persistent=False)
        self.register_buffer("delta_tau_table_100", delta_tau_100, persistent=False)

        # 25 spatial tokens table in physical mm
        coords_25_t = torch.from_numpy(offsets).float()  # [25, 2] in mm
        diff_25 = coords_25_t.unsqueeze(1) - coords_25_t.unsqueeze(0)  # [25, 25, 2]
        dist_sq_25 = torch.sum(diff_25 ** 2, dim=-1)  # [25, 25] in mm^2
        self.register_buffer("dist_sq_table_25", dist_sq_25, persistent=False)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Operator Query Condition
        if diffusion_operator is None:
            if freq_condition is not None:
                if freq_condition.dtype in (torch.int32, torch.int64):
                    freq_norm = freq_condition.float() / float(self.num_freq_bins)
                else:
                    freq_norm = freq_condition.float()
                op_cond = self.op_embedding(freq_norm)
            else:
                op_cond = self.op_embedding.default_op
        else:
            op_cond = diffusion_operator

        # 2. Context-Conditioned Query Initialization
        if context_indices is not None and context_indices.max() >= 50:
            mask_t0 = (context_indices % 4 == 0).to(device)  # [B, N_ctx]
            denom = mask_t0.sum(dim=-1, keepdim=True).clamp(min=1).unsqueeze(-1)
            H_ctx_t0 = (H_context * mask_t0.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B, 1, D]
        else:
            H_ctx_t0 = H_context.mean(dim=1, keepdim=True)

        q_baseline = self.norm_ctx_init(self.ctx_init_proj(H_ctx_t0))  # [B, 1, D]
        queries = q_baseline.expand(B, N_tgt, -1) + self.mask_token.expand(B, N_tgt, -1) + target_pos
        if op_cond is not None:
            if op_cond.ndim == 2:
                op_cond = op_cond.unsqueeze(1)
            queries = queries + op_cond

        # 3. Exact Parabolic Green's Function Attention Bias
        attn_bias = None
        if context_indices is not None and target_indices is not None:
            is_100_tokens = (
                hasattr(self, "dist_sq_table_100")
                and (target_indices.max() >= 25 or context_indices.max() >= 25)
            )
            if is_100_tokens:
                t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
                c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]

                d_sq = self.dist_sq_table_100[t_idx, c_idx]      # [B, N_tgt, N_ctx]
                d_tau = self.delta_tau_table_100[t_idx, c_idx]   # [B, N_tgt, N_ctx] (tau_tgt - tau_ctx)

                if freq_condition is not None:
                    if freq_condition.dtype in (torch.int32, torch.int64):
                        f_val = freq_condition.float() / float(self.num_freq_bins)
                    else:
                        f_val = freq_condition.float()
                    sqrt_freq = torch.sqrt(f_val.clamp(min=1e-4)).view(B, 1, 1).to(device)
                else:
                    sqrt_freq = torch.ones(B, 1, 1, device=device)

                gamma = F.softplus(self.raw_gamma) * sqrt_freq
                alpha = F.softplus(self.raw_alpha)

                causal_mask = (d_tau > 0)
                has_causal = causal_mask.any(dim=-1, keepdim=True)
                safe_d_tau = torch.clamp(d_tau.abs(), min=1.0)

                # M_{ij}^{diff} = - gamma * (||Delta r||^2 / Delta tau) - alpha * log(Delta tau)
                M_diff = - gamma * (d_sq / safe_d_tau) - alpha * torch.log(safe_d_tau)
                M_diff = torch.where(
                    causal_mask | (~has_causal),
                    M_diff,
                    torch.tensor(-10000.0, device=device, dtype=M_diff.dtype)
                )
                attn_bias = M_diff.unsqueeze(1)  # [B, 1, N_tgt, N_ctx]
            else:
                # 25 spatial tokens mode
                t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
                c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]
                d_sq = self.dist_sq_table_25[t_idx, c_idx]       # [B, N_tgt, N_ctx]
                gamma = F.softplus(self.raw_gamma)
                alpha = F.softplus(self.raw_alpha)
                safe_r = torch.clamp(d_sq, min=1.0)
                M_spatial = - gamma * d_sq - alpha * torch.log(safe_r)
                attn_bias = M_spatial.unsqueeze(1)               # [B, 1, N_tgt, N_ctx]

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        return self.norm(q)


class ResidualDiffusionPredictor5x5(ParabolicDiffusionPredictor5x5):
    """
    Residual Diffusion Predictor for PECT-JEPA.

    Instead of predicting the full macro target token H_tgt (which is 98% dominated
    by the primary excitation field decay), this predictor models the DYNAMIC DIFFUSION RESIDUAL:
        Delta H_tgt(tau) = H_tgt(tau) - h_base(tau=0)
    
    1. Extracts the early excitation baseline h_base from tau=0 context tokens.
    2. Conditions target queries on h_base + target position + Green's parabolic diffusion bias.
    3. Outputs the dynamic residual Delta H_pred in representation space.
    4. Full predicted representation: H_pred = h_base + Delta H_pred.
    
    This mathematically decouples the static excitation field from the defect interaction,
    enforcing that prediction capacity is dedicated 100% to eddy current diffusion and defect sizing.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_freq_bins: int = 14,
        gamma_init: float = 1.0,
        alpha_init: float = 0.5,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_freq_bins=num_freq_bins,
            gamma_init=gamma_init,
            alpha_init=alpha_init,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
        )
        # Dedicated residual projection head
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.last_h_base: Optional[torch.Tensor] = None
        self.last_delta_pred: Optional[torch.Tensor] = None

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Operator Query Condition
        if diffusion_operator is None:
            if freq_condition is not None:
                if freq_condition.dtype in (torch.int32, torch.int64):
                    freq_norm = freq_condition.float() / float(self.num_freq_bins)
                else:
                    freq_norm = freq_condition.float()
                op_cond = self.op_embedding(freq_norm)
            else:
                op_cond = self.op_embedding.default_op
        else:
            op_cond = diffusion_operator

        # 2. Extract Early Context Baseline (tau = 0)
        if context_indices is not None and context_indices.max() >= 50:
            mask_t0 = (context_indices % 4 == 0).to(device)  # [B, N_ctx]
            denom = mask_t0.sum(dim=-1, keepdim=True).clamp(min=1).unsqueeze(-1)
            H_ctx_t0 = (H_context * mask_t0.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # [B, 1, D]
        else:
            H_ctx_t0 = H_context.mean(dim=1, keepdim=True)

        h_base = self.norm_ctx_init(self.ctx_init_proj(H_ctx_t0))  # [B, 1, D]
        queries = h_base.expand(B, N_tgt, -1) + self.mask_token.expand(B, N_tgt, -1) + target_pos
        if op_cond is not None:
            if op_cond.ndim == 2:
                op_cond = op_cond.unsqueeze(1)
            queries = queries + op_cond

        # 3. Exact Parabolic Green's Function Attention Bias
        attn_bias = None
        if context_indices is not None and target_indices is not None:
            is_100_tokens = (
                hasattr(self, "dist_sq_table_100")
                and (target_indices.max() >= 25 or context_indices.max() >= 25)
            )
            if is_100_tokens:
                t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
                c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]

                d_sq = self.dist_sq_table_100[t_idx, c_idx]      # [B, N_tgt, N_ctx]
                d_tau = self.delta_tau_table_100[t_idx, c_idx]   # [B, N_tgt, N_ctx]

                if freq_condition is not None:
                    if freq_condition.dtype in (torch.int32, torch.int64):
                        f_val = freq_condition.float() / float(self.num_freq_bins)
                    else:
                        f_val = freq_condition.float()
                    sqrt_freq = torch.sqrt(f_val.clamp(min=1e-4)).view(B, 1, 1).to(device)
                else:
                    sqrt_freq = torch.ones(B, 1, 1, device=device)

                gamma = F.softplus(self.raw_gamma) * sqrt_freq
                alpha = F.softplus(self.raw_alpha)

                causal_mask = (d_tau > 0)
                has_causal = causal_mask.any(dim=-1, keepdim=True)
                safe_d_tau = torch.clamp(d_tau.abs(), min=1.0)

                # Green's Attention Bias
                M_diff = - gamma * (d_sq / safe_d_tau) - alpha * torch.log(safe_d_tau)
                M_diff = torch.where(
                    causal_mask | (~has_causal),
                    M_diff,
                    torch.tensor(-10000.0, device=device, dtype=M_diff.dtype)
                )
                attn_bias = M_diff.unsqueeze(1)  # [B, 1, N_tgt, N_ctx]
            else:
                # 25 spatial tokens mode
                t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
                c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]
                d_sq = self.dist_sq_table_25[t_idx, c_idx]       # [B, N_tgt, N_ctx]
                gamma = F.softplus(self.raw_gamma)
                alpha = F.softplus(self.raw_alpha)
                safe_r = torch.clamp(d_sq, min=1.0)
                M_spatial = - gamma * d_sq - alpha * torch.log(safe_r)
                attn_bias = M_spatial.unsqueeze(1)               # [B, 1, N_tgt, N_ctx]

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        delta_pred = self.residual_head(self.norm(q))  # [B, N_tgt, D]
        H_pred = h_base.expand(B, N_tgt, -1) + delta_pred  # [B, N_tgt, D]

        self.last_h_base = h_base
        self.last_delta_pred = delta_pred

        if return_residual:
            return H_pred, delta_pred, h_base
        return H_pred


class ContinuousHelmholtzPredictor5x5(Predictor5x5):
    """
    Continuous Green's Helmholtz Operator Predictor for 5x5 PECT-JEPA (EXP-18).
    
    Models 3D eddy current electromagnetic scattering governed by the vector Helmholtz equation:
        nabla^2 B - k^2 B = 0, where k = (1 + j) / delta(omega)
    
    1. Continuous Helmholtz Propagator:
       Propagates visible context representations H_context into target query locations via
       analytical Green's function in 3D half-space:
           G(Delta r, Delta z; omega) = exp(-(1+j) * sqrt(||Delta r||^2 + (Delta z * d_scale)^2) / delta(omega))
                                       / (sqrt(||Delta r||^2 + (Delta z * d_scale)^2) + eps)
       Physical baseline:
           H_helmholtz_i = sum_j Softmax(log |G_ij|) * LayerNorm(W_prop * H_ctx_j)
    
    2. Dynamic Perturbation Residual Head:
       Refines target queries through Transformer cross-attention blocks with Helmholtz log-bias:
           M_ij^{diff} = - gamma * sqrt(omega) * R_ij - alpha * log(R_ij + eps)
       and outputs the dynamic defect scattering perturbation:
           Delta H_pred = MLP(q)
           H_pred = H_helmholtz + Delta H_pred
       
       On sound metal: Delta H_pred -> 0 (pure Helmholtz propagation).
       On flaws: Delta H_pred directly encodes flaw scattering, monotonic in flaw depth.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_freq_bins: int = 14,
        gamma_init: float = 1.0,
        alpha_init: float = 0.5,
        d_scale_init: float = 1.5,
        d_scale_min: float = 1.0,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.num_freq_bins = num_freq_bins
        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.d_scale_min = d_scale_min
        self.op_embedding = DiffusionOperatorEmbedding(embed_dim=embed_dim)

        # Context propagation projection
        self.w_prop = nn.Linear(embed_dim, embed_dim)
        self.norm_prop = nn.LayerNorm(embed_dim)

        # Learnable physical coupling parameters
        raw_gamma = math.log(math.exp(gamma_init) - 1.0) if gamma_init > 0 else 0.0
        raw_alpha = math.log(math.exp(alpha_init) - 1.0) if alpha_init > 0 else 0.0
        init_delta = max(d_scale_init - d_scale_min, 0.05)
        raw_d_scale = math.log(math.exp(init_delta) - 1.0)
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))
        self.raw_d_scale = nn.Parameter(torch.tensor(raw_d_scale, dtype=torch.float32))

        # Dynamic defect scattering perturbation head
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.last_h_base: Optional[torch.Tensor] = None
        self.last_delta_pred: Optional[torch.Tensor] = None

        self._register_helmholtz_tables()

    def _register_helmholtz_tables(self):
        try:
            offsets = get_spatial_topology_offsets(
                topology=self.spatial_topology,
                star_radii=self.star_radii
            )
        except Exception:
            offsets = np.array([(float(k // 5), float(k % 5)) for k in range(25)], dtype=np.float32)

        # 50 tokens: 25 spatial points * 2 diffusion scales (shallow=0.0, deep=1.0)
        coords_xy_50 = []
        depth_z_50 = []
        for k in range(50):
            sp = k // 2
            coords_xy_50.append((float(offsets[sp, 0]), float(offsets[sp, 1])))
            depth_z_50.append(float(k % 2))

        coords_xy_50_t = torch.tensor(coords_xy_50, dtype=torch.float32)  # [50, 2] in mm
        depth_z_50_t = torch.tensor(depth_z_50, dtype=torch.float32)      # [50]

        diff_xy_50 = coords_xy_50_t.unsqueeze(1) - coords_xy_50_t.unsqueeze(0)  # [50, 50, 2]
        dist_xy_sq_50 = torch.sum(diff_xy_50 ** 2, dim=-1)                      # [50, 50] mm^2
        diff_z_50 = depth_z_50_t.unsqueeze(1) - depth_z_50_t.unsqueeze(0)       # [50, 50]
        dz_sq_50 = diff_z_50 ** 2                                               # [50, 50]

        self.register_buffer("dist_xy_sq_50", dist_xy_sq_50, persistent=False)
        self.register_buffer("dz_sq_50", dz_sq_50, persistent=False)

        # 25 tokens: 25 spatial points * 1 scale (z=0)
        coords_25_t = torch.from_numpy(offsets).float()  # [25, 2]
        diff_xy_25 = coords_25_t.unsqueeze(1) - coords_25_t.unsqueeze(0)
        dist_xy_sq_25 = torch.sum(diff_xy_25 ** 2, dim=-1)
        dz_sq_25 = torch.zeros(25, 25, dtype=torch.float32)

        self.register_buffer("dist_xy_sq_25", dist_xy_sq_25, persistent=False)
        self.register_buffer("dz_sq_25", dz_sq_25, persistent=False)

        # 100 tokens: 25 spatial points * 4 temporal diffusion stages (z in {0, 1, 2, 3})
        coords_xy_100 = []
        depth_z_100 = []
        for k in range(100):
            sp = k // 4
            coords_xy_100.append((float(offsets[sp, 0]), float(offsets[sp, 1])))
            depth_z_100.append(float(k % 4) / 3.0)  # normalized 0.0 .. 1.0

        coords_xy_100_t = torch.tensor(coords_xy_100, dtype=torch.float32)
        depth_z_100_t = torch.tensor(depth_z_100, dtype=torch.float32)

        diff_xy_100 = coords_xy_100_t.unsqueeze(1) - coords_xy_100_t.unsqueeze(0)
        dist_xy_sq_100 = torch.sum(diff_xy_100 ** 2, dim=-1)
        diff_z_100 = depth_z_100_t.unsqueeze(1) - depth_z_100_t.unsqueeze(0)
        dz_sq_100 = diff_z_100 ** 2

        self.register_buffer("dist_xy_sq_100", dist_xy_sq_100, persistent=False)
        self.register_buffer("dz_sq_100", dz_sq_100, persistent=False)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Operator Query Condition
        if diffusion_operator is None:
            if freq_condition is not None:
                if freq_condition.dtype in (torch.int32, torch.int64):
                    freq_norm = freq_condition.float() / float(self.num_freq_bins)
                else:
                    freq_norm = freq_condition.float()
                op_cond = self.op_embedding(freq_norm)
            else:
                op_cond = self.op_embedding.default_op
        else:
            op_cond = diffusion_operator

        # 2. Compute 3D Physical Helmholtz Propagator
        gamma = F.softplus(self.raw_gamma)
        alpha = F.softplus(self.raw_alpha)
        d_scale = F.softplus(self.raw_d_scale) + self.d_scale_min

        if freq_condition is not None:
            if freq_condition.dtype in (torch.int32, torch.int64):
                f_val = freq_condition.float() / float(self.num_freq_bins)
            else:
                f_val = freq_condition.float()
            sqrt_freq = torch.sqrt(f_val.clamp(min=1e-4)).view(B, 1, 1).to(device)
        else:
            sqrt_freq = torch.ones(B, 1, 1, device=device)

        eff_gamma = gamma * sqrt_freq  # [B, 1, 1]

        if context_indices is not None and target_indices is not None:
            max_idx = max(int(target_indices.max().item()), int(context_indices.max().item()))
            if max_idx >= 50:
                dist_xy_sq = self.dist_xy_sq_100
                dz_sq = self.dz_sq_100
            elif max_idx >= 25:
                dist_xy_sq = self.dist_xy_sq_50
                dz_sq = self.dz_sq_50
            else:
                dist_xy_sq = self.dist_xy_sq_25
                dz_sq = self.dz_sq_25

            t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
            c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]

            d_xy2 = dist_xy_sq[t_idx, c_idx]                 # [B, N_tgt, N_ctx]
            d_z2 = dz_sq[t_idx, c_idx]                       # [B, N_tgt, N_ctx]
            R_sq = d_xy2 + d_z2 * (d_scale ** 2)
            R = torch.sqrt(R_sq + 1e-4)                      # [B, N_tgt, N_ctx]

            # Continuous Green's Helmholtz Kernel
            log_G = - eff_gamma * R - torch.log(R + 0.1)     # [B, N_tgt, N_ctx]
            weights = F.softmax(log_G, dim=-1)                # [B, N_tgt, N_ctx]

            # Context Propagation
            H_ctx_proj = self.norm_prop(self.w_prop(H_context))  # [B, N_ctx, D]
            H_helmholtz = torch.bmm(weights, H_ctx_proj)        # [B, N_tgt, D]

            # Cross-Attention Attention Bias
            M_diff = - eff_gamma * R - alpha * torch.log(R + 0.1)
            attn_bias = M_diff.unsqueeze(1)                     # [B, 1, N_tgt, N_ctx]
        else:
            H_ctx_mean = H_context.mean(dim=1, keepdim=True).expand(-1, N_tgt, -1)
            H_helmholtz = self.norm_prop(self.w_prop(H_ctx_mean))
            attn_bias = None

        # 3. Target Query Formulation
        queries = H_helmholtz + self.mask_token.expand(B, N_tgt, -1) + target_pos
        if op_cond is not None:
            if op_cond.ndim == 2:
                op_cond = op_cond.unsqueeze(1)
            queries = queries + op_cond

        # 4. Transformer Attention Refinement
        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        # 5. Dynamic Defect Scattering Perturbation Residual
        delta_pred = self.residual_head(self.norm(q))           # [B, N_tgt, D]
        H_pred = H_helmholtz + delta_pred                      # [B, N_tgt, D]

        self.last_h_base = H_helmholtz
        self.last_delta_pred = delta_pred

        if return_residual:
            return H_pred, delta_pred, H_helmholtz
        return H_pred


class AnisotropicDiffusionPredictor5x5(Predictor5x5):
    """
    Anisotropic Spatio-Diffusion Operator Predictor for 5x5 PECT-JEPA (EXP-20).

    Unifies:
    1. Continuous 3D analytical Helmholtz diffusion with strictly bounded thickness scale d_scale >= 1.0 mm.
    2. Learnable lateral spatial anisotropy (alpha_x, alpha_y) modeling directional coil sensitivities (Bx != By).
    3. Directional dipole projection head providing directional attention bias.
    4. Uncrushed 28D harmonic dispersion and physical baseline propagation.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_freq_bins: int = 14,
        gamma_init: float = 1.0,
        alpha_init: float = 0.5,
        alpha_x_init: float = 1.0,
        alpha_y_init: float = 1.0,
        d_scale_init: float = 1.5,
        d_scale_min: float = 1.0,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.num_freq_bins = num_freq_bins
        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.d_scale_min = d_scale_min
        self.op_embedding = DiffusionOperatorEmbedding(embed_dim=embed_dim)

        # Context propagation projection
        self.w_prop = nn.Linear(embed_dim, embed_dim)
        self.norm_prop = nn.LayerNorm(embed_dim)

        # Learnable physical coupling parameters
        raw_gamma = math.log(math.exp(gamma_init) - 1.0) if gamma_init > 0 else 0.0
        raw_alpha = math.log(math.exp(alpha_init) - 1.0) if alpha_init > 0 else 0.0
        raw_ax = math.log(math.exp(alpha_x_init) - 1.0) if alpha_x_init > 0 else 0.0
        raw_ay = math.log(math.exp(alpha_y_init) - 1.0) if alpha_y_init > 0 else 0.0
        init_delta = max(d_scale_init - d_scale_min, 0.05)
        raw_d_scale = math.log(math.exp(init_delta) - 1.0)

        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha, dtype=torch.float32))
        self.raw_alpha_x = nn.Parameter(torch.tensor(raw_ax, dtype=torch.float32))
        self.raw_alpha_y = nn.Parameter(torch.tensor(raw_ay, dtype=torch.float32))
        self.raw_d_scale = nn.Parameter(torch.tensor(raw_d_scale, dtype=torch.float32))

        # Directional dipole MLP: projects 2D normalized direction into attention bias
        self.dir_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Dynamic defect scattering perturbation head
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.last_h_base: Optional[torch.Tensor] = None
        self.last_delta_pred: Optional[torch.Tensor] = None

        self._register_anisotropic_tables()

    def _register_anisotropic_tables(self):
        try:
            offsets = get_spatial_topology_offsets(
                topology=self.spatial_topology,
                star_radii=self.star_radii
            )
        except Exception:
            offsets = np.array([(float(k // 5), float(k % 5)) for k in range(25)], dtype=np.float32)

        # 50 tokens: 25 spatial points * 2 diffusion scales
        coords_xy_50 = []
        depth_z_50 = []
        for k in range(50):
            sp = k // 2
            coords_xy_50.append((float(offsets[sp, 0]), float(offsets[sp, 1])))
            depth_z_50.append(float(k % 2))

        coords_xy_50_t = torch.tensor(coords_xy_50, dtype=torch.float32)  # [50, 2] in mm
        depth_z_50_t = torch.tensor(depth_z_50, dtype=torch.float32)      # [50]

        diff_xy_50 = coords_xy_50_t.unsqueeze(1) - coords_xy_50_t.unsqueeze(0)  # [50, 50, 2]
        dx_sq_50 = diff_xy_50[:, :, 0] ** 2                                      # [50, 50]
        dy_sq_50 = diff_xy_50[:, :, 1] ** 2                                      # [50, 50]
        diff_z_50 = depth_z_50_t.unsqueeze(1) - depth_z_50_t.unsqueeze(0)       # [50, 50]
        dz_sq_50 = diff_z_50 ** 2                                               # [50, 50]

        self.register_buffer("diff_xy_50", diff_xy_50, persistent=False)
        self.register_buffer("dx_sq_50", dx_sq_50, persistent=False)
        self.register_buffer("dy_sq_50", dy_sq_50, persistent=False)
        self.register_buffer("dz_sq_50", dz_sq_50, persistent=False)

        # 25 tokens: 25 spatial points * 1 scale (z=0)
        coords_25_t = torch.from_numpy(offsets).float()  # [25, 2]
        diff_xy_25 = coords_25_t.unsqueeze(1) - coords_25_t.unsqueeze(0)
        dx_sq_25 = diff_xy_25[:, :, 0] ** 2
        dy_sq_25 = diff_xy_25[:, :, 1] ** 2
        dz_sq_25 = torch.zeros(25, 25, dtype=torch.float32)

        self.register_buffer("diff_xy_25", diff_xy_25, persistent=False)
        self.register_buffer("dx_sq_25", dx_sq_25, persistent=False)
        self.register_buffer("dy_sq_25", dy_sq_25, persistent=False)
        self.register_buffer("dz_sq_25", dz_sq_25, persistent=False)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Operator Query Condition
        if diffusion_operator is None:
            if freq_condition is not None:
                if freq_condition.dtype in (torch.int32, torch.int64):
                    freq_norm = freq_condition.float() / float(self.num_freq_bins)
                else:
                    freq_norm = freq_condition.float()
                op_cond = self.op_embedding(freq_norm)
            else:
                op_cond = self.op_embedding.default_op
        else:
            op_cond = diffusion_operator

        # 2. Compute 3D Anisotropic Diffusion Propagator
        gamma = F.softplus(self.raw_gamma)
        alpha = F.softplus(self.raw_alpha)
        alpha_x = F.softplus(self.raw_alpha_x) + 0.1
        alpha_y = F.softplus(self.raw_alpha_y) + 0.1
        d_scale = F.softplus(self.raw_d_scale) + self.d_scale_min

        if freq_condition is not None:
            if freq_condition.dtype in (torch.int32, torch.int64):
                f_val = freq_condition.float() / float(self.num_freq_bins)
            else:
                f_val = freq_condition.float()
            sqrt_freq = torch.sqrt(f_val.clamp(min=1e-4)).view(B, 1, 1).to(device)
        else:
            sqrt_freq = torch.ones(B, 1, 1, device=device)

        eff_gamma = gamma * sqrt_freq  # [B, 1, 1]

        if context_indices is not None and target_indices is not None:
            max_idx = max(int(target_indices.max().item()), int(context_indices.max().item()))
            if max_idx >= 25:
                diff_xy = self.diff_xy_50
                dx_sq = self.dx_sq_50
                dy_sq = self.dy_sq_50
                dz_sq = self.dz_sq_50
            else:
                diff_xy = self.diff_xy_25
                dx_sq = self.dx_sq_25
                dy_sq = self.dy_sq_25
                dz_sq = self.dz_sq_25

            t_idx = target_indices.to(device).unsqueeze(-1)  # [B, N_tgt, 1]
            c_idx = context_indices.to(device).unsqueeze(1)   # [B, 1, N_ctx]

            dx2 = dx_sq[t_idx, c_idx]                        # [B, N_tgt, N_ctx]
            dy2 = dy_sq[t_idx, c_idx]                        # [B, N_tgt, N_ctx]
            dz2 = dz_sq[t_idx, c_idx]                        # [B, N_tgt, N_ctx]
            d_xy = diff_xy[t_idx, c_idx]                     # [B, N_tgt, N_ctx, 2]

            # Anisotropic Distance Metric R_A
            R_sq = alpha_x * dx2 + alpha_y * dy2 + dz2 * (d_scale ** 2)
            R = torch.sqrt(R_sq + 1e-4)                      # [B, N_tgt, N_ctx]

            # Continuous Green's Helmholtz Kernel
            log_G = - eff_gamma * R - torch.log(R + 0.1)     # [B, N_tgt, N_ctx]
            weights = F.softmax(log_G, dim=-1)                # [B, N_tgt, N_ctx]

            # Context Propagation
            H_ctx_proj = self.norm_prop(self.w_prop(H_context))  # [B, N_ctx, D]
            H_base = torch.bmm(weights, H_ctx_proj)             # [B, N_tgt, D]

            # Directional dipole attention bias
            u_dir = torch.cat([
                (torch.sqrt(alpha_x) * d_xy[..., 0:1]) / (R.unsqueeze(-1) + 1e-4),
                (torch.sqrt(alpha_y) * d_xy[..., 1:2]) / (R.unsqueeze(-1) + 1e-4),
            ], dim=-1)                                       # [B, N_tgt, N_ctx, 2]
            M_dir = self.dir_mlp(u_dir).squeeze(-1)          # [B, N_tgt, N_ctx]

            # Cross-Attention Attention Bias
            M_diff = - eff_gamma * R - alpha * torch.log(R + 0.1) + M_dir
            attn_bias = M_diff.unsqueeze(1)                  # [B, 1, N_tgt, N_ctx]
        else:
            H_ctx_mean = H_context.mean(dim=1, keepdim=True).expand(-1, N_tgt, -1)
            H_base = self.norm_prop(self.w_prop(H_ctx_mean))
            attn_bias = None

        # 3. Target Query Formulation
        queries = H_base + self.mask_token.expand(B, N_tgt, -1) + target_pos
        if op_cond is not None:
            if op_cond.ndim == 2:
                op_cond = op_cond.unsqueeze(1)
            queries = queries + op_cond

        # 4. Predictor Transformer Layers
        q = queries
        for block in self.blocks:
            q = block(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        # 5. Dynamic Perturbation Head
        delta_pred = self.residual_head(q)                   # [B, N_tgt, D]
        H_pred = H_base + delta_pred                         # [B, N_tgt, D]

        self.last_h_base = H_base
        self.last_delta_pred = delta_pred

        if return_residual:
            return H_pred, delta_pred, H_base
        return H_pred

class NeuralFieldSubspacePredictor5x5(Predictor5x5):
    """
    Data-Driven Neural Field Operator Predictor with Latent Subspace Decomposition (EXP-22).

    Evolution & Heritage:
      - Inherits the 25-token spatial cross-attention paradigm from early experiments (EXP-01..EXP-07).
      - Replaces hand-forced analytical isotropic Green's formulas with a learned spatial relative coordinate
        embedding MLP: e_ij = MLP(Delta x, Delta y, ||Delta r||), allowing data-driven transfer kernel learning.
      - Implements Latent Subspace Decomposition (Phys-JEPA arXiv:2606.16076 & SubspaceAD arXiv:2308.06733):
        Explicitly separates latent prediction into:
          1. Nominal Background Field (z_base) capturing common-mode plate/fastener baseline.
          2. Residual Scattering Perturbation (delta_pred) capturing localized flaw disturbances.
        Full prediction: H_pred = z_base + delta_pred.
      - Resolves the Fastener Clutter Paradox: prevents the 0.0577V fastener jump from drowning out
        the 0.0076V subsurface corrosion perturbation.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.grid_size = grid_size

        # Relative spatial coordinate embedding MLP: [dx, dy, dist] -> embed_dim
        self.rel_pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Context initialization projection
        self.ctx_init_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_ctx_init = nn.LayerNorm(embed_dim)

        # Dual-Head Latent Subspace Decomposition
        self.base_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Precompute spatial coordinate offsets [25, 2] in mm
        offsets = get_spatial_topology_offsets(
            topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
        )
        self.register_buffer("coords_25", torch.from_numpy(offsets).float(), persistent=False)

        self.last_h_base: Optional[torch.Tensor] = None
        self.last_delta_pred: Optional[torch.Tensor] = None

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Nominal Context Baseline
        h_ctx_mean = H_context.mean(dim=1, keepdim=True)  # [B, 1, D]
        h_base_init = self.norm_ctx_init(self.ctx_init_proj(h_ctx_mean))  # [B, 1, D]

        # 2. Target Query Formulation with Relative Position Conditioning
        queries = h_base_init.expand(B, N_tgt, -1) + self.mask_token.expand(B, N_tgt, -1) + target_pos

        # 3. Relative Coordinate Cross-Attention Bias
        attn_bias = None
        if context_indices is not None and target_indices is not None and hasattr(self, "coords_25"):
            coords = self.coords_25.to(device)  # [25, 2]
            c_idx = torch.clamp(context_indices.long(), 0, coords.shape[0] - 1)  # [B, N_ctx]
            t_idx = torch.clamp(target_indices.long(), 0, coords.shape[0] - 1)   # [B, N_tgt]

            pos_tgt = coords[t_idx]             # [B, N_tgt, 2]
            pos_ctx = coords[c_idx]             # [B, N_ctx, 2]

            delta_r = pos_tgt.unsqueeze(2) - pos_ctx.unsqueeze(1)  # [B, N_tgt, N_ctx, 2]
            dist_r = torch.sqrt(torch.sum(delta_r ** 2, dim=-1, keepdim=True) + 1e-4)  # [B, N_tgt, N_ctx, 1]

            rel_feat = torch.cat([delta_r, dist_r], dim=-1)  # [B, N_tgt, N_ctx, 3]
            rel_emb = self.rel_pos_mlp(rel_feat)             # [B, N_tgt, N_ctx, D]
            attn_bias = rel_emb.mean(dim=-1).unsqueeze(1)    # [B, 1, N_tgt, N_ctx]

        # 4. Transformer Refinement
        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        q = self.norm(q)

        # 5. Dual-Head Latent Subspace Decomposition
        H_base = self.base_head(q)
        delta_pred = self.residual_head(q)
        H_pred = H_base + delta_pred

        self.last_h_base = H_base
        self.last_delta_pred = delta_pred

        if return_residual:
            return H_pred, delta_pred, H_base
        return H_pred



class FrequencyConditionedDiffusionPredictor5x5(nn.Module):
    """
    Frequency-Conditioned Diffusion World Model Predictor for 5x5 PECT-JEPA (EXP-28).

    Physical Grounding:
      Eddy current spatial diffusion is fundamentally frequency-dependent:
          delta(omega) = sqrt(2 / (omega * mu * sigma)) ~ 1 / sqrt(omega)
      The spatial diffusion attenuation kernel decays over characteristic length scale delta(omega).
      High-frequency transient components (near-surface) decay rapidly with spatial radius r,
      while low-frequency components (deep penetration) diffuse broadly across the sensor array.

      This predictor computes a continuous relative coordinate & frequency cross-attention bias:
          rel_feat = [Delta x, Delta y, ||Delta r||, omega_char, ||Delta r|| * sqrt(omega_char)]
          attn_bias = MLP(rel_feat) in R^(B, num_heads, N_tgt, N_ctx)
      Where omega_char in (0, 1] is the normalized spectral centroid extracted purely unsupervised
      from the observed signal FFT power spectrum.

    Subspace Separation:
      Preserves the full-rank dual-head latent subspace decomposition:
          H_pred = H_base + Delta H_pred
      where H_base models the nominal incident diffusion carrier and Delta H_pred models
      the localized flaw scattering perturbation.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.grid_size = grid_size

        # Frequency-conditioned relative coordinate diffusion MLP:
        # Input features: [dx, dy, dist, freq_val, dist * sqrt(freq_val)] -> 5 features
        # Output: num_heads per-head attention biases
        self.rel_diff_mlp = nn.Sequential(
            nn.Linear(5, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_heads),
        )

        # Context initialization projection
        self.ctx_init_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_ctx_init = nn.LayerNorm(embed_dim)

        # Dual-Head Latent Subspace Decomposition
        self.base_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Precompute spatial coordinate offsets [25, 2] in mm
        offsets = get_spatial_topology_offsets(
            topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
        )
        self.register_buffer("coords_25", torch.from_numpy(offsets).float(), persistent=False)

        self.last_h_base: Optional[torch.Tensor] = None
        self.last_delta_pred: Optional[torch.Tensor] = None

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # 1. Nominal Context Baseline
        h_ctx_mean = H_context.mean(dim=1, keepdim=True)  # [B, 1, D]
        h_base_init = self.norm_ctx_init(self.ctx_init_proj(h_ctx_mean))  # [B, 1, D]

        # 2. Target Query Formulation with Relative Position Conditioning
        queries = h_base_init.expand(B, N_tgt, -1) + self.mask_token.expand(B, N_tgt, -1) + target_pos

        # 3. Frequency-Conditioned Relative Diffusion Cross-Attention Bias
        attn_bias = None
        if context_indices is not None and target_indices is not None and hasattr(self, "coords_25"):
            coords = self.coords_25.to(device)  # [25, 2]
            c_idx = torch.clamp(context_indices.long(), 0, coords.shape[0] - 1)  # [B, N_ctx]
            t_idx = torch.clamp(target_indices.long(), 0, coords.shape[0] - 1)   # [B, N_tgt]

            pos_tgt = coords[t_idx]             # [B, N_tgt, 2]
            pos_ctx = coords[c_idx]             # [B, N_ctx, 2]

            delta_r = pos_tgt.unsqueeze(2) - pos_ctx.unsqueeze(1)  # [B, N_tgt, N_ctx, 2]
            dist_r = torch.sqrt(torch.sum(delta_r ** 2, dim=-1, keepdim=True) + 1e-4)  # [B, N_tgt, N_ctx, 1]

            # Frequency conditioning
            if freq_condition is None:
                f_val = torch.full((B, N_tgt, c_idx.shape[1], 1), 0.5, device=device, dtype=torch.float32)
            else:
                if freq_condition.ndim == 1:
                    f_val = freq_condition.view(B, 1, 1, 1).expand(B, N_tgt, c_idx.shape[1], 1)
                elif freq_condition.ndim == 2:
                    f_val = freq_condition.unsqueeze(1).expand(B, N_tgt, c_idx.shape[1], 1)
                else:
                    f_val = freq_condition.view(B, 1, 1, 1).expand(B, N_tgt, c_idx.shape[1], 1)

            diff_scale = dist_r * torch.sqrt(torch.clamp(f_val, min=1e-4))  # [B, N_tgt, N_ctx, 1]
            rel_feat = torch.cat([delta_r, dist_r, f_val, diff_scale], dim=-1)  # [B, N_tgt, N_ctx, 5]
            rel_emb = self.rel_diff_mlp(rel_feat)  # [B, N_tgt, N_ctx, num_heads]
            attn_bias = rel_emb.permute(0, 3, 1, 2)  # [B, num_heads, N_tgt, N_ctx]

        # 4. Transformer Refinement
        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)

        q = self.norm(q)

        # 5. Dual-Head Latent Subspace Decomposition
        H_base = self.base_head(q)
        delta_pred = self.residual_head(q)
        H_pred = H_base + delta_pred

        self.last_h_base = H_base
        self.last_delta_pred = delta_pred

        if return_residual:
            return H_pred, delta_pred, H_base
        return H_pred


class AnalyticalHelmholtzCarrierPropagator(nn.Module):
    """
    Analytical Zero-Parameter Helmholtz Green's Propagator for Eddy Current Carrier Field:
    Solves boundary-to-interior quasi-static Helmholtz diffusion:
        w_{ij} = softmax( - r_{ij} / delta_eff )
        H_carrier,i = sum_j w_{ij} * H_ctx,j
    where:
        r_{ij} = ||pos_tgt,i - pos_ctx,j||_2
        delta_eff = r_coil_ref * sqrt(0.25 / max(omega_bar, 0.01))
    """
    def __init__(self, r_coil_ref: float = 3.0):
        super().__init__()
        self.r_coil_ref = r_coil_ref

    def forward(
        self,
        pos_tgt: torch.Tensor,       # [B, N_tgt, 2]
        pos_ctx: torch.Tensor,       # [B, N_ctx, 2]
        H_context: torch.Tensor,     # [B, N_ctx, D]
        freq_val: Optional[torch.Tensor] = None, # [B] or [B, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_r = pos_tgt.unsqueeze(2) - pos_ctx.unsqueeze(1)  # [B, N_tgt, N_ctx, 2]
        dist_r = torch.sqrt(torch.sum(delta_r ** 2, dim=-1) + 1e-6)  # [B, N_tgt, N_ctx]

        B = pos_tgt.shape[0]
        device = pos_tgt.device
        if freq_val is None:
            f_val = torch.full((B, 1, 1), 0.25, device=device, dtype=torch.float32)
        else:
            f_val = freq_val.view(B, 1, 1).clamp(min=1e-4)

        delta_eff = self.r_coil_ref * torch.sqrt(0.25 / f_val)  # [B, 1, 1]
        logits = - dist_r / delta_eff.clamp(min=1e-3)  # [B, N_tgt, N_ctx]
        weights = F.softmax(logits, dim=-1)  # [B, N_tgt, N_ctx]

        H_carrier = torch.bmm(weights, H_context)  # [B, N_tgt, D]
        return H_carrier, dist_r, delta_eff, delta_r


class DipolarAttentionBias(nn.Module):
    """
    Learnable Maxwell Dipolar Scattering Cross-Attention Bias:
    Models magnetic field diversion around defects in conductive media:
        M_{ij}^{(h)} = - softplus(gamma_h) * (r_{ij} / delta_eff)
                       + kappa_h * cos(2 * theta_{ij})
                       - softplus(alpha_h) * ln(1 + r_{ij})
    where:
        theta_{ij} = atan2(Delta y_{ij}, Delta x_{ij})
        cos(2*theta_{ij}) reflects dipolar field perturbation symmetry
    """
    def __init__(
        self,
        num_heads: int = 4,
        init_gamma: float = 1.0,
        init_alpha: float = 0.5,
        init_kappa: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        inv_softplus_gamma = math.log(math.exp(init_gamma) - 1.0) if init_gamma > 0 else 0.0
        inv_softplus_alpha = math.log(math.exp(init_alpha) - 1.0) if init_alpha > 0 else 0.0
        self.gamma_raw = nn.Parameter(torch.full((num_heads,), inv_softplus_gamma, dtype=torch.float32))
        self.alpha_raw = nn.Parameter(torch.full((num_heads,), inv_softplus_alpha, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.full((num_heads,), init_kappa, dtype=torch.float32))

    def forward(
        self,
        dist_r: torch.Tensor,       # [B, N_tgt, N_ctx]
        delta_eff: torch.Tensor,    # [B, 1, 1]
        delta_r: torch.Tensor,      # [B, N_tgt, N_ctx, 2]
    ) -> torch.Tensor:
        theta = torch.atan2(delta_r[..., 1], delta_r[..., 0])  # [B, N_tgt, N_ctx]

        gamma = F.softplus(self.gamma_raw).view(1, self.num_heads, 1, 1)
        alpha = F.softplus(self.alpha_raw).view(1, self.num_heads, 1, 1)
        kappa = self.kappa.view(1, self.num_heads, 1, 1)

        dist_exp = dist_r.unsqueeze(1)  # [B, 1, N_tgt, N_ctx]
        delta_exp = delta_eff.unsqueeze(1)  # [B, 1, 1, 1]
        theta_exp = theta.unsqueeze(1)  # [B, 1, N_tgt, N_ctx]

        decay_term = - gamma * (dist_exp / delta_exp.clamp(min=1e-3))
        dipole_term = kappa * torch.cos(2.0 * theta_exp)
        geom_term = - alpha * torch.log(1.0 + dist_exp)

        bias = decay_term + dipole_term + geom_term
        return bias  # [B, num_heads, N_tgt, N_ctx]


class DipolarScatteringPredictor5x5(nn.Module):
    """
    Dipolar Scattering World Model Predictor (EXP-37 Stage 4):
    Co-designed with HarmonicIsometricContextEncoder5x5.

    Key Innovations:
      1. Analytical Helmholtz Carrier Propagator: Computes the continuous nominal
         incident carrier field H_carrier analytically from visible boundary context tokens.
         Eliminates the dead-weight base_head collapse (norm 0.152 -> 0) observed in EXP-35.
      2. Dipolar Scattering Cross-Attention Bias: Injects Maxwell dipolar perturbation physics
         (-gamma * r/delta + kappa * cos(2*theta) - alpha * ln(1+r)) into cross-attention.
      3. Dedicated Flaw Scattering Head: 100% of predictor capacity focuses on predicting
         the localized scattering disturbance Delta H_scat.
      4. Full-Rank Output: H_pred = H_carrier + Delta H_scat.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        r_coil_ref: float = 3.0,
        diffusion_gamma_init: float = 1.0,
        diffusion_alpha_init: float = 0.5,
        dipolar_kappa_init: float = 0.2,
        relative_perturbation_target: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.relative_perturbation_target = relative_perturbation_target
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.carrier_propagator = AnalyticalHelmholtzCarrierPropagator(r_coil_ref=r_coil_ref)
        self.dipolar_bias = DipolarAttentionBias(
            num_heads=num_heads,
            init_gamma=diffusion_gamma_init,
            init_alpha=diffusion_alpha_init,
            init_kappa=dipolar_kappa_init,
        )

        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Dedicated Scattering Head (100% capacity)
        self.scattering_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Residual head alias for downstream inspection and compatibility
        self.residual_head = self.scattering_head

        # Spatial coordinates buffer [25, 2] in mm
        offsets = get_spatial_topology_offsets(
            topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
        )
        self.register_buffer("coords_25", torch.from_numpy(offsets).float(), persistent=False)

        # Cached monitoring stats
        self.last_h_base: Optional[torch.Tensor] = None
        self.last_delta_pred: Optional[torch.Tensor] = None

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, N_tgt, D = target_pos.shape
        device = target_pos.device

        # Resolve probe coordinates
        if context_indices is not None and target_indices is not None and hasattr(self, "coords_25"):
            coords = self.coords_25.to(device)
            c_idx = torch.clamp(context_indices.long(), 0, coords.shape[0] - 1)
            t_idx = torch.clamp(target_indices.long(), 0, coords.shape[0] - 1)
            pos_tgt = coords[t_idx]  # [B, N_tgt, 2]
            pos_ctx = coords[c_idx]  # [B, N_ctx, 2]
        else:
            # Default fallback: 9 core target probes, 16 outer context probes
            coords = self.coords_25.to(device) if hasattr(self, "coords_25") else torch.zeros(25, 2, device=device)
            default_tgt_idx = torch.tensor([6, 7, 8, 11, 12, 13, 16, 17, 18], device=device).unsqueeze(0).expand(B, -1)
            default_ctx_idx = torch.tensor([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24], device=device).unsqueeze(0).expand(B, -1)
            pos_tgt = coords[default_tgt_idx]
            pos_ctx = coords[default_ctx_idx]

        # 1. Analytical Helmholtz Carrier Propagation
        H_carrier, dist_r, delta_eff, delta_r = self.carrier_propagator(
            pos_tgt=pos_tgt,
            pos_ctx=pos_ctx,
            H_context=H_context,
            freq_val=freq_condition,
        )  # H_carrier: [B, N_tgt, D]

        # 2. Maxwell Dipolar Scattering Cross-Attention Bias
        attn_bias = self.dipolar_bias(
            dist_r=dist_r,
            delta_eff=delta_eff,
            delta_r=delta_r,
        )  # [B, num_heads, N_tgt, N_ctx]

        # 3. Target Query Formulation: initial carrier + query prior + spatial position
        queries = H_carrier + self.mask_token.expand(B, N_tgt, -1) + target_pos

        # 4. Predictor Transformer Refinement
        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context, attn_bias=attn_bias)
        q = self.norm(q)

        # 5. Dedicated Flaw Scattering Output
        delta_pred = self.scattering_head(q)  # [B, N_tgt, D]

        if self.relative_perturbation_target:
            h_carrier_base = H_carrier - H_context.mean(dim=1, keepdim=True)
        else:
            h_carrier_base = H_carrier

        H_pred = h_carrier_base + delta_pred

        self.last_h_base = h_carrier_base
        self.last_delta_pred = delta_pred

        if return_residual:
            return H_pred, delta_pred, h_carrier_base
        return H_pred


def build_predictor_5x5(config) -> nn.Module:
    """
    Factory function to construct Predictor based on config.
    Defaults to ContinuousHelmholtzPredictor5x5 or ResidualDiffusionPredictor5x5.
    """
    predictor_type = getattr(config, "predictor_type", "dipolar_scattering")
    spatial_topology = getattr(config, "spatial_topology", "concentric_star")
    star_radii = getattr(config, "star_radii", (1, 3, 7))

    if predictor_type in ("dipolar_scattering", "dipolar", "dipolar_diffusion"):
        return DipolarScatteringPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
            grid_size=config.grid_size,
            r_coil_ref=getattr(config, "r_coil_ref", 3.0),
            diffusion_gamma_init=getattr(config, "diffusion_gamma_init", 1.0),
            diffusion_alpha_init=getattr(config, "diffusion_alpha_init", 0.5),
            dipolar_kappa_init=getattr(config, "dipolar_kappa_init", 0.2),
            relative_perturbation_target=getattr(config, "relative_perturbation_target", True),
        )
    elif predictor_type in ("freq_conditioned_diffusion", "frequency_conditioned_diffusion", "frequency_diffusion"):
        return FrequencyConditionedDiffusionPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
            grid_size=config.grid_size,
        )
    elif predictor_type in ("neural_field_subspace", "subspace_field_operator", "subspace_neural_field", "continuous_field_subspace"):
        return NeuralFieldSubspacePredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            spatial_topology=spatial_topology,
            star_radii=star_radii,
            grid_size=config.grid_size,
        )
    elif predictor_type in ("anisotropic_diffusion", "anisotropic_helmholtz", "anisotropic"):
        return AnisotropicDiffusionPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            gamma_init=getattr(config, "diffusion_gamma_init", 1.0),
            alpha_init=getattr(config, "diffusion_alpha_init", 0.5),
            alpha_x_init=getattr(config, "diffusion_alpha_x_init", 1.0),
            alpha_y_init=getattr(config, "diffusion_alpha_y_init", 1.0),
            d_scale_init=getattr(config, "diffusion_d_scale_init", 1.5),
            d_scale_min=getattr(config, "diffusion_d_scale_min", 1.0),
            spatial_topology=spatial_topology,
            star_radii=star_radii,
        )
    elif predictor_type in ("continuous_helmholtz", "helmholtz", "helmholtz_diffusion"):
        return ContinuousHelmholtzPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            gamma_init=getattr(config, "diffusion_gamma_init", 1.0),
            alpha_init=getattr(config, "diffusion_alpha_init", 0.5),
            d_scale_init=getattr(config, "diffusion_d_scale_init", 1.5),
            d_scale_min=getattr(config, "diffusion_d_scale_min", 1.0),
            spatial_topology=spatial_topology,
            star_radii=star_radii,
        )
    elif predictor_type in ("residual_diffusion", "residual", "auto", "default"):
        return ResidualDiffusionPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            gamma_init=getattr(config, "diffusion_gamma_init", 1.0),
            alpha_init=getattr(config, "diffusion_alpha_init", 0.5),
            spatial_topology=spatial_topology,
            star_radii=star_radii,
        )
    elif predictor_type in ("parabolic_diffusion", "parabolic_greens", "parabolic"):
        return ParabolicDiffusionPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            gamma_init=getattr(config, "diffusion_gamma_init", 1.0),
            alpha_init=getattr(config, "diffusion_alpha_init", 0.5),
            spatial_topology=spatial_topology,
            star_radii=star_radii,
        )
    elif predictor_type == "operator_diffusion":
        return OperatorDiffusionPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
            gamma_init=getattr(config, "diffusion_gamma_init", 1.0),
            beta_init=getattr(config, "diffusion_beta_init", 0.5),
            spatial_topology=spatial_topology,
            star_radii=star_radii,
        )
    elif predictor_type == "standard":
        return Predictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")
