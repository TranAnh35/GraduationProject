"""
Predictor for 5x5 PECT-JEPA.
Predicts target latent representations from target queries (mask_token + target_pos)
and context key/value representations (H_context). Zero information leakage.
"""

import math
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
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.num_freq_bins = num_freq_bins
        self.op_embedding = DiffusionOperatorEmbedding(embed_dim=embed_dim)

        # Learnable physical coupling parameters (strictly non-negative via softplus)
        raw_gamma = math.log(math.exp(gamma_init) - 1.0) if gamma_init > 0 else 0.0
        raw_beta = math.log(math.exp(beta_init) - 1.0) if beta_init > 0 else 0.0
        self.raw_gamma = nn.Parameter(torch.tensor(raw_gamma, dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.tensor(raw_beta, dtype=torch.float32))

        self._register_distance_tables()

    def _register_distance_tables(self):
        # 50 tokens: 25 spatial points * 2 diffusion scales (shallow=0, deep=1)
        coords_50 = []
        scales_50 = []
        for k in range(50):
            sp = k // 2
            coords_50.append((float(sp % 5), float(sp // 5)))
            scales_50.append(float(k % 2))

        coords_50_t = torch.tensor(coords_50, dtype=torch.float32)  # [50, 2]
        scales_50_t = torch.tensor(scales_50, dtype=torch.float32)  # [50]

        diff_50 = coords_50_t.unsqueeze(1) - coords_50_t.unsqueeze(0)  # [50, 50, 2]
        dist_50 = torch.norm(diff_50, p=2, dim=-1)  # [50, 50]
        scale_diff_50 = torch.abs(scales_50_t.unsqueeze(1) - scales_50_t.unsqueeze(0))  # [50, 50]

        self.register_buffer("dist_table_50", dist_50, persistent=False)
        self.register_buffer("scale_diff_table_50", scale_diff_50, persistent=False)

        # 25 tokens: 25 spatial points * 1 scale
        coords_25 = []
        for k in range(25):
            coords_25.append((float(k % 5), float(k // 5)))
        coords_25_t = torch.tensor(coords_25, dtype=torch.float32)  # [25, 2]
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
            coords_100.append((float(sp % 5), float(sp // 5)))
            scales_100.append(float(k % 4))

        coords_100_t = torch.tensor(coords_100, dtype=torch.float32)  # [100, 2]
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
            if target_indices.max() >= 50:
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
                safe_d_tau = torch.clamp(d_tau, min=1.0)

                # M_{ij}^{diff} = - gamma * (||Delta r||^2 / Delta tau) - alpha * log(Delta tau)
                M_diff = - gamma * (d_sq / safe_d_tau) - alpha * torch.log(safe_d_tau)
                M_diff = torch.where(
                    causal_mask,
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
            if target_indices.max() >= 50:
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
                safe_d_tau = torch.clamp(d_tau, min=1.0)

                # Green's Attention Bias
                M_diff = - gamma * (d_sq / safe_d_tau) - alpha * torch.log(safe_d_tau)
                M_diff = torch.where(
                    causal_mask,
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


def build_predictor_5x5(config) -> nn.Module:
    """
    Factory function to construct Predictor based on config.
    Defaults to ResidualDiffusionPredictor5x5 (Residual Diffusion Predictor).
    """
    predictor_type = getattr(config, "predictor_type", "residual_diffusion")
    spatial_topology = getattr(config, "spatial_topology", "concentric_star")
    star_radii = getattr(config, "star_radii", (1, 3, 7))

    if predictor_type in ("residual_diffusion", "residual", "auto", "default"):
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
