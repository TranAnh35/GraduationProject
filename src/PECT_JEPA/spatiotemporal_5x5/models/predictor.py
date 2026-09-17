"""
Predictor for 5x5 PECT-JEPA.
Predicts target latent representations from target queries (mask_token + target_pos)
and context key/value representations (H_context). Zero information leakage.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from .attention import MultiheadSelfAttention, MultiheadCrossAttention, MLP


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

    def forward(self, target_queries: torch.Tensor, H_context: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention
        q = target_queries + self.self_attn(self.norm_self(target_queries))
        # 2. Cross-Attention to Context
        q = q + self.cross_attn(
            query=self.norm_cross_q(q),
            key_value=self.norm_cross_kv(H_context)
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
        q_diff = self.mlp(fourier_feats).unsqueeze(1)  # [B, 1, embed_dim]
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
        diffusion_operator: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N_tgt, D = target_pos.shape
        queries = self.mask_token.expand(B, N_tgt, -1) + target_pos
        if diffusion_operator is not None:
            if diffusion_operator.ndim == 2:
                diffusion_operator = diffusion_operator.unsqueeze(1)
            queries = queries + diffusion_operator

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context)

        return self.norm(q)


class OperatorDiffusionPredictor5x5(Predictor5x5):
    """
    Physics Neural Operator Predictor for PECT-JEPA:
    Conditions prediction on electromagnetic diffusion query q_diff(omega_k)
    governed by the Helmholtz diffusion equation:
        nabla^2 B = j * omega * mu * sigma * B

    Queries = mask_token + target_pos + q_diff(omega_k)
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_freq_bins: int = 14,
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

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        freq_condition: Optional[torch.Tensor] = None,
        diffusion_operator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            H_context: [B, N_ctx, D]
            target_pos: [B, N_tgt, D]
            freq_condition: [B] normalized frequency or frequency bin index
            diffusion_operator: optional explicit [B, 1, D] condition
        """
        if diffusion_operator is None:
            if freq_condition is not None:
                # If integer indices (1..K), normalize by num_freq_bins
                if freq_condition.dtype in (torch.int32, torch.int64):
                    freq_norm = freq_condition.float() / float(self.num_freq_bins)
                else:
                    freq_norm = freq_condition.float()
                op_cond = self.op_embedding(freq_norm)
            else:
                op_cond = self.op_embedding.default_op
        else:
            op_cond = diffusion_operator

        return super().forward(H_context, target_pos, diffusion_operator=op_cond)


def build_predictor_5x5(config) -> nn.Module:
    """
    Factory function to construct Predictor based on config.
    """
    predictor_type = getattr(config, "predictor_type", "operator_diffusion")
    if predictor_type == "operator_diffusion":
        return OperatorDiffusionPredictor5x5(
            embed_dim=config.embed_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            num_freq_bins=getattr(config, "num_freq_bins", 14),
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
