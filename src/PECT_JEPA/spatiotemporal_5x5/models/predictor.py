"""
Predictor for 5x5 PECT-JEPA.
Predicts target latent representations from target queries (mask_token + target_pos)
and context key/value representations (H_context). Zero information leakage.
"""

import torch
import torch.nn as nn
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


class Predictor5x5(nn.Module):
    """
    Predictor Network:
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

    def forward(self, H_context: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        B, N_tgt, D = target_pos.shape
        queries = self.mask_token.expand(B, N_tgt, -1) + target_pos

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context)

        return self.norm(q)
