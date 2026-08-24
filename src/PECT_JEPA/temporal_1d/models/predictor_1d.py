"""
Predictor for 1D Temporal PECT-JEPA (implement.md, Section 4.6).
Predicts target latent representations from Target Queries (e_mask + target_pos)
and Context Keys/Values (H_context).
Target token content is never visible to the Predictor (Zero Leakage).
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiheadSelfAttention, MultiheadCrossAttention, MLP


class PredictorBlock1D(nn.Module):
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
        # 1. Self-Attention on target queries
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 2. Cross-Attention to context memory
        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 3. Feed-Forward MLP
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(self, target_queries: torch.Tensor, H_context: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention on target queries
        q = target_queries + self.self_attn(self.norm_self(target_queries))

        # 2. Cross-Attention: target queries attend to context
        q_norm = self.norm_cross_q(q)
        kv_norm = self.norm_cross_kv(H_context)
        q = q + self.cross_attn(query=q_norm, key_value=kv_norm)

        # 3. MLP
        q = q + self.mlp(self.norm_mlp(q))
        return q


class Predictor1D(nn.Module):
    """
    Module 5: 1D Predictor.
    Inputs:
        - H_context: [B, N_ctx, D] (encoded visible context tokens)
        - target_pos: [B, N_tgt, D] (1D positional embeddings of masked target locations)
    Output:
        - H_pred: [B, N_tgt, D] (predicted target representations)
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
        self.depth = depth

        # Learnable mask token e_mask
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock1D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.pred_head = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            H_context: [B, N_ctx, D] encoded context representations
            target_pos: [B, N_tgt, D] 1D positional embeddings of target tokens

        Returns:
            H_pred: [B, N_tgt, D] predicted target latents
        """
        B, N_tgt, D = target_pos.shape

        # Target Queries: Q = e_mask + target_pos
        target_queries = self.mask_token.expand(B, N_tgt, -1) + target_pos

        q = target_queries
        for block in self.blocks:
            q = block(q, H_context)

        H_pred = self.pred_head(self.norm(q))
        return H_pred
