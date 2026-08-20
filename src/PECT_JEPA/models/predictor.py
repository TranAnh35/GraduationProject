"""
Predictor for PECT-JEPA (Module H, Section 13).
Predicts target latent representations from encoded context and target positional embeddings.
Target token content is never visible to the Predictor.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiheadSelfAttention, MultiheadCrossAttention, MLP


class PredictorBlock(nn.Module):
    """
    Predictor Transformer Block:
    Target queries attend to themselves (Self-Attention) and to Context latents (Cross-Attention).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        # Self-attention among target queries
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-attention to context memory
        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(self, target_queries: torch.Tensor, H_context: torch.Tensor) -> torch.Tensor:
        # 1. Self-attention on target queries
        q = target_queries + self.self_attn(self.norm_self(target_queries))

        # 2. Cross-attention: queries attend to context
        q_norm = self.norm_cross_q(q)
        kv_norm = self.norm_cross_kv(H_context)
        q = q + self.cross_attn(query=q_norm, key_value=kv_norm)

        # 3. MLP
        q = q + self.mlp(self.norm_mlp(q))
        return q


class Predictor(nn.Module):
    """
    Module H: Predictor (Section 13).
    Inputs:
        - H_context: [B, N_ctx, D] (encoded visible context)
        - target_pos: [B, N_tgt, D] (positional embeddings for target tokens)
    Output:
        - H_pred: [B, N_tgt, D] (predicted target latents)
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

        # Learnable mask token to initialize target query vectors
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
        self.pred_head = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            H_context: [B, N_ctx, D] encoded context representations
            target_pos: [B, N_tgt, D] positional embeddings of masked target locations

        Returns:
            H_pred: [B, N_tgt, D] predicted target latents
        """
        B, N_tgt, D = target_pos.shape

        # Initialize queries: mask_token + target_positional_embeddings
        target_queries = self.mask_token.expand(B, N_tgt, -1) + target_pos

        # Process through predictor blocks
        q = target_queries
        for block in self.blocks:
            q = block(q, H_context)

        H_pred = self.pred_head(self.norm(q))
        return H_pred
