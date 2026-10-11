"""
Predictor for Cross-Condition PECT-JEPA.
Predicts masked target representations of Condition B from Condition A context
and Condition B self-supervised transfer function condition token c_B.
"""

from typing import Optional
import torch
import torch.nn as nn
from .attention import MultiheadSelfAttention, MultiheadCrossAttention, MLP
from ..configs.config import CrossConditionConfig


class PredictorBlock(nn.Module):
    """
    Predictor Transformer Block:
    1. Self-Attention among Target Queries
    2. Cross-Attention from Target Queries to Context H_context
    3. Feed-Forward MLP
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiheadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(
        self,
        target_queries: torch.Tensor,
        H_context: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Self-Attention
        q = target_queries + self.self_attn(self.norm_self(target_queries))
        # 2. Cross-Attention to Context
        q = q + self.cross_attn(
            query=self.norm_cross_q(q),
            key_value=self.norm_cross_kv(H_context),
        )
        # 3. MLP
        q = q + self.mlp(self.norm_mlp(q))
        return q


class CrossConditionPredictor(nn.Module):
    """
    Cross-Condition Predictor Network:
    Input:
        H_context: [B, N_ctx, D] representation from Condition A.
        target_pos: [B, N_tgt, D] target position embeddings.
        condition_token: [B, 1, D] Condition B transfer function token.
    Output:
        H_pred: [B, N_tgt, D] predicted target representations.
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock(
                embed_dim=config.embed_dim,
                num_heads=config.predictor_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.predictor_depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        H_context: torch.Tensor,
        target_pos: torch.Tensor,
        condition_token: torch.Tensor,
    ) -> torch.Tensor:
        B, N_tgt, D = target_pos.shape
        # Target queries: mask token + spatial position + condition modulation
        queries = self.mask_token.expand(B, N_tgt, -1) + target_pos + condition_token

        q = queries
        for blk in self.blocks:
            q = blk(target_queries=q, H_context=H_context)

        return self.norm(q)
