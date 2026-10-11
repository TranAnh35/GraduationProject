"""
Context Encoder for Cross-Condition PECT-JEPA.
Encodes visible context probes from Condition A using Pre-LN Transformer blocks.
"""

from typing import Optional
import torch
import torch.nn as nn
from .attention import TransformerBlock
from ..configs.config import CrossConditionConfig


class ContextEncoder(nn.Module):
    """
    Context Encoder processing visible outer context probes.
    Input: context_tokens [B, N_ctx, D], context_pos [B, N_ctx, D]
    Output: H_ctx [B, N_ctx, D]
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.encoder_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            )
            for _ in range(config.encoder_depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: [B, N_ctx, D]
            context_pos: [B, N_ctx, D]
        Returns:
            H_ctx: [B, N_ctx, D]
        """
        h = context_tokens + context_pos
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)
