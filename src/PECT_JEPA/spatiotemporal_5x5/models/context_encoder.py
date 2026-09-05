"""
Context Encoder for 5x5 PECT-JEPA.
Processes only visible context tokens (with their positional embeddings).
"""

import torch
import torch.nn as nn
from .attention import TransformerBlock


class ContextEncoder5x5(nn.Module):
    """
    Context Encoder processing visible context points:
    Input: context_tokens [B, N_ctx, D], context_pos [B, N_ctx, D]
    Output: H_ctx [B, N_ctx, D]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
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

    def forward(self, context_tokens: torch.Tensor, context_pos: torch.Tensor) -> torch.Tensor:
        h = context_tokens + context_pos
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)
