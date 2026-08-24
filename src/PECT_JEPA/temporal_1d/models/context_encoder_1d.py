"""
Context Encoder for 1D Temporal PECT-JEPA (implement.md, Section 4.5).
Encodes ONLY visible context patches (Token Removal / Dropping).
Masked target patches are completely excluded from the Context Encoder.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import TransformerBlock1D


class ContextEncoder1D(nn.Module):
    """
    Module 3: 1D Context Encoder.
    Processes ONLY visible context tokens [B, N_ctx, D] (e.g. N_ctx = 5).
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
        self.embed_dim = embed_dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            TransformerBlock1D(
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
        context_tokens: torch.Tensor,
        context_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: [B, N_ctx, D] visible context patches
            context_pos: [B, N_ctx, D] 1D positional embeddings for visible context patches

        Returns:
            H_context: [B, N_ctx, D] encoded context representations
        """
        x = context_tokens
        if context_pos is not None:
            x = x + context_pos

        for block in self.blocks:
            x = block(x)

        H_context = self.norm(x)
        return H_context
