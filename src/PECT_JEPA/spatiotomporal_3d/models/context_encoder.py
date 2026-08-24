"""
Context Encoder for PECT-JEPA v0.2 (Module 3, implement.md).
Encodes ONLY visible context tokens (Token Removal / Dropping).
Masked tokens are completely excluded from the Context Encoder.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiheadSelfAttention, MLP


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block with Multi-Head Self-Attention and MLP."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    """
    Module 3: Context Encoder (implement.md).
    Processes ONLY visible context tokens [B, N_ctx, D].
    Token Dropping: Masked tokens are completely removed, not replaced with placeholders or zeros.
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
            TransformerBlock(
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
            context_tokens: [B, N_ctx, D] visible context tokens
            context_pos: [B, N_ctx, D] 3D positional embeddings for visible context tokens

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
