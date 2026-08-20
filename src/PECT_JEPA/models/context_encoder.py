"""
Context Encoder for PECT-JEPA (Module F, Section 11 & 16).
Supports Factorized Spatio-Temporal Attention and Full Global Attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiheadSelfAttention, FactorizedSpatioTemporalBlock, MLP


class SequenceTransformerBlock(nn.Module):
    """Standard 1D Sequence Transformer Block."""
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
    Module F: Context Encoder (Section 11 & 16).
    Supports:
      - 'factorized': Factorized Spatio-Temporal Attention (Spatial + Temporal factorized)
      - 'full': Full Global Self-Attention
    """
    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_type: str = "factorized"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.attention_type = attention_type

        if attention_type == "factorized":
            self.blocks = nn.ModuleList([
                FactorizedSpatioTemporalBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(depth)
            ])
            # Learnable placeholder for positions not in context
            self.mask_placeholder = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
            nn.init.trunc_normal_(self.mask_placeholder, std=0.02)
        else:
            self.blocks = nn.ModuleList([
                SequenceTransformerBlock(
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
        context_pos: Optional[torch.Tensor] = None,
        context_indices: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: [B, N_ctx, D] visible context tokens
            context_pos: [B, N_ctx, D] positional embeddings for context tokens
            context_indices: [B, N_ctx] flat token indices of context positions
            grid_shape: (H_t, W_t, K_t)

        Returns:
            H_context: [B, N_ctx, D] encoded context latents
        """
        x = context_tokens
        if context_pos is not None:
            x = x + context_pos

        B, N_ctx, D = x.shape

        # Use factorized attention if specified and grid dimensions are available
        if self.attention_type == "factorized" and grid_shape is not None and context_indices is not None:
            H_t, W_t, K_t = grid_shape
            total_tokens = H_t * W_t * K_t

            # Scatter context tokens into full 3D grid
            grid_flat = self.mask_placeholder.expand(B, total_tokens, D).clone()
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N_ctx)
            grid_flat[batch_idx, context_indices] = x
            grid_3d = grid_flat.view(B, H_t, W_t, K_t, D)

            # Create padding mask for masked tokens (True = ignore in attention)
            pad_mask_flat = torch.ones(B, total_tokens, dtype=torch.bool, device=x.device)
            pad_mask_flat[batch_idx, context_indices] = False
            pad_mask_3d = pad_mask_flat.view(B, H_t, W_t, K_t)

            for block in self.blocks:
                grid_3d = block(grid_3d, key_padding_mask_grid=pad_mask_3d)

            grid_3d = self.norm(grid_3d)
            grid_flat_out = grid_3d.view(B, total_tokens, D)
            H_context = grid_flat_out[batch_idx, context_indices]
            return H_context

        # Fallback to full sequence attention
        for block in self.blocks:
            x = block(x)

        H_context = self.norm(x)
        return H_context
