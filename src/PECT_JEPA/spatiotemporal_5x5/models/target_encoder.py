"""
Target Encoder for 5x5 PECT-JEPA.
Encodes masked target tokens (detached, updated via EMA from Context Encoder).
"""

import torch
import torch.nn as nn
from .context_encoder import ContextEncoder5x5


class TargetEncoder5x5(nn.Module):
    """
    Target Encoder (EMA Teacher) for target tokens:
    Input: target_tokens [B, N_tgt, D], target_pos [B, N_tgt, D]
    Output: H_target [B, N_tgt, D] (detached)
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
        self.encoder = ContextEncoder5x5(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        # Freeze target encoder parameters
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder5x5, momentum: float = 0.996):
        """Update weights via EMA: θ_target = momentum * θ_target + (1 - momentum) * θ_context."""
        for p_tgt, p_ctx in zip(self.encoder.parameters(), context_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)

    def forward(self, target_tokens: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        return self.encoder(target_tokens, target_pos)
