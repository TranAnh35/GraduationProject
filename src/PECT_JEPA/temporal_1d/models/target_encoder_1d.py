"""
EMA Target Encoder for 1D Temporal PECT-JEPA (implement.md, Section 4.5).
Encodes masked target patches to generate target latents H_target.
Parameters are frozen from autograd (requires_grad = False) and updated exclusively via EMA.
"""

import torch
import torch.nn as nn
from typing import Optional

from .context_encoder_1d import ContextEncoder1D


class TargetEncoder1D(nn.Module):
    """
    Module 4: 1D EMA Target Encoder.
    Processes masked target tokens [B, N_tgt, D] and target positional embeddings [B, N_tgt, D].
    Updated exclusively via Exponential Moving Average (EMA) from ContextEncoder1D.
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
        self.encoder = ContextEncoder1D(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        # Freeze parameters from gradient computation
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        target_tokens: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            target_tokens: [B, N_tgt, D] masked target tokens
            target_pos: [B, N_tgt, D] 1D positional embeddings for target tokens

        Returns:
            H_target: [B, N_tgt, D] target latent representations (detached)
        """
        H_target = self.encoder(
            context_tokens=target_tokens,
            context_pos=target_pos
        )
        return H_target.detach()

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder1D, momentum: float):
        """
        EMA parameter update:
        theta_T <- m * theta_T + (1 - m) * theta_C
        """
        for param_t, param_c in zip(self.encoder.parameters(), context_encoder.parameters()):
            param_t.data.mul_(momentum).add_(param_c.data, alpha=1.0 - momentum)
