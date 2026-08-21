"""
EMA Target Encoder for PECT-JEPA v0.2 (Module 4, implement.md).
Encodes masked target tokens to generate prediction target latents H_target.
Parameters are frozen from autograd (requires_grad = False) and updated exclusively via EMA.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional

from .context_encoder import ContextEncoder


class TargetEncoder(nn.Module):
    """
    Module 4: EMA Target Encoder (implement.md).
    Architecture clone of ContextEncoder.
    Processes masked target tokens [B, N_tgt, D] and target positional embeddings [B, N_tgt, D].
    Updated exclusively via EMA from ContextEncoder.
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
        self.encoder = ContextEncoder(
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
            target_pos: [B, N_tgt, D] 3D positional embeddings for target tokens

        Returns:
            H_target: [B, N_tgt, D] target latent representations (detached)
        """
        H_target = self.encoder(
            context_tokens=target_tokens,
            context_pos=target_pos
        )
        return H_target.detach()

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder, momentum: float):
        """
        Exponential Moving Average parameter update:
        theta_T <- m * theta_T + (1 - m) * theta_C
        """
        for param_t, param_c in zip(self.encoder.parameters(), context_encoder.parameters()):
            param_t.data.mul_(momentum).add_(param_c.data, alpha=1.0 - momentum)
