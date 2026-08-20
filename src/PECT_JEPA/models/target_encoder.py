"""
EMA Target Encoder for PECT-JEPA (Module G, Section 12).
Encodes masked target tokens to generate prediction target latents H_target.
Parameters are frozen from autograd and updated exclusively via Exponential Moving Average.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .context_encoder import ContextEncoder


class TargetEncoder(nn.Module):
    """
    Module G: EMA Target Encoder.
    Encodes masked target tokens into target latents H_target.
    Updated via EMA from Context Encoder parameters.
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
        self.encoder = ContextEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_type=attention_type
        )
        # Freeze parameters from gradient computation (Section 12.5)
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        target_tokens: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            target_tokens: [B, N_tgt, D] masked target tokens
            target_pos: [B, N_tgt, D] positional embeddings for target tokens
            target_indices: [B, N_tgt] token indices of target positions
            grid_shape: (H_t, W_t, K_t)

        Returns:
            H_target: [B, N_tgt, D] target latent representations (detached)
        """
        H_target = self.encoder(
            context_tokens=target_tokens,
            context_pos=target_pos,
            context_indices=target_indices,
            grid_shape=grid_shape
        )
        return H_target.detach()

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder, momentum: float):
        """
        Exponential Moving Average parameter update (Section 12.3):
        phi_target = m * phi_target + (1 - m) * phi_context
        """
        for param_t, param_c in zip(self.encoder.parameters(), context_encoder.parameters()):
            param_t.data.mul_(momentum).add_(param_c.data, alpha=1.0 - momentum)
