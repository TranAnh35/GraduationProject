"""
Target Encoder for Cross-Condition PECT-JEPA.
Encodes masked core target probes from Condition B (EMA momentum updated, gradient-free).
"""

from typing import Optional
import torch
import torch.nn as nn
from .context_encoder import ContextEncoder
from ..configs.config import CrossConditionConfig


class TargetEncoder(nn.Module):
    """
    Target Encoder (EMA Teacher) for target tokens.
    Input: target_tokens [B, N_tgt, D], target_pos [B, N_tgt, D]
    Output: H_tgt [B, N_tgt, D] (detached)
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        self.encoder = ContextEncoder(config)
        # Target encoder never receives gradient directly
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self, context_encoder: nn.Module, momentum: float = 0.996):
        """EMA weight update: theta_tgt = momentum * theta_tgt + (1 - momentum) * theta_ctx."""
        for p_tgt, p_ctx in zip(self.encoder.parameters(), context_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)

    def forward(
        self,
        target_tokens: torch.Tensor,
        target_pos: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(target_tokens, target_pos, coords=coords)
