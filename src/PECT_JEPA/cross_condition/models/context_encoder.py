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
        self.subspace_norm = bool(getattr(config, "subspace_norm", False)) and bool(
            getattr(config, "latent_disentanglement", False)
        )
        if self.subspace_norm:
            self.carrier_dim = int(getattr(config, "carrier_dim", config.embed_dim // 2))
            self.scattering_dim = int(getattr(config, "scattering_dim", config.embed_dim - self.carrier_dim))
            self.norm = None
            self.norm_c = nn.LayerNorm(self.carrier_dim)
            self.norm_s = nn.LayerNorm(self.scattering_dim, bias=False)
        else:
            self.norm = nn.LayerNorm(config.embed_dim)
            self.norm_c = None
            self.norm_s = None

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
        if self.subspace_norm:
            h_c = self.norm_c(h[..., :self.carrier_dim])
            h_s = self.norm_s(h[..., self.carrier_dim:])
            return torch.cat([h_c, h_s], dim=-1)
        return self.norm(h)
