"""
Context Encoder for Cross-Condition PECT-JEPA.
Encodes visible context probes from Condition A using Pre-LN Transformer blocks.
"""

from typing import Optional
import torch
import torch.nn as nn
from .attention import TransformerBlock
from ..configs.config import CrossConditionConfig


class TwoStreamContextEncoder(nn.Module):
    """
    Two-Stream Decoupled Encoder for Cross-Condition PECT-JEPA (Direction B).
    Maintains structurally isolated Carrier (z_c) and Scattering (z_s) streams:
    - Carrier Stream: models smooth background incident field B_inc.
    - Scattering Stream: models localized flaw perturbation Delta B.
    - Unidirectional Cross-Stream Gating: Carrier informs Scattering with stop-gradient
      (z_c.detach()), guaranteeing that flaw scattering can NEVER leak into carrier.
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config
        self.carrier_dim = int(getattr(config, "carrier_dim", config.embed_dim // 2))
        self.scattering_dim = int(getattr(config, "scattering_dim", config.embed_dim - self.carrier_dim))
        diff_attn = bool(getattr(config, "diffusion_attention", False))

        num_heads_c = max(1, config.encoder_heads // 2)
        num_heads_s = max(1, config.encoder_heads // 2)

        self.carrier_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.carrier_dim,
                num_heads=num_heads_c,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                diffusion_attention=diff_attn,
            )
            for _ in range(config.encoder_depth)
        ])
        self.scattering_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.scattering_dim,
                num_heads=num_heads_s,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                diffusion_attention=diff_attn,
            )
            for _ in range(config.encoder_depth)
        ])

        # Unidirectional gating projections (Carrier -> Scattering only)
        self.carrier_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.carrier_dim, self.scattering_dim),
                nn.Sigmoid(),
            )
            for _ in range(config.encoder_depth)
        ])
        self.carrier_proj = nn.ModuleList([
            nn.Linear(self.carrier_dim, self.scattering_dim)
            for _ in range(config.encoder_depth)
        ])

        self.coupling_mode = getattr(config, "carrier_coupling_mode", "additive")

        self.norm_c = nn.LayerNorm(self.carrier_dim)
        self.norm_s = nn.LayerNorm(self.scattering_dim, bias=False)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = context_tokens + context_pos
        h_c = h[..., :self.carrier_dim]
        h_s = h[..., self.carrier_dim:]

        for c_blk, s_blk, gate, proj in zip(
            self.carrier_blocks, self.scattering_blocks, self.carrier_gates, self.carrier_proj
        ):
            h_c = c_blk(h_c, coords=coords)
            c_detached = h_c.detach()
            if self.coupling_mode == "none":
                h_s = s_blk(h_s, coords=coords)
            elif self.coupling_mode == "multiplicative":
                # Pure multiplicative modulation: scales scattering sensitivity without injecting DC carrier energy
                c_mod = gate(c_detached) * torch.tanh(proj(c_detached))
                h_s = s_blk(h_s * (1.0 + c_mod), coords=coords)
            else:
                # Additive carrier injection (legacy)
                c_mod = gate(c_detached) * proj(c_detached)
                h_s = s_blk(h_s + c_mod, coords=coords)

        out_c = self.norm_c(h_c)
        out_s = self.norm_s(h_s)
        return torch.cat([out_c, out_s], dim=-1)


class ContextEncoder(nn.Module):
    """
    Context Encoder processing visible outer context probes.
    Input: context_tokens [B, N_ctx, D], context_pos [B, N_ctx, D], coords [B, N_ctx, 2] (optional)
    Output: H_ctx [B, N_ctx, D]
    """
    def __init__(self, config: Optional[CrossConditionConfig] = None):
        super().__init__()
        if config is None:
            config = CrossConditionConfig()
        self.config = config
        self.two_stream = bool(getattr(config, "two_stream_backbone", False))

        if self.two_stream:
            self.stream_encoder = TwoStreamContextEncoder(config)
            self.blocks = None
            self.norm = None
            self.norm_c = None
            self.norm_s = None
        else:
            self.stream_encoder = None
            diff_attn = bool(getattr(config, "diffusion_attention", False))
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.encoder_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    diffusion_attention=diff_attn,
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
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: [B, N_ctx, D]
            context_pos: [B, N_ctx, D]
            coords: Optional [B, N_ctx, 2] physical probe coordinates in [-1, 1]^2
        Returns:
            H_ctx: [B, N_ctx, D]
        """
        if self.two_stream:
            return self.stream_encoder(context_tokens, context_pos, coords=coords)

        h = context_tokens + context_pos
        for blk in self.blocks:
            h = blk(h, coords=coords)
        if self.subspace_norm:
            h_c = self.norm_c(h[..., :self.carrier_dim])
            h_s = self.norm_s(h[..., self.carrier_dim:])
            return torch.cat([h_c, h_s], dim=-1)
        return self.norm(h)
