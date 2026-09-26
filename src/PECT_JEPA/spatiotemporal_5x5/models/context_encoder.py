from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .attention import TransformerBlock, RadialAttentionBias
from ..data.topologies import get_spatial_topology_offsets


class ContextEncoder5x5(nn.Module):
    """
    Context Encoder processing visible context points:
    Input: context_tokens [B, N_ctx, D], context_pos [B, N_ctx, D]
    Output: H_ctx [B, N_ctx, D]
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_radial_attention_bias: bool = False,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        diffusion_gamma_init: float = 0.05,
        diffusion_alpha_init: float = 0.1,
    ):
        super().__init__()
        self.use_radial_attention_bias = use_radial_attention_bias
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

        if use_radial_attention_bias:
            self.radial_bias = RadialAttentionBias(
                num_heads=num_heads,
                init_gamma=diffusion_gamma_init,
                init_alpha=diffusion_alpha_init,
            )
            offsets = get_spatial_topology_offsets(
                topology=spatial_topology, star_radii=star_radii, grid_size=grid_size
            )  # [25, 2]
            diff = offsets[:, None, :] - offsets[None, :, :]
            dist_25 = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)  # [25, 25] in mm

            # Precompute 100x100 matrix for spatio-spectral 4-scale tokens
            dist_100 = dist_25[np.arange(100) // 4, :][:, np.arange(100) // 4]  # [100, 100]
            self.register_buffer("dist_matrix_100", torch.from_numpy(dist_100), persistent=False)
            self.register_buffer("dist_matrix_25", torch.from_numpy(dist_25), persistent=False)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        h = context_tokens + context_pos
        B, N_ctx, D = h.shape

        attn_bias = None
        if self.use_radial_attention_bias:
            if context_indices is not None:
                # context_indices: [B, N_ctx]
                dist_full = self.dist_matrix_100 if hasattr(self, "dist_matrix_100") and self.dist_matrix_100.shape[0] >= N_ctx else getattr(self, "dist_matrix_25", None)
                if dist_full is not None:
                    if dist_full.device != h.device:
                        dist_full = dist_full.to(h.device)
                    idx = context_indices.long()
                    dist_ctx = dist_full[idx.unsqueeze(2), idx.unsqueeze(1)]  # [B, N_ctx, N_ctx]
                    attn_bias = self.radial_bias(dist_ctx)  # [B, H, N_ctx, N_ctx]
            elif N_ctx == 100 and hasattr(self, "dist_matrix_100"):
                mat = self.dist_matrix_100 if self.dist_matrix_100.device == h.device else self.dist_matrix_100.to(h.device)
                attn_bias = self.radial_bias(mat)
            elif N_ctx == 25 and hasattr(self, "dist_matrix_25"):
                mat = self.dist_matrix_25 if self.dist_matrix_25.device == h.device else self.dist_matrix_25.to(h.device)
                attn_bias = self.radial_bias(mat)

        last_attn = None
        for i, blk in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                h, last_attn = blk(h, return_attention=True, attn_bias=attn_bias)
            else:
                h = blk(h, attn_bias=attn_bias)
        h = self.norm(h)
        if return_attention:
            return h, last_attn
        return h
