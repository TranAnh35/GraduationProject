"""
Spatial Grid Tokenizer for 5x5 PECT-JEPA.

Linearly projects the C-channel temporal waveform vector at each spatial coordinate (i, j)
into an embedding token of dimension D, combined with a 2D spatial positional embedding.
"""

import math
import torch
import torch.nn as nn


def build_2d_sinusoidal_pos_embedding(grid_size: int, embed_dim: int) -> torch.Tensor:
    """
    2D Sinusoidal Positional Embedding for a grid_size x grid_size grid.
    Returns: [1, grid_size * grid_size, embed_dim] float32 tensor.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal pos embedding"
    half_d = embed_dim // 2

    # Generate coordinates for (x, y)
    coords_y = torch.arange(grid_size, dtype=torch.float32)
    coords_x = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
    grid_y = grid_y.flatten()  # [N]
    grid_x = grid_x.flatten()  # [N]

    dim_t = torch.arange(half_d // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (2.0 * dim_t / half_d))

    out_x = torch.einsum("m,d->md", grid_x, omega)
    out_y = torch.einsum("m,d->md", grid_y, omega)

    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)  # [N, half_d]
    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)  # [N, half_d]

    pos_2d = torch.cat([pos_x, pos_y], dim=-1).unsqueeze(0)  # [1, N, embed_dim]
    return pos_2d


class SpatialGridTokenizer5x5(nn.Module):
    """
    Unified Tokenizer for 5x5 C-scan grid:
    Input: [B, 5, 5, C] -> Output: tokens [B, 25, D], pos [B, 25, D]
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 128,
        grid_size: int = 5,
        pos_embed_type: str = "learnable_2d",
        dropout: float = 0.0
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size  # 25
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        if pos_embed_type == "learnable_2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal_2d":
            pos = build_2d_sinusoidal_pos_embedding(grid_size, embed_dim)
            self.register_buffer("pos_embed", pos, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 5, 5, C] tensor (or [5, 5, C])

        Returns:
            tokens:     [B, 25, D]
            pos_expand: [B, 25, D]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, H, W, C = x.shape
        assert H == self.grid_size and W == self.grid_size, f"Expected {self.grid_size}x{self.grid_size}, got {H}x{W}"
        assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        x_flat = x.reshape(B, self.num_tokens, C)
        tokens = self.drop(self.norm(self.proj(x_flat)))  # [B, 25, D]
        pos = self.pos_embed.expand(B, -1, -1)           # [B, 25, D]
        return tokens, pos
