"""
Spatio-Temporal Tokenizer for PECT-JEPA v0.2 (Module 1, implement.md).
Processes input clip [B, H, W, T_c] frame-by-frame with spatial patch projection (P_s x P_s -> D)
and generates 3D Sinusoidal Positional Embeddings [1, H_t, W_t, T_c, D].
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class TemporalClipExtractor(nn.Module):
    """Utility to slice continuous temporal sequences into sliding clips."""
    def __init__(self, clip_length: int = 16, stride: int = 8):
        super().__init__()
        self.clip_length = clip_length
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., T] -> unfold on last dimension
        return x.unfold(dimension=-1, size=self.clip_length, step=self.stride)


class SpatioTemporal3DPositionalEmbedding(nn.Module):
    """
    Dynamic 3D Spatio-Temporal Positional Embedding for (H_t, W_t, T_c) token grid.
    Dynamically generates 3D sinusoidal positional embeddings combining spatial (h, w) and temporal (t) axes.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        pos_type: str = "sinusoidal"  # 'sinusoidal' or 'sinusoidal_projected'
    ):
        super().__init__()
        assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) must be divisible by 4 for 3D sinusoidal positional embeddings"
        self.embed_dim = embed_dim
        self.pos_type = pos_type
        if pos_type in ("learnable", "sinusoidal_projected"):
            self.proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.proj = nn.Identity()

    @staticmethod
    def _build_1d_sincos(n_pos: int, dim: int, device: torch.device) -> torch.Tensor:
        grid = torch.arange(n_pos, dtype=torch.float32, device=device)
        omega = torch.arange(dim // 2, dtype=torch.float32, device=device)
        omega /= (dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        out = torch.einsum('m,d->md', grid, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def forward(self, H_t: int, W_t: int, T_c: int, device: torch.device) -> torch.Tensor:
        """
        Generate 3D positional embeddings dynamically for grid (H_t, W_t, T_c).
        Returns:
            [1, H_t, W_t, T_c, embed_dim]
        """
        spatial_dim = self.embed_dim // 2
        temporal_dim = self.embed_dim // 2

        # 1D sincos for each axis
        h_emb = self._build_1d_sincos(H_t, spatial_dim // 2, device=device)  # [H_t, D/4]
        w_emb = self._build_1d_sincos(W_t, spatial_dim // 2, device=device)  # [W_t, D/4]
        t_emb = self._build_1d_sincos(T_c, temporal_dim, device=device)      # [T_c, D/2]

        # Spatial 2D pos: [H_t, W_t, D/2]
        h_expanded = h_emb.unsqueeze(1).expand(H_t, W_t, -1)
        w_expanded = w_emb.unsqueeze(0).expand(H_t, W_t, -1)
        spatial_pos = torch.cat([h_expanded, w_expanded], dim=-1)  # [H_t, W_t, D/2]

        # Combine spatial 2D and temporal 1D into 3D: [H_t, W_t, T_c, D]
        spatial_3d = spatial_pos.unsqueeze(2).expand(H_t, W_t, T_c, -1)
        temporal_3d = t_emb.view(1, 1, T_c, -1).expand(H_t, W_t, T_c, -1)
        pos_3d = torch.cat([spatial_3d, temporal_3d], dim=-1)  # [H_t, W_t, T_c, D]

        pos_3d = self.proj(pos_3d)
        return pos_3d.unsqueeze(0)  # [1, H_t, W_t, T_c, D]


class SpatioTemporalTokenizer(nn.Module):
    """
    Module 1: Spatio-Temporal Tokenizer (implement.md).
    Processes clip [B, H, W, T_c] frame-by-frame:
    - Slices spatial patches of size P_s x P_s at each frame t in [0..T_c-1].
    - Projects each patch (P_s * P_s) -> D via Linear + LayerNorm.
    - Generates 3D positional embeddings [1, H_t, W_t, T_c, D].
    """
    def __init__(
        self,
        spatial_patch: int = 8,
        clip_length: int = 16,
        embed_dim: int = 128,
        pos_embed_type: str = "sinusoidal",
        dropout: float = 0.0
    ):
        super().__init__()
        self.spatial_patch = spatial_patch
        self.clip_length = clip_length
        self.embed_dim = embed_dim

        self.patch_dim = spatial_patch * spatial_patch  # 8 * 8 = 64 per frame

        # Linear projection for each frame's spatial patch: 64 -> D
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 3D Positional Embedding Generator
        self.pos_embedding = SpatioTemporal3DPositionalEmbedding(
            embed_dim=embed_dim,
            pos_type=pos_embed_type
        )

    def forward(self, clip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            clip: Input clip tensor [B, H, W, T_c] (e.g. [B, 300, 300, 16])

        Returns:
            tokens: [B, M, D] flat token sequence (M = H_t * W_t * T_c)
            pos_embed: [1, M, D] flat 3D positional embeddings
            grid_shape: (H_t, W_t, T_c) e.g. (37, 37, 16)
        """
        orig_ndim = clip.ndim
        if orig_ndim == 3:
            clip = clip.unsqueeze(0)  # [1, H, W, T_c]

        B, H, W, T_c = clip.shape
        P_s = self.spatial_patch
        H_t = H // P_s
        W_t = W // P_s

        # Crop to exact divisible dimensions if needed
        clip = clip[:, :H_t * P_s, :W_t * P_s, :]

        # Reshape to [B, H_t, P_s, W_t, P_s, T_c]
        clip_reshaped = clip.view(B, H_t, P_s, W_t, P_s, T_c)

        # Permute to [B, H_t, W_t, T_c, P_s, P_s]
        clip_permuted = clip_reshaped.permute(0, 1, 3, 5, 2, 4).contiguous()

        # Flatten spatial patch dimension: [B, H_t, W_t, T_c, P_s * P_s]
        patches = clip_permuted.view(B, H_t, W_t, T_c, self.patch_dim)

        # Project patch (64) -> D (128) frame-by-frame: [B, H_t, W_t, T_c, D]
        tokens_3d = self.proj(patches)
        tokens_3d = self.norm(tokens_3d)
        tokens_3d = self.drop(tokens_3d)

        # Generate 3D Positional Embeddings: [1, H_t, W_t, T_c, D]
        pos_3d = self.pos_embedding(H_t, W_t, T_c, device=clip.device)

        # Flatten to sequences
        M = H_t * W_t * T_c
        tokens_flat = tokens_3d.view(B, M, self.embed_dim)
        pos_flat = pos_3d.view(1, M, self.embed_dim)

        return tokens_flat, pos_flat, (H_t, W_t, T_c)

    def forward_dense(self, clip: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Dense feature extraction with stride=1 (Dense Convolutional Patching).
        Output: [B, H, W, T_c, embed_dim] at native resolution (e.g. 300x300) without interpolation.
        """
        orig_ndim = clip.ndim
        if orig_ndim == 3:
            clip = clip.unsqueeze(0)

        B, H, W, T_c = clip.shape
        P_s = self.spatial_patch
        pad = P_s // 2

        # Permute to [B * T_c, 1, H, W]
        x_2d = clip.permute(0, 3, 1, 2).reshape(B * T_c, 1, H, W)
        x_padded = torch.nn.functional.pad(x_2d, (pad, pad - 1, pad, pad - 1), mode='replicate')

        # Dense sliding window with stride=1: [B * T_c, P_s * P_s, H * W]
        patches = torch.nn.functional.unfold(x_padded, kernel_size=(P_s, P_s), stride=(1, 1))
        patches = patches.transpose(1, 2)  # [B * T_c, H * W, P_s * P_s]

        # Linear projection to D=128
        tokens = self.proj(patches)  # [B * T_c, H * W, D]
        tokens = self.norm(tokens)
        tokens_5d = tokens.view(B, T_c, H, W, self.embed_dim).permute(0, 2, 3, 1, 4).contiguous()  # [B, H, W, T_c, D]

        return tokens_5d, (H, W, T_c)
