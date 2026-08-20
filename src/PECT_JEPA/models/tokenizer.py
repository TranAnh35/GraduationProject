"""
Temporal Clip Construction (Module B) & Spatio-Temporal Tokenizer (Module C).
Partitions latent temporal sequences into clips and tokens (P_s x P_s x T_c) with 3D positional embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalClipExtractor(nn.Module):
    """
    Module B: Temporal Clip Construction (Section 7).
    Extracts sliding temporal clips of length T_c with configurable stride.
    """
    def __init__(self, clip_length: int = 16, stride: int = 8):
        super().__init__()
        self.clip_length = clip_length
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent temporal representation of shape [B, H, W, T']
        Returns:
            Clips of shape [B, H, W, K_t, T_c]
        """
        B, H, W, T_prime = x.shape
        assert T_prime >= self.clip_length, f"T' ({T_prime}) must be >= clip_length ({self.clip_length})"

        # unfold along temporal axis (dim=-1)
        # Result shape: [B, H, W, K_t, T_c]
        clips = x.unfold(dimension=-1, size=self.clip_length, step=self.stride)
        return clips


class SpatioTemporal3DPositionalEmbedding(nn.Module):
    """
    Dynamic 3D Spatio-Temporal Positional Embedding for (H_t, W_t, K_t) token grid (Section 8.4).
    Dynamically generates 3D sinusoidal or learnable positional embeddings for arbitrary grid dimensions.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        pos_type: str = "sinusoidal"  # 'sinusoidal' (dynamic default) or 'learnable'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_type = pos_type
        # Optional learnable linear projection of sinusoidal embeddings
        if pos_type == "learnable":
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

    def forward(self, H_t: int, W_t: int, K_t: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional embeddings dynamically for grid (H_t, W_t, K_t).
        Works for arbitrary H_t, W_t, K_t (e.g. P_s=4 -> H_t=75).

        Returns:
            [1, H_t, W_t, K_t, embed_dim]
        """
        spatial_dim = self.embed_dim // 2
        temporal_dim = self.embed_dim // 2

        # 1D sincos for each axis
        h_emb = self._build_1d_sincos(H_t, spatial_dim // 2, device=device)  # [H_t, D/4]
        w_emb = self._build_1d_sincos(W_t, spatial_dim // 2, device=device)  # [W_t, D/4]
        t_emb = self._build_1d_sincos(K_t, temporal_dim, device=device)      # [K_t, D/2]

        # Spatial 2D pos: [H_t, W_t, D/2]
        h_expanded = h_emb.unsqueeze(1).expand(H_t, W_t, -1)
        w_expanded = w_emb.unsqueeze(0).expand(H_t, W_t, -1)
        spatial_pos = torch.cat([h_expanded, w_expanded], dim=-1)  # [H_t, W_t, D/2]

        # Combine with temporal 1D pos: [H_t, W_t, K_t, D]
        spatial_3d = spatial_pos.unsqueeze(2).expand(H_t, W_t, K_t, -1)
        temporal_3d = t_emb.view(1, 1, K_t, -1).expand(H_t, W_t, K_t, -1)
        pos_3d = torch.cat([spatial_3d, temporal_3d], dim=-1)  # [H_t, W_t, K_t, D]

        pos_3d = self.proj(pos_3d)
        return pos_3d.unsqueeze(0)  # [1, H_t, W_t, K_t, D]


class SpatioTemporalTokenizer(nn.Module):
    """
    Module C: Spatio-temporal Tokenization (Section 8).
    Maps local spatio-temporal regions [P_s, P_s, T_c] to D-dimensional token embeddings.
    """
    def __init__(
        self,
        spatial_patch: int = 8,
        clip_length: int = 16,
        stride: int = 8,
        embed_dim: int = 128,
        pos_embed_type: str = "learnable",
        dropout: float = 0.0
    ):
        super().__init__()
        self.spatial_patch = spatial_patch
        self.clip_length = clip_length
        self.stride = stride
        self.embed_dim = embed_dim

        self.clip_extractor = TemporalClipExtractor(clip_length=clip_length, stride=stride)

        # Region dimension: P_s * P_s * T_c
        self.patch_dim = spatial_patch * spatial_patch * clip_length

        # Linear projection to D
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 3D Positional Embedding generator
        self.pos_embedding = SpatioTemporal3DPositionalEmbedding(
            embed_dim=embed_dim,
            pos_type=pos_embed_type
        )

    def forward(self, x_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x_latent: [B, H, W, T'] from Temporal Encoder

        Returns:
            tokens: [B, M, D] (where M = H_t * W_t * K_t)
            pos_embed: [1, M, D]
            grid_shape: (H_t, W_t, K_t)
        """
        # Step 1: Temporal clip extraction -> [B, H, W, K_t, T_c]
        clips = self.clip_extractor(x_latent)
        B, H, W, K_t, T_c = clips.shape

        # Step 2: Spatial patch partition
        P_s = self.spatial_patch
        H_t = H // P_s
        W_t = W // P_s

        # Crop to divisible size if needed
        clips = clips[:, :H_t * P_s, :W_t * P_s, :, :]

        # Reshape to [B, H_t, P_s, W_t, P_s, K_t, T_c]
        clips = clips.view(B, H_t, P_s, W_t, P_s, K_t, T_c)

        # Permute to [B, H_t, W_t, K_t, P_s, P_s, T_c]
        clips = clips.permute(0, 1, 3, 5, 2, 4, 6).contiguous()

        # Flatten region to [B, H_t, W_t, K_t, P_s * P_s * T_c]
        regions = clips.view(B, H_t, W_t, K_t, self.patch_dim)

        # Step 3: Project to embed_dim D -> [B, H_t, W_t, K_t, D]
        tokens_3d = self.proj(regions)
        tokens_3d = self.norm(tokens_3d)
        tokens_3d = self.drop(tokens_3d)

        # Step 4: 3D Positional Embeddings
        pos_3d = self.pos_embedding(H_t, W_t, K_t, device=x_latent.device)  # [1, H_t, W_t, K_t, D]

        # Flatten to sequence: [B, M, D] and [1, M, D]
        M = H_t * W_t * K_t
        tokens = tokens_3d.view(B, M, self.embed_dim)
        pos_embed = pos_3d.view(1, M, self.embed_dim)

        return tokens, pos_embed, (H_t, W_t, K_t)
