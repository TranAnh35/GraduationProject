"""
1D Temporal Tokenizer for PECT-JEPA (implement.md, Section 4.3).
Slices 1D waveform [B, T=500] into N=16 patches of length P=32,
projects each patch to embedding dimension D=128, and adds 1D Sinusoidal Positional Embeddings.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding1D(nn.Module):
    """Generates 1D Sinusoidal Positional Embeddings for sequence of length N."""
    def __init__(self, embed_dim: int = 128, pos_type: str = "sinusoidal"):
        super().__init__()
        assert embed_dim % 2 == 0, f"embed_dim ({embed_dim}) must be even"
        self.embed_dim = embed_dim
        self.pos_type = pos_type
        if pos_type in ("learnable", "sinusoidal_projected"):
            self.proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, num_positions: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            pos_emb: [1, num_positions, embed_dim]
        """
        grid = torch.arange(num_positions, dtype=torch.float32, device=device)
        omega = torch.arange(self.embed_dim // 2, dtype=torch.float32, device=device)
        omega /= (self.embed_dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        out = torch.einsum('m,d->md', grid, omega)
        pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # [num_positions, D]
        pos_emb = self.proj(pos_emb)
        return pos_emb.unsqueeze(0)  # [1, num_positions, D]


class TemporalTokenizer1D(nn.Module):
    """
    Module 1: 1D Temporal Tokenizer.
    Converts raw 1D waveform [B, T=500] into sequence of N=16 tokens [B, 16, D=128].
    """
    def __init__(
        self,
        time_samples: int = 500,
        patch_length: int = 32,
        stride: int = 31,
        embed_dim: int = 128,
        pos_embed_type: str = "sinusoidal",
        dropout: float = 0.0
    ):
        super().__init__()
        self.time_samples = time_samples
        self.patch_length = patch_length
        self.stride = stride
        self.embed_dim = embed_dim

        # Calculate expected number of patches
        # For T=500, P=32, stride=31: (500 - 32) // 31 + 1 = 16
        self.num_patches = (time_samples - patch_length) // stride + 1

        # Patch projection layer: P=32 -> D=128
        self.proj = nn.Linear(patch_length, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        # 1D Positional Embedding
        self.pos_embedding = SinusoidalPositionalEmbedding1D(
            embed_dim=embed_dim,
            pos_type=pos_embed_type
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Raw 1D waveform tensor [B, T=500] (or [T=500])

        Returns:
            tokens: [B, num_patches=16, embed_dim=128]
            pos_embed: [1, num_patches=16, embed_dim=128]
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, T]

        B, T = x.shape
        assert T >= self.patch_length, f"Waveform length T ({T}) must be >= patch_length ({self.patch_length})"

        # Slicing / Unfold into sliding 1D patches: [B, num_patches, patch_length]
        # unfold(dimension=-1, size=32, step=31)
        patches = x.unfold(dimension=-1, size=self.patch_length, step=self.stride)  # [B, 16, 32]
        num_extracted = patches.shape[1]

        # Linear projection to embed_dim D=128
        tokens = self.proj(patches)  # [B, 16, D]
        tokens = self.norm(tokens)
        tokens = self.drop(tokens)

        # Generate 1D positional embeddings
        pos_embed = self.pos_embedding(num_extracted, device=x.device)  # [1, 16, D]

        return tokens, pos_embed
