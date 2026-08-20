"""
Multi-Scale Temporal Encoder for PECT Signals (Module A, Section 6).
Processes 500 raw temporal samples per scan point into T' latent temporal positions
using multi-scale Conv1D branches followed by a lightweight temporal Transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class Conv1DBranch(nn.Module):
    """Single Conv1D branch with configurable kernel size and dilation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        # Calculate padding to keep output length same as input length
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        # Use GroupNorm(1, C) so normalization is per-waveform and chunk-independent
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return self.act(self.norm(self.conv(x)))


class MultiScaleConvFrontEnd(nn.Module):
    """
    Multi-scale Conv1D front-end with short, mid, and long-range branches (Section 6.2 & 6.3).
    """
    def __init__(
        self,
        in_channels: int = 1,
        branch_channels: int = 32,
        kernel_sizes: List[int] = (5, 9, 15),
        dilations: List[int] = (1, 1, 2)
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must have equal length"
        self.branches = nn.ModuleList([
            Conv1DBranch(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=k,
                dilation=d
            )
            for k, d in zip(kernel_sizes, dilations)
        ])
        total_channels = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, branch_channels, kernel_size=1),
            nn.GroupNorm(1, branch_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_total, 1, L]
        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=1)  # [N_total, total_channels, L]
        fused = self.fusion(concat)  # [N_total, branch_channels, L]
        return fused


class TemporalPositionalEncoding(nn.Module):
    """Learnable 1D temporal positional encoding."""
    def __init__(self, num_positions: int, dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T', dim]
        return x + self.pos_embed


class MultiScaleTemporalEncoder(nn.Module):
    """
    Module A: Multi-scale Temporal Encoder.
    Transforms raw temporal acquisition [B, H, W, 500] -> [B, H, W, T'].
    T' is configurable (64 or 128).
    """
    def __init__(
        self,
        raw_samples: int = 500,
        t_prime: int = 64,
        kernel_sizes: List[int] = (5, 9, 15),
        dilations: List[int] = (1, 1, 2),
        hidden_dim: int = 32,
        transformer_blocks: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.0,
        out_dim: int = 1,
        chunk_size: int = 4096
    ):
        super().__init__()
        self.raw_samples = raw_samples
        self.t_prime = t_prime
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.chunk_size = chunk_size

        # Multi-scale Conv front-end
        self.conv_frontend = MultiScaleConvFrontEnd(
            in_channels=1,
            branch_channels=hidden_dim,
            kernel_sizes=kernel_sizes,
            dilations=dilations
        )

        # Temporal downsampling from raw_samples (500) to t_prime (64 or 128)
        self.downsample = nn.AdaptiveAvgPool1d(t_prime)

        # Linear projection to hidden_dim before Transformer
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temporal positional encoding
        self.pos_encoder = TemporalPositionalEncoding(num_positions=t_prime, dim=hidden_dim)

        # Lightweight temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_blocks,
            norm=nn.LayerNorm(hidden_dim)
        )

        # Final projection to output representation
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _forward_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        # Multi-scale Conv1D front-end: [chunk_size, hidden_dim, 500]
        conv_feats = self.conv_frontend(chunk)

        # Temporal downsampling: [chunk_size, hidden_dim, T']
        downsampled = self.downsample(conv_feats)

        # Transpose for Transformer: [chunk_size, T', hidden_dim]
        tokens = downsampled.transpose(1, 2)
        tokens = self.input_proj(tokens)
        tokens = self.pos_encoder(tokens)

        # Transformer encoding
        encoded = self.transformer(tokens)  # [chunk_size, T', hidden_dim]

        # Project to output dimension
        out = self.out_proj(encoded)  # [chunk_size, T', out_dim]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw acquisition tensor of shape [B, H, W, 500] (or [H, W, 500])

        Returns:
            Latent temporal representation of shape [B, H, W, T']
            (or [B, H, W, T', out_dim] if out_dim > 1)
        """
        orig_ndim = x.ndim
        if orig_ndim == 3:
            x = x.unsqueeze(0)  # [1, H, W, 500]

        B, H, W, L = x.shape
        assert L == self.raw_samples, f"Expected {self.raw_samples} temporal samples, got {L}"

        # Flatten spatial dimensions into batch: [B * H * W, 1, 500]
        total_points = B * H * W
        x_flat = x.view(total_points, 1, L)

        # Chunked processing to handle large spatial grids (e.g. 300x300 = 90,000 points) without memory explosion
        if total_points <= self.chunk_size:
            out = self._forward_chunk(x_flat)
        else:
            out_chunks = []
            for i in range(0, total_points, self.chunk_size):
                chunk = x_flat[i : i + self.chunk_size]
                out_chunk = self._forward_chunk(chunk)
                out_chunks.append(out_chunk)
            out = torch.cat(out_chunks, dim=0)

        if self.out_dim == 1:
            out = out.squeeze(-1)  # [B * H * W, T']
            out = out.view(B, H, W, self.t_prime)  # [B, H, W, T']
        else:
            out = out.view(B, H, W, self.t_prime, self.out_dim)

        if orig_ndim == 3:
            out = out.squeeze(0)

        return out
