"""
Modular Attention Mechanisms for PECT-JEPA (Section 16).
Implements:
1. MultiheadSelfAttention (Standard Full Attention)
2. FactorizedSpatioTemporalBlock (Spatial Self-Attention + Temporal Self-Attention)
3. WindowAttentionBlock (Local Spatial Window Attention + Temporal Attention)
4. MultiheadCrossAttention (Predictor cross-attention)
5. MLP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiheadSelfAttention(nn.Module):
    """Standard multi-head self-attention module."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        if hasattr(F, 'scaled_dot_product_attention') and key_padding_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                # key_padding_mask: [B, N] (True = ignore)
                attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            # Handle row-all-inf if key_padding_mask masked all tokens
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiheadCrossAttention(nn.Module):
    """
    Multi-head cross-attention where query attends to key-value memory.
    Used by the Predictor to attend to Context Encoder latents (Section 13).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # query: [B, N_q, D], key_value: [B, N_kv, D]
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if hasattr(F, 'scaled_dot_product_attention') and key_padding_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """Standard Feed-Forward Network."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FactorizedSpatioTemporalBlock(nn.Module):
    """
    Factorized Spatio-Temporal Attention Block (Section 16.3).
    Factorizes full attention into:
      1. Spatial self-attention across (H_t x W_t) for each temporal clip K_t
      2. Temporal self-attention across K_t for each spatial location (H_t x W_t)
      3. MLP feed-forward
    Scales efficiently with large spatial token counts.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        # Spatial Attention
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.spatial_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Temporal Attention
        self.norm_temporal = nn.LayerNorm(embed_dim)
        self.temporal_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # MLP
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(
        self,
        x_grid: torch.Tensor,
        key_padding_mask_grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_grid: [B, H_t, W_t, K_t, D] 3D token grid
            key_padding_mask_grid: [B, H_t, W_t, K_t] bool mask (True = masked / invalid)

        Returns:
            [B, H_t, W_t, K_t, D] processed 3D token grid
        """
        B, H_t, W_t, K_t, D = x_grid.shape

        # --- 1. Spatial Attention ---
        # Permute to [B, K_t, H_t * W_t, D] -> reshape to [B * K_t, H_t * W_t, D]
        x_sp = x_grid.permute(0, 3, 1, 2, 4).reshape(B * K_t, H_t * W_t, D)
        norm_sp = self.norm_spatial(x_sp)

        mask_sp = None
        if key_padding_mask_grid is not None:
            mask_sp = key_padding_mask_grid.permute(0, 3, 1, 2).reshape(B * K_t, H_t * W_t)

        attn_sp = self.spatial_attn(norm_sp, key_padding_mask=mask_sp)
        x_sp = x_sp + attn_sp

        # Reshape back to [B, H_t, W_t, K_t, D]
        x_grid = x_sp.view(B, K_t, H_t, W_t, D).permute(0, 2, 3, 1, 4).contiguous()

        # --- 2. Temporal Attention ---
        # Permute to [B, H_t, W_t, K_t, D] -> reshape to [B * H_t * W_t, K_t, D]
        x_tp = x_grid.view(B * H_t * W_t, K_t, D)
        norm_tp = self.norm_temporal(x_tp)

        mask_tp = None
        if key_padding_mask_grid is not None:
            mask_tp = key_padding_mask_grid.view(B * H_t * W_t, K_t)

        attn_tp = self.temporal_attn(norm_tp, key_padding_mask=mask_tp)
        x_tp = x_tp + attn_tp

        # Reshape back to [B, H_t, W_t, K_t, D]
        x_grid = x_tp.view(B, H_t, W_t, K_t, D)

        # --- 3. MLP ---
        x_grid = x_grid + self.mlp(self.norm_mlp(x_grid))

        return x_grid
