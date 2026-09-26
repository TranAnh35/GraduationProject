"""
Attention building blocks for 5x5 Spatiotemporal PECT-JEPA Transformer.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention with Pre-LN and SDPA numerical stability."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()  # [B, H, N, d]

        if attn_bias is not None and attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(dtype=q.dtype)

        attn_weights = None
        if return_attention or not hasattr(F, "scaled_dot_product_attention"):
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attn_scores = torch.clamp(attn_scores, min=-65000.0, max=65000.0)
            attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)
            attn_drop = self.drop(attn_weights)
            out = attn_drop @ v
        else:
            # FlashAttention / Memory-Efficient attention with internal FP32 accumulation
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        if return_attention:
            return out, attn_weights
        return out


class MultiheadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention from Query to Key/Value memory with SDPA numerical stability."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N_q, D = query.shape
        _, N_kv, _ = key_value.shape

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if attn_bias is not None and attn_bias.dtype != q.dtype:
            attn_bias = attn_bias.to(dtype=q.dtype)

        if hasattr(F, "scaled_dot_product_attention"):
            # FlashAttention / Memory-Efficient attention with internal FP32 accumulation
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )
        else:
            # Fallback: clamp attention logits and compute softmax in FP32 to prevent FP16 overflow
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attn_scores = torch.clamp(attn_scores, min=-65000.0, max=65000.0)
            attn = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)
            attn = self.drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU activation."""
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block."""
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True, attn_bias=attn_bias)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class RadialAttentionBias(nn.Module):
    """
    Continuous Physical Radial Distance Attention Bias.
    Computes B_radial(i, j) = -softplus(gamma_h) * d_ij^2 - softplus(alpha_h) * ln(1 + d_ij^2)
    where d_ij is the true Euclidean distance between probes in millimeters.
    """
    def __init__(
        self,
        num_heads: int = 4,
        init_gamma: float = 0.05,
        init_alpha: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Learnable decay rates per head
        self.log_gamma = nn.Parameter(torch.full((1, num_heads, 1, 1), math.log(max(1e-4, init_gamma))))
        self.log_alpha = nn.Parameter(torch.full((1, num_heads, 1, 1), math.log(max(1e-4, init_alpha))))

    def forward(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist_matrix: [N, N] or [B, N, N] pairwise physical Euclidean distance in mm.
        Returns:
            attn_bias: [1, H, N, N] or [B, H, N, N] attention bias tensor.
        """
        gamma = torch.exp(self.log_gamma)  # [1, H, 1, 1] > 0
        alpha = torch.exp(self.log_alpha)  # [1, H, 1, 1] > 0
        if dist_matrix.ndim == 2:
            d = dist_matrix.unsqueeze(0).unsqueeze(0).to(gamma.device)  # [1, 1, N, N]
        elif dist_matrix.ndim == 3:
            d = dist_matrix.unsqueeze(1).to(gamma.device)  # [B, 1, N, N]
        else:
            d = dist_matrix.to(gamma.device)

        d2 = d ** 2
        # Parabolic Green's diffusion log-polynomial decay: -gamma * d^2 - alpha * ln(1 + d^2)
        bias = - gamma * d2 - alpha * torch.log1p(d2)  # [B/1, H, N, N]
        return bias
