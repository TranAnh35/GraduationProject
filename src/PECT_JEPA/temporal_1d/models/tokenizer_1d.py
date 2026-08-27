"""
Log-time tokenizer with physical positional embedding for PECT-JEPA (Stage A2).

- Input: two-channel log-time waveform [B, 2, T'=128].
- Non-overlapping patches along log-time -> [B, N=16, 2*P] -> Linear -> [B, N, D].
- Positional embedding encodes PHYSICAL patch-center time (log-scaled), not token
  index, so the predictor can learn diffusion scaling (depth ~ sqrt(t)).
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from ..data.preprocessing import log_time_grid_ms, linear_time_grid_ms


def physical_log_time_pos_embedding(
    t_values: np.ndarray,
    embed_dim: int,
    span_scale: float = 8.0,
    eps: float = 1e-8,
    use_log: bool = True
) -> torch.Tensor:
    """
    Sinusoidal embedding of physical patch-center time.

    Args:
        t_values: [N] physical patch-center times (ms), ascending, > 0
        embed_dim: D (must be even)
        span_scale: horizontal frequency scale across the time span
        use_log: True -> normalize log(t) to [0, 1] (log-time axis);
                 False -> normalize t linearly to [0, 1] (raw/uniform axis)

    Returns:
        [1, N, D] float32 tensor
    """
    assert embed_dim % 2 == 0
    t = np.maximum(np.asarray(t_values, dtype=np.float64), eps)
    if use_log:
        y = np.log(t / t.min()) / max(np.log(t.max() / t.min()), eps)  # [0, 1]
    else:
        y = (t - t.min()) / max(t.max() - t.min(), eps)                # [0, 1]
    y = y * span_scale
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    out = np.einsum("m,d->md", y, omega)
    pos = np.concatenate([np.sin(out), np.cos(out)], axis=-1)
    return torch.from_numpy(pos.astype(np.float32)).unsqueeze(0)  # [1, N, D]


class TemporalTokenizer1D(nn.Module):
    """
    Two-channel tokenizer: [B, 2, T] -> tokens [B, N, D] + pos [1, N, D].

    mode='resampled' (B1/B2): T = log_time_samples (128), non-overlapping
    patches on the LOG-time grid; pos embeds physical patch-center LOG-time.
    mode='raw' (B0 baseline): T = padded raw length (e.g. 512), patches on the
    uniform raw axis; pos embeds physical patch-center LINEAR time (padding
    continues the time axis forward — content is padded, time is not mirrored).
    """

    def __init__(
        self,
        log_time_samples: int = 128,
        num_patches: int = 16,
        num_channels: int = 2,
        embed_dim: int = 128,
        pos_embed_type: str = "physical_log_time",
        t_total_ms: float = 5.0,
        t_start_frac: float = 0.02,
        dropout: float = 0.0,
        mode: str = "resampled"
    ):
        super().__init__()
        assert mode in ("resampled", "raw")
        self.mode = mode
        total_samples = log_time_samples  # in 'raw' mode this is the PADDED raw length
        assert total_samples % num_patches == 0, (
            f"input length ({total_samples}) must be divisible by num_patches ({num_patches})"
        )
        self.total_samples = total_samples
        self.num_patches = num_patches
        self.patch_length = total_samples // num_patches
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.pos_embed_type = pos_embed_type

        # Physical patch-center times (ms): log-time grid (resampled) or
        # uniform linear grid (raw)
        if mode == "resampled":
            grid = log_time_grid_ms(t_total_ms, total_samples, t_start_frac)
        else:
            grid = linear_time_grid_ms(t_total_ms, total_samples)
        self.register_buffer(
            "t_grid_ms",
            torch.from_numpy(np.asarray(grid, dtype=np.float32)),
            persistent=False,
        )

        # Patch projection: concatenated channels per patch -> D
        self.proj = nn.Linear(num_channels * self.patch_length, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        if pos_embed_type == "physical_log_time":
            self.register_buffer(
                "pos_embed",
                physical_log_time_pos_embedding(
                    self.patch_center_times_ms(), embed_dim,
                    use_log=(mode == "resampled"),
                ),
                persistent=False,
            )
        else:  # 'sinusoidal' fallback (token-index based)
            grid = torch.arange(num_patches, dtype=torch.float32)
            omega = torch.arange(embed_dim // 2, dtype=torch.float32)
            omega /= (embed_dim / 2.0)
            omega = 1.0 / (10000.0 ** omega)
            out = torch.einsum("m,d->md", grid, omega)
            pos = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
            self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)

    def patch_center_times_ms(self) -> np.ndarray:
        """Physical center time of each patch on the log-time grid."""
        t = self.t_grid_ms.numpy()
        centers = t.reshape(self.num_patches, self.patch_length).mean(axis=1)
        return centers

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 2, T'] (or [2, T'])

        Returns:
            tokens:     [B, N, D]
            pos_expand: [B, N, D]
            pos_embed:  [1, N, D]
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1, 2, T']
        B, C, T = x.shape
        assert C == self.num_channels, f"expected {self.num_channels} channels, got {C}"

        # [B, C, N, P] -> [B, N, C*P]
        patches = x.reshape(B, C, self.num_patches, self.patch_length)
        patches = patches.permute(0, 2, 1, 3).reshape(B, self.num_patches, C * self.patch_length)

        tokens = self.drop(self.norm(self.proj(patches)))  # [B, N, D]
        return tokens, self.pos_embed.expand(B, -1, -1), self.pos_embed
