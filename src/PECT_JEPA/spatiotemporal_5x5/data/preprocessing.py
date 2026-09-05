"""
Data Preprocessing for 5x5 PECT-JEPA.
Supports Pure Linear Resampling [128] and legacy Dual-Channel [256].
"""

from typing import Optional
import numpy as np

from ...temporal_1d.data.preprocessing import (
    parse_metadata_from_path,
    find_all_tdms_files,
    read_tdms_1d_waveforms,
    linear_time_grid_ms,
    pad_waveforms,
    log_time_resample,
    log_time_grid_ms,
    moving_rms_envelope,
    normalize_waveforms,
    build_two_channel_input,
)


def linear_time_resample(x: np.ndarray, n_out: int = 128) -> np.ndarray:
    """
    Resample the temporal dimension (last axis) from T_raw to n_out points
    uniformly using linear interpolation.
    
    Args:
        x: [..., T_raw] numpy array
        n_out: target number of time steps (default: 128)
    Returns:
        [..., n_out] float32 numpy array
    """
    T = x.shape[-1]
    if T == n_out:
        return x.astype(np.float32, copy=False)

    pos = np.linspace(0, T - 1, n_out)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(x.dtype)
    out = x[..., lo] * (1.0 - w) + x[..., hi] * w
    return out.astype(np.float32)


def normalize_waveforms_linear(
    x: np.ndarray,
    normalization: str = "global_peak",
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize waveforms along the temporal axis.
    - 'global_peak': divide by max |x| across entire record (shape-preserving, universal for all waveforms).
    - 'zscore': (x - mean) / std.
    - 'min_max': scale to [0, 1].
    - 'none': pass-through.
    """
    if normalization == "global_peak":
        peak = np.max(np.abs(x), axis=-1, keepdims=True)
        return (x / (peak + eps)).astype(np.float32)
    elif normalization == "zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        return ((x - mu) / (sd + eps)).astype(np.float32)
    elif normalization == "min_max":
        mn = x.min(axis=-1, keepdims=True)
        mx = x.max(axis=-1, keepdims=True)
        return ((x - mn) / (mx - mn + eps)).astype(np.float32)
    elif normalization == "none":
        return x.astype(np.float32)
    else:
        return normalize_waveforms(x, normalization=normalization, eps=eps)


__all__ = [
    "parse_metadata_from_path",
    "find_all_tdms_files",
    "read_tdms_1d_waveforms",
    "linear_time_grid_ms",
    "linear_time_resample",
    "normalize_waveforms_linear",
    "pad_waveforms",
    "log_time_resample",
    "log_time_grid_ms",
    "moving_rms_envelope",
    "normalize_waveforms",
    "build_two_channel_input",
]
