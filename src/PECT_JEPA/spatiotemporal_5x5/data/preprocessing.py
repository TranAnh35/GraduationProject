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
    apply_lowpass_filter,
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
    normalization: str = "file_peak",
    scale_factor: Optional[float] = None,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize waveforms along the temporal and spatial dimensions.
    - 'file_peak': divide by single scalar max |x| across the ENTIRE C-scan record
      (preserves 100% spatial amplitude contrast Delta V, universal for Absolute and Diffensors).
    - 'dataset_peak': divide by a constant scalar scale_factor across the entire dataset split.
    - 'per_sample_peak': legacy per-sample peak normalization (np.max(abs(x), axis=-1)).
    - 'global_peak': alias for 'file_peak' (true file-level peak).
    - 'zscore': (x - mean) / std.
    - 'min_max': scale to [0, 1].
    - 'none': pass-through.
    """
    if normalization in ("file_peak", "global_peak"):
        # True scalar peak over the entire C-scan record / array
        peak = float(np.max(np.abs(x)))
        return (x / (peak + eps)).astype(np.float32)
    elif normalization == "dataset_peak":
        if scale_factor is None or scale_factor <= 0:
            peak = float(np.max(np.abs(x)))
        else:
            peak = float(scale_factor)
        return (x / (peak + eps)).astype(np.float32)
    elif normalization == "per_sample_peak":
        # Legacy mode for strict backward compatibility
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
    "apply_lowpass_filter",
]
