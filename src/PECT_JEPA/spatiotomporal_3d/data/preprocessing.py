"""
Preprocessing routines for PECT (Pulsed Eddy Current Testing) data.
Implements per-file normalization and raster scan conversion according to
Section 4 and reference implementation.
"""

import numpy as np
import torch
from typing import Union, Tuple
from scipy.signal import firwin, lfilter


def reshape_raster(data: np.ndarray, sX: int) -> np.ndarray:
    """
    Reconstruct 2D spatial raster scan from 1D continuous acquisition points.
    Alternate rows are reversed due to serpentine scan path.

    Args:
        data: Array of shape [total_points, samples]
        sX: Number of columns (points per scan line)

    Returns:
        2D raster mapped array of shape [sY, sX, samples]
    """
    num_rows = data.shape[0] // sX
    arr = []
    for i in range(num_rows):
        xi = data[sX * i : sX * (i + 1)]
        if i % 2 == 1:
            xi = xi[::-1]
        arr.append(xi)
    return np.array(arr)


def normalize_per_file(
    data: Union[np.ndarray, torch.Tensor],
    method: str = "min_max",
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply per-file deterministic normalization independently to each acquisition.
    Removes global amplitude differences across diverse experimental setups (Section 4).

    Args:
        data: Acquisition array or tensor of shape [..., samples] or [N, N, samples]
        method: 'min_max', 'standard', or 'none'
        eps: Small constant to avoid division by zero

    Returns:
        Normalized array or tensor with the same shape
    """
    if method == "none":
        return data

    is_torch = isinstance(data, torch.Tensor)

    if is_torch:
        if method == "min_max":
            v_min = torch.min(data)
            v_max = torch.max(data)
            return (data - v_min) / (v_max - v_min + eps)
        elif method == "standard":
            mean = torch.mean(data)
            std = torch.std(data)
            return (data - mean) / (std + eps)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        if method == "min_max":
            v_min = np.min(data)
            v_max = np.max(data)
            return (data - v_min) / (v_max - v_min + eps)
        elif method == "standard":
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + eps)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def lowpass_filter_1d(
    x: np.ndarray,
    cutoff: float = 0.04,
    order: int = 24
) -> np.ndarray:
    """
    Apply FIR lowpass filter with Hamming window along temporal axis.
    """
    b = firwin(order + 1, cutoff, window='hamming', pass_zero="lowpass")
    padded = np.pad(x, (50, 50), 'constant', constant_values=(x[0], x[-1]))
    y = lfilter(b, 1.0, padded)
    return y[50 + order // 2 : -50 + order // 2]
