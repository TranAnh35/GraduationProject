"""
Physical Feature Extraction for Raw PECT Signals based on authoritative NDT reference methods.
Implements:
1. Filtered Peak-to-Peak (LPF + DC Baseline Offset Removal)
2. Raw Peak-to-Peak (Standard V_pp)
3. RMS Feature (Root-Mean-Square Energy)

Note: Chronological temporal windowing (Slice, LOI) is permanently removed to comply with
the multi-waveform physical invariant (GEMINI.md): temporal slicing breaks under Chirp
frequency dispersion and Gaussian centered wavepackets.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt


def filter_cscan_waveforms(
    cscan: np.ndarray,
    cutoff: float = 0.15,
    order: int = 2,
    subtract_baseline: bool = True,
) -> np.ndarray:
    """
    Applies zero-phase lowpass Butterworth filtering and initial DC offset removal
    along the temporal axis (axis=-1) to eliminate high-frequency scanner jitter.
    
    Args:
        cscan: [H, W, T] float32 array
        cutoff: normalized cutoff frequency (0 < cutoff < 1, relative to Nyquist)
        order: Butterworth filter order
        subtract_baseline: if True, subtracts initial time sample cscan[..., 0:1]
    Returns:
        filtered: [H, W, T] float32 array
    """
    H, W, T = cscan.shape
    data = cscan.astype(np.float32)

    if subtract_baseline:
        data = data - data[:, :, :1]

    # Design stable second-order sections (SOS) zero-phase filter
    sos = butter(order, cutoff, btype="lowpass", output="sos")
    # Reshape to 2D [H*W, T] for fast batch filtering along axis=-1
    flat = data.reshape(-1, T)
    filtered = sosfiltfilt(sos, flat, axis=-1).reshape(H, W, T)

    if subtract_baseline:
        filtered = filtered - filtered[:, :, :1]

    return filtered.astype(np.float32)


def extract_cscan_raw_features(
    cscan: np.ndarray,
    method: str = "filtered_ptp",
    lpf_cutoff: float = 0.15,
    lpf_order: int = 2,
) -> np.ndarray:
    """
    Extracts a 2D physical raw signal heatmap [H, W] from a [H, W, T] C-scan.

    Methods:
        - 'filtered_ptp': Zero-phase LPF + baseline subtraction + Peak-to-Peak (authoritative NDT default)
        - 'ptp': Standard Peak-to-Peak (max - min)
        - 'rms': Root-mean-square amplitude across all temporal samples
    """
    method = method.lower()

    if method == "filtered_ptp":
        filt = filter_cscan_waveforms(cscan, cutoff=lpf_cutoff, order=lpf_order, subtract_baseline=True)
        return np.ptp(filt, axis=-1).astype(np.float32)

    elif method == "ptp":
        return np.ptp(cscan, axis=-1).astype(np.float32)

    elif method == "rms":
        return np.sqrt(np.mean(cscan ** 2, axis=-1)).astype(np.float32)

    else:
        # Fallback to filtered PTP
        filt = filter_cscan_waveforms(cscan, cutoff=lpf_cutoff, order=lpf_order, subtract_baseline=True)
        return np.ptp(filt, axis=-1).astype(np.float32)


def extract_all_raw_heatmaps(
    cscan: np.ndarray,
    lpf_cutoff: float = 0.15,
    lpf_order: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Extracts authoritative raw feature heatmaps (Raw PTP and Filtered PTP):
    1. PTP (Raw Peak-to-Peak)
    2. Filtered PTP (LPF + Baseline-corrected Peak-to-Peak)
    """
    filt = filter_cscan_waveforms(cscan, cutoff=lpf_cutoff, order=lpf_order, subtract_baseline=True)

    feat_ptp = np.ptp(cscan, axis=-1).astype(np.float32)
    feat_filt_ptp = np.ptp(filt, axis=-1).astype(np.float32)

    return {
        "ptp": feat_ptp,
        "filtered_ptp": feat_filt_ptp,
    }
