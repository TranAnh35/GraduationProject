"""
Physical Feature Extraction for Raw PECT Signals based on authoritative NDT reference methods.
Implements:
1. Filtered Peak-to-Peak (LPF + DC Baseline Offset Removal)
2. Slice Feature (Transient Window Energy / Mean Absolute Amplitude)
3. LOI Feature (Lift-Off Intersection / Temporal Derivative Peak)
4. RMS Feature (Root-Mean-Square Energy)
5. Multi-method 4-Panel Raw Feature Extraction
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
    slice_window: Optional[Tuple[int, int]] = None,
    loi_window: Optional[Tuple[int, int]] = None,
    lpf_cutoff: float = 0.15,
    lpf_order: int = 2,
) -> np.ndarray:
    """
    Extracts a 2D physical raw signal heatmap [H, W] from a [H, W, T] C-scan.

    Methods:
        - 'filtered_ptp': Zero-phase LPF + baseline subtraction + Peak-to-Peak (authoritative NDT default)
        - 'ptp': Standard Peak-to-Peak (max - min)
        - 'slice': Mean absolute transient amplitude within slice_window
        - 'loi': Peak absolute temporal gradient (derivative) within loi_window
        - 'rms': Root-mean-square amplitude across all temporal samples
    """
    H, W, T = cscan.shape
    method = method.lower()

    if slice_window is None:
        # Default transient active window: 10% to 60% of temporal response
        slice_window = (int(0.10 * T), int(0.60 * T))
    if loi_window is None:
        # Default early derivative window for lift-off point of intersection
        loi_window = (int(0.05 * T), int(0.40 * T))

    if method == "filtered_ptp":
        filt = filter_cscan_waveforms(cscan, cutoff=lpf_cutoff, order=lpf_order, subtract_baseline=True)
        return np.ptp(filt, axis=-1).astype(np.float32)

    elif method == "ptp":
        return np.ptp(cscan, axis=-1).astype(np.float32)

    elif method == "slice":
        w0, w1 = max(0, slice_window[0]), min(T, slice_window[1])
        sub = cscan[:, :, w0:w1]
        return np.mean(np.abs(sub), axis=-1).astype(np.float32)

    elif method == "loi":
        # Temporal gradient / derivative
        grad = np.diff(cscan, axis=-1)
        w0, w1 = max(0, loi_window[0]), min(grad.shape[-1], loi_window[1])
        sub_grad = grad[:, :, w0:w1]
        return np.max(np.abs(sub_grad), axis=-1).astype(np.float32)

    elif method == "rms":
        return np.sqrt(np.mean(cscan ** 2, axis=-1)).astype(np.float32)

    else:
        # Fallback to standard PTP
        return np.ptp(cscan, axis=-1).astype(np.float32)


def extract_all_raw_heatmaps(
    cscan: np.ndarray,
    lpf_cutoff: float = 0.15,
    lpf_order: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Extracts all 4 authoritative raw feature heatmaps from references:
    1. Slice (Transient window integration)
    2. LOI (Gradient / Derivative peak)
    3. PTP (Raw Peak-to-Peak)
    4. Filtered PTP (LPF + Baseline-corrected Peak-to-Peak)
    """
    H, W, T = cscan.shape
    filt = filter_cscan_waveforms(cscan, cutoff=lpf_cutoff, order=lpf_order, subtract_baseline=True)

    # 1. Slice (Window [15, 65] on 128 samples)
    w0, w1 = int(0.12 * T), int(0.55 * T)
    feat_slice = np.mean(np.abs(filt[:, :, w0:w1]), axis=-1).astype(np.float32)

    # 2. LOI (Gradient on filtered waveforms)
    grad_filt = np.diff(filt, axis=-1)
    gw0, gw1 = int(0.05 * T), int(0.35 * T)
    feat_loi = np.max(np.abs(grad_filt[:, :, gw0:gw1]), axis=-1).astype(np.float32)

    # 3. Peak / Raw PTP
    feat_ptp = np.ptp(cscan, axis=-1).astype(np.float32)

    # 4. Filtered PTP (cleanest defect contrast)
    feat_filt_ptp = np.ptp(filt, axis=-1).astype(np.float32)

    return {
        "slice": feat_slice,
        "loi": feat_loi,
        "ptp": feat_ptp,
        "filtered_ptp": feat_filt_ptp,
    }
