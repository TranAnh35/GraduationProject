"""
Preprocessing and TDMS Reading for 1D Temporal PECT-JEPA.
Completely self-contained: reads TDMS files, extracts 1D transient waveforms,
resamples to target time points, and applies Min-Max per-file normalization.
"""

import os
import glob
import re
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from nptdms import TdmsFile


def parse_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Parse experiment metadata (sensor, waveform, defect type, liftoff) from file path and filename.
    """
    norm_path = os.path.normpath(file_path).replace("\\", "/")
    filename = os.path.basename(norm_path)

    # 1. Sensor type
    sensor = "Unknown"
    sensor_map = {
        "Differential_Pot_Core": "Differential_Pot_Core",
        "Hall_Air_Core": "Hall_Air_Core",
        "Hall_Pot_Core": "Hall_Pot_Core",
        "TMR": "TMR"
    }
    for key, val in sensor_map.items():
        if key.lower() in norm_path.lower():
            sensor = val
            break

    # 2. Excitation waveform
    waveform = "Unknown"
    waveform_map = {
        "Square": "Square",
        "Gaussian": "Gaussian",
        "Chirp": "Chirp"
    }
    for key, val in waveform_map.items():
        if key.lower() in norm_path.lower():
            waveform = val
            break

    # 3. Lift-off (z1, z2, z3)
    liftoff = "Unknown"
    for z in ["z1", "z2", "z3"]:
        if z in filename.lower() or z in norm_path.lower():
            liftoff = z
            break

    # 4. Defect type
    defect = "Unknown"
    if "defect" in norm_path.lower():
        defect = "Defect"
    elif "normal" in norm_path.lower() or "reference" in norm_path.lower():
        defect = "Normal"

    return {
        "file_path": file_path,
        "filename": filename,
        "sensor": sensor,
        "waveform": waveform,
        "liftoff": liftoff,
        "defect": defect
    }


def find_all_tdms_files(data_dir: str) -> List[str]:
    """
    Recursively search for all .tdms files in data_dir.
    Excludes .tdms_index files.
    """
    if not os.path.exists(data_dir):
        return []

    pattern = os.path.join(data_dir, "**", "*.tdms")
    files = glob.glob(pattern, recursive=True)
    valid_files = [f for f in files if not f.endswith(".tdms_index")]
    return sorted(valid_files)


def read_tdms_1d_waveforms(
    file_path: str,
    target_time_samples: int = 500,
    normalization: str = "min_max",
    raster_correction: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Reads a TDMS file and extracts all 1D transient waveforms as a 2D array [N_waveforms, target_time_samples].

    Args:
        file_path: Path to .tdms file
        target_time_samples: Resample length (default: 500)
        normalization: 'min_max' or 'none'
        raster_correction: Whether to rearrange bidirectional raster lines
        eps: Small epsilon for numerical stability

    Returns:
        waveforms: np.ndarray of shape [N_waveforms, target_time_samples] (float32)
    """
    with TdmsFile.read(file_path) as tdms_file:
        if "Freq_Sampling_SizeX_SizeY" in tdms_file:
            infor = tdms_file["Freq_Sampling_SizeX_SizeY"].channels()[0][:]
            f = float(infor[0])
            sampling = float(infor[1])
            sX = int(infor[2])
            sY = int(infor[3])
            samples = int(sampling / f)
        else:
            sX, sY, samples = 300, 300, 500

        if "Waveform" in tdms_file:
            raw_data = tdms_file["Waveform"].channels()[0][:]
        else:
            groups = tdms_file.groups()
            raw_data = groups[-1].channels()[0][:]

    n_waveforms = len(raw_data) // samples
    data_2d = np.reshape(raw_data[:n_waveforms * samples], (n_waveforms, samples)).astype(np.float32)

    # Downsample / resample temporal dimension if needed
    if samples != target_time_samples:
        indices = np.linspace(0, samples - 1, target_time_samples).astype(np.int64)
        data_2d = data_2d[:, indices]

    # Rearrange bidirectional raster lines if spatial width sX is known
    if raster_correction and sX > 0 and data_2d.shape[0] >= sX * 2:
        n_rows = data_2d.shape[0] // sX
        grid = data_2d[:n_rows * sX].reshape(n_rows, sX, target_time_samples)
        grid[1::2, :, :] = grid[1::2, ::-1, :]
        data_2d[:n_rows * sX] = grid.reshape(n_rows * sX, target_time_samples)

    if normalization == "min_max":
        f_min = float(np.min(data_2d))
        f_max = float(np.max(data_2d))
        data_2d = (data_2d - f_min) / (f_max - f_min + eps)

    return data_2d.astype(np.float32)


def linear_time_grid_ms(
    t_total_ms: float,
    n_out: int = 512
) -> np.ndarray:
    """
    Uniform (linear) physical time grid in ms, patch-center style:
    t_i = t_total_ms * (i + 0.5) / n_out. Used by the 'raw' tokenizer mode (B0),
    where padded samples keep a forward-continuing time axis (content is padded,
    time is not mirrored).
    """
    idx = np.arange(n_out, dtype=np.float64)
    return t_total_ms * (idx + 0.5) / float(n_out)


def pad_waveforms(
    x: np.ndarray,
    target_len: int,
    mode: str = "edge",
    eps: float = 1e-8
) -> np.ndarray:
    """
    Pad the last axis of x [..., T_raw] up to `target_len` (B0 baseline).

    Modes:
      'edge'    — replicate the last sample (monotone-safe for decaying tails)
      'reflect' — mirror the tail (keeps oscillatory texture, but a monotone
                  decay appears to "turn around" — physically questionable)
      'zero'    — zero-fill (sudden drop; generally worst for PECT tails)
    """
    x = np.asarray(x)
    T = x.shape[-1]
    if T > target_len:
        raise ValueError(f"input length {T} exceeds target {target_len}")
    if T == target_len:
        return x.astype(np.float32, copy=False)
    np_mode = {"edge": "edge", "reflect": "reflect", "zero": "constant"}[mode]
    kwargs = {"constant_values": 0.0} if mode == "zero" else {}
    pads = [(0, 0)] * (x.ndim - 1) + [(0, target_len - T)]
    return np.pad(x, pads, mode=np_mode, **kwargs).astype(np.float32)


def log_time_resample(
    x: np.ndarray,
    n_out: int = 128,
    t_start_frac: float = 0.02,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Resample the last axis of `x` onto a log-spaced time grid (linear interpolation).

    Physics rationale: PECT diffusion depth ~ sqrt(t), so a log-time grid places
    token density uniformly in depth instead of uniformly in raw sample index.

    Args:
        x: [..., T_raw] array (any number of leading dims)
        n_out: output length T'
        t_start_frac: fraction of the record where the log grid starts
        eps: numerical epsilon

    Returns:
        [..., n_out] array (float32)
    """
    T = x.shape[-1]
    i0 = max(1.0, float(t_start_frac) * (T - 1))
    pos = np.geomspace(i0, T - 1.0, n_out)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(x.dtype)
    lo = np.clip(lo, 0, T - 1)
    out = x[..., lo] * (1.0 - w) + x[..., hi] * w
    return out.astype(np.float32)


def log_time_grid_ms(
    t_total_ms: float,
    n_out: int = 128,
    t_start_frac: float = 0.02,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Physical time axis (ms) of the log-time grid — used by the physical
    positional embedding. Returns [n_out] array in ms, ascending.
    """
    t = np.geomspace(max(t_start_frac, eps) * t_total_ms, t_total_ms, n_out)
    return t.astype(np.float32)


def moving_rms_envelope(x: np.ndarray, win: int = 9, eps: float = 1e-8) -> np.ndarray:
    """
    Moving-RMS envelope along the last axis (edge-padded, length preserved).
    Pure numpy (no scipy dependency).
    """
    T = x.shape[-1]
    win = int(min(max(3, win), T))
    half = win // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(half, half)], mode="edge")
    p = xp ** 2
    cs = np.cumsum(p, axis=-1)
    cs = np.concatenate([np.zeros_like(cs[..., :1]), cs], axis=-1)
    env2 = (cs[..., win:] - cs[..., :-win]) / float(win)
    return np.sqrt(np.maximum(env2, eps * eps))


def normalize_waveforms(
    x: np.ndarray,
    normalization: str = "peak_early",
    early_window_frac: float = 0.10,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Per-file waveform normalization on the raw uniform-time axis.

    - 'peak_early': divide by the max |x| over the early-time window.
      Shape-preserving: the absolute decay shape (which carries depth /
      thickness information) is kept, only the nuisance gain is removed.
    - 'zscore': (x - mean) / std over the whole record.
    - 'min_max': legacy [0, 1] scaling (kept for ablation; destroys amplitude).
    """
    if normalization == "peak_early":
        T = x.shape[-1]
        w = max(1, int(early_window_frac * T))
        peak = np.max(np.abs(x[..., :w]), axis=-1, keepdims=True)
        global_peak = np.max(np.abs(x), axis=-1, keepdims=True)
        # Fallback to global peak if early window has negligible amplitude (e.g. Gaussian or Chirp)
        effective_peak = np.where(peak > 0.05 * global_peak, peak, global_peak)
        return (x / (effective_peak + eps)).astype(np.float32)
    elif normalization == "zscore":
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        return ((x - mu) / (sd + eps)).astype(np.float32)
    elif normalization == "min_max":
        mn = x.min(axis=-1, keepdims=True)
        mx = x.max(axis=-1, keepdims=True)
        return ((x - mn) / (mx - mn + eps)).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def build_two_channel_input(
    raw: np.ndarray,
    log_time_samples: int = 128,
    t_start_frac: float = 0.02,
    normalization: str = "peak_early",
    early_window_frac: float = 0.10,
    envelope_win: int = 9,
    eps: float = 1e-8,
    pad_to: Optional[int] = None,
    pad_mode: str = "edge"
) -> np.ndarray:
    """
    Stage A1 transform: raw [N, T_raw] -> two-channel input [N, 2, T'].

    Two modes:
      pad_to=None (default, B1/B2 'resampled'):
        1. Per-file normalization on raw axis (shape-preserving by default).
        2. Moving-RMS envelope of the normalized waveform.
        3. Log-time resample both channels -> T' = log_time_samples.
        4. Standardize the log-envelope channel (early-window stats only).
      pad_to=int (B0 'raw' baseline):
        No resampling. Both channels keep the raw time axis; the
        log-envelope channel is standardized with early-window stats on the
        RAW axis, then both channels are padded (content only) to `pad_to`.
    """
    raw = np.asarray(raw, dtype=np.float32)
    x = normalize_waveforms(raw, normalization, early_window_frac, eps)
    env = moving_rms_envelope(x, win=envelope_win, eps=eps)
    log_env = np.log(env + eps)

    if pad_to is None:
        ch0 = log_time_resample(x, log_time_samples, t_start_frac, eps)
        ch1 = log_time_resample(log_env, log_time_samples, t_start_frac, eps)
        n_early = max(1, int(early_window_frac * log_time_samples))
    else:
        # B0 raw baseline: no resampling, keep full raw fidelity
        ch0 = x
        ch1 = log_env
        n_early = max(1, int(early_window_frac * x.shape[-1]))

    # Standardize the log-envelope channel using EARLY-WINDOW statistics only.
    # Using whole-record stats would let late-time (target) information rescale
    # the early-time (context) tokens — a genuine leakage channel for the JEPA
    # context/target split (caught by test_leakage_1d).
    mu = ch1[..., :n_early].mean(axis=-1, keepdims=True)
    sd = ch1[..., :n_early].std(axis=-1, keepdims=True)
    # Lower-bound sd to prevent division-by-near-zero on flat baseline windows
    sd = np.maximum(sd, 0.1)
    ch1 = (ch1 - mu) / sd

    # Numerical clip to strictly prevent FP16 overflow under mixed-precision AMP
    ch0 = np.clip(ch0, -10.0, 10.0)
    ch1 = np.clip(ch1, -10.0, 10.0)

    if pad_to is not None:
        # Pad CONTENT (time axis continues forward in the tokenizer's pos embed)
        ch0 = pad_waveforms(ch0, pad_to, pad_mode)
        ch1 = pad_waveforms(ch1, pad_to, pad_mode)

    return np.stack([ch0, ch1], axis=1).astype(np.float32)  # [N, 2, T']

