"""
Quantitative Information Preservation & Resampling Fidelity Benchmark for PECT.
Evaluates RRE (Relative Reconstruction Error %), PSNR (dB), Spectral Fidelity (Cosine Sim),
Peak Amplitude Error (%), and Peak Time Error (us) across:
- Resample sizes T' in [32, 64, 96, 128, 160, 256, 500]
- Resample modes: Log-Time vs Linear Uniform
- All 81 TDMS dataset files (grouped by Waveform, Sensor, Lift-off, Scenario).
"""

import os
import glob
import re
import numpy as np
from nptdms import TdmsFile
import pandas as pd
from typing import List, Dict, Tuple, Any

# Add project root to sys.path
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.temporal_1d.data.preprocessing import (
    log_time_resample,
    build_two_channel_input,
    normalize_waveforms,
    find_all_tdms_files,
    parse_metadata_from_path,
)


def linear_resample_1d(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear uniform resampling."""
    T = x.shape[-1]
    if T == n_out:
        return x.astype(np.float32)
    indices = np.linspace(0, T - 1, n_out)
    lo = np.floor(indices).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (indices - lo).astype(x.dtype)
    lo = np.clip(lo, 0, T - 1)
    out = x[..., lo] * (1.0 - w) + x[..., hi] * w
    return out.astype(np.float32)


def interpolate_back(x_resampled: np.ndarray, target_len: int, mode: str = "linear", t_start_frac: float = 0.02) -> np.ndarray:
    """
    Interpolates resampled signal back to original target_len (500) to measure reconstruction fidelity.
    """
    n_in = x_resampled.shape[-1]
    if n_in == target_len:
        return x_resampled.copy()
    
    orig_indices = np.arange(target_len, dtype=np.float64)
    
    if mode == "log":
        # Log grid positions in original scale
        i0 = max(1.0, float(t_start_frac) * (target_len - 1))
        log_positions = np.geomspace(i0, target_len - 1.0, n_in)
        # Interpolate from log_positions to orig_indices
        out = np.zeros_like(x_resampled, shape=(x_resampled.shape[0], target_len))
        for i in range(x_resampled.shape[0]):
            out[i] = np.interp(orig_indices, log_positions, x_resampled[i], left=x_resampled[i, 0], right=x_resampled[i, -1])
        return out.astype(np.float32)
    else:
        # Linear uniform positions
        lin_positions = np.linspace(0, target_len - 1, n_in)
        out = np.zeros_like(x_resampled, shape=(x_resampled.shape[0], target_len))
        for i in range(x_resampled.shape[0]):
            out[i] = np.interp(orig_indices, lin_positions, x_resampled[i])
        return out.astype(np.float32)


def compute_metrics(x_raw: np.ndarray, x_recon: np.ndarray, dt_us: float = 10.0) -> Dict[str, float]:
    """
    Computes fidelity metrics between raw [N, T=500] and reconstructed [N, T=500].
    """
    eps = 1e-8
    # 1. RRE (%): Relative Reconstruction Error
    diff_norm = np.linalg.norm(x_raw - x_recon, axis=-1)
    raw_norm = np.linalg.norm(x_raw, axis=-1) + eps
    rre = np.mean(diff_norm / raw_norm) * 100.0  # in %

    # 2. RMSE & PSNR (dB)
    mse = np.mean((x_raw - x_recon) ** 2, axis=-1)
    rmse = np.sqrt(mse + eps)
    data_range = np.ptp(x_raw, axis=-1) + eps
    psnr = np.mean(20.0 * np.log10(data_range / (rmse + eps)))

    # 3. Spectral Cosine Similarity
    fft_raw = np.abs(np.fft.rfft(x_raw, axis=-1))
    fft_rec = np.abs(np.fft.rfft(x_recon, axis=-1))
    dot = np.sum(fft_raw * fft_rec, axis=-1)
    norm_r = np.linalg.norm(fft_raw, axis=-1) + eps
    norm_c = np.linalg.norm(fft_rec, axis=-1) + eps
    spectral_sim = np.mean(dot / (norm_r * norm_c)) * 100.0  # in %

    # 4. Peak Amplitude & Time Error
    pk_idx_raw = np.argmax(np.abs(x_raw), axis=-1)
    pk_idx_rec = np.argmax(np.abs(x_recon), axis=-1)
    pk_amp_raw = np.abs(x_raw[np.arange(len(x_raw)), pk_idx_raw]) + eps
    pk_amp_rec = np.abs(x_recon[np.arange(len(x_recon)), pk_idx_rec])
    pk_amp_err = np.mean(np.abs(pk_amp_raw - pk_amp_rec) / pk_amp_raw) * 100.0  # in %
    pk_time_err_us = np.mean(np.abs(pk_idx_raw - pk_idx_rec)) * dt_us  # in us

    return {
        "RRE_pct": float(rre),
        "PSNR_dB": float(psnr),
        "Spectral_Sim_pct": float(spectral_sim),
        "Peak_Amp_Err_pct": float(pk_amp_err),
        "Peak_Time_Err_us": float(pk_time_err_us)
    }


def evaluate_dataset(data_dir: str = "data", n_samples_per_file: int = 50) -> pd.DataFrame:
    all_files = find_all_tdms_files(data_dir)
    print(f"Total TDMS files found: {len(all_files)}", flush=True)

    T_prime_list = [32, 64, 96, 128, 160, 256, 500]
    results = []

    for f_idx, fp in enumerate(all_files):
        meta = parse_metadata_from_path(fp)
        try:
            with TdmsFile.open(fp) as tdms:
                channel = tdms["Waveform"].channels()[0]
                total_len = len(channel)
                n_total_points = total_len // 500
                
                # Sample indices uniformly across scan grid
                sample_indices = np.linspace(0, n_total_points - 1, n_samples_per_file, dtype=int)
                
                # Read slices
                waveforms_list = []
                for s_idx in sample_indices:
                    start = s_idx * 500
                    end = start + 500
                    if end <= total_len:
                        waveforms_list.append(channel[start:end])
                
                if len(waveforms_list) == 0:
                    continue
                waveforms = np.array(waveforms_list, dtype=np.float32)

            norm_wf = normalize_waveforms(waveforms, normalization="peak_early")

            for T_prime in T_prime_list:
                for mode in ["linear", "log"]:
                    if mode == "linear":
                        resampled = linear_resample_1d(norm_wf, T_prime)
                        recon = interpolate_back(resampled, target_len=500, mode="linear")
                    else:
                        resampled = log_time_resample(norm_wf, n_out=T_prime, t_start_frac=0.02)
                        recon = interpolate_back(resampled, target_len=500, mode="log", t_start_frac=0.02)

                    metrics = compute_metrics(norm_wf, recon)
                    res_row = {
                        "file": os.path.basename(fp),
                        "sensor": meta["sensor"],
                        "waveform": meta["waveform"],
                        "liftoff": meta["liftoff"],
                        "T_prime": T_prime,
                        "mode": mode,
                        **metrics
                    }
                    results.append(res_row)

        except Exception as e:
            print(f"Error processing {fp}: {e}", flush=True)
            continue

        if (f_idx + 1) % 20 == 0 or (f_idx + 1) == len(all_files):
            print(f"Processed {f_idx + 1}/{len(all_files)} files...", flush=True)

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    df = evaluate_dataset("data", n_samples_per_file=50)
    
    print("\n" + "=" * 85)
    print("RESAMPLING FIDELITY SUMMARY ACROSS ALL 81 FILES (GROUPED BY T' & MODE)")
    print("=" * 85)
    summary = df.groupby(["T_prime", "mode"])[["RRE_pct", "PSNR_dB", "Spectral_Sim_pct", "Peak_Amp_Err_pct", "Peak_Time_Err_us"]].mean()
    print(summary.to_string())

    print("\n" + "=" * 85)
    print("BREAKDOWN AT T'=128 (GROUPED BY WAVEFORM TYPE & MODE)")
    print("=" * 85)
    wf_summary = df[df["T_prime"] == 128].groupby(["waveform", "mode"])[["RRE_pct", "PSNR_dB", "Spectral_Sim_pct", "Peak_Amp_Err_pct", "Peak_Time_Err_us"]].mean()
    print(wf_summary.to_string())

    print("\n" + "=" * 85)
    print("BREAKDOWN AT T'=128 (GROUPED BY SENSOR TYPE & MODE)")
    print("=" * 85)
    sensor_summary = df[df["T_prime"] == 128].groupby(["sensor", "mode"])[["RRE_pct", "PSNR_dB", "Spectral_Sim_pct", "Peak_Amp_Err_pct"]].mean()
    print(sensor_summary.to_string())

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/resampling_fidelity_benchmark.csv", index=False)
    print("\nDetailed results saved to results/resampling_fidelity_benchmark.csv")
