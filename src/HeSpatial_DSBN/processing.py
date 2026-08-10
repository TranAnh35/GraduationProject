"""
Signal Processing and Metric Evaluation Utilities for HeSpatial-DSBN.
Standalone module - no external project dependencies.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Union


# =============================================================================
# NORMALIZATION & SIGNAL CLEANING
# =============================================================================

def norm_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-Max normalization to [0, 1]."""
    x_min = np.min(x, axis=-2, keepdims=True)
    x_max = np.max(x, axis=-2, keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


def norm_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-Score normalization (zero mean, unit variance)."""
    mean = np.mean(x, axis=-2, keepdims=True)
    std = np.std(x, axis=-2, keepdims=True)
    return (x - mean) / (std + eps)


def remove_baseline(data: np.ndarray, baseline_pts: int = 20) -> np.ndarray:
    """Subtract baseline offset from initial quiet samples."""
    baseline = np.mean(data[..., :baseline_pts, :], axis=-2, keepdims=True)
    return data - baseline


def svd_denoise(data_2d: np.ndarray, rank_keep: int = 3) -> np.ndarray:
    """Denoise 2D scan raster (N, T) using Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(data_2d, full_matrices=False)
    S_clean = np.zeros_like(S)
    S_clean[:rank_keep] = S[:rank_keep]
    return np.dot(U * S_clean, Vt)


# =============================================================================
# QUANTITATIVE EVALUATION METRICS
# =============================================================================

def compute_cnr(error_map: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR).
    CNR = |mean_defect - mean_normal| / sqrt(0.5 * (std_defect^2 + std_normal^2))
    """
    mask_bool = mask.astype(bool)
    defect_vals = error_map[mask_bool]
    normal_vals = error_map[~mask_bool]

    if len(defect_vals) == 0 or len(normal_vals) == 0:
        return 0.0

    mu_d, sigma_d = np.mean(defect_vals), np.std(defect_vals)
    mu_n, sigma_n = np.mean(normal_vals), np.std(normal_vals)

    denom = np.sqrt(0.5 * (sigma_d**2 + sigma_n**2) + 1e-8)
    return float(np.abs(mu_d - mu_n) / denom)


def compute_ncc(signal_true: np.ndarray, signal_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between 1D signals or tensors.
    """
    s = signal_true - np.mean(signal_true)
    sp = signal_pred - np.mean(signal_pred)
    
    num = np.sum(s * sp)
    den = np.sqrt(np.sum(s**2) * np.sum(sp**2)) + eps
    return float(num / den)


def compute_ssim_1d(s_true: np.ndarray, s_pred: np.ndarray, K1: float = 0.01, K2: float = 0.03) -> float:
    """
    Compute 1D Structural Similarity Index (SSIM) between true and predicted waveforms.
    """
    mu1 = np.mean(s_true)
    mu2 = np.mean(s_pred)
    sigma1_sq = np.var(s_true)
    sigma2_sq = np.var(s_pred)
    sigma12 = np.cov(s_true.flatten(), s_pred.flatten())[0, 1]

    L = np.max(s_true) - np.min(s_true) + 1e-8
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_val)


def compute_pixel_auc(error_map: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Pixel-level ROC-AUC and PR-AUC given error map and Ground-Truth binary mask.
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    y_true = mask.flatten().astype(int)
    y_score = error_map.flatten()

    if len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.5, "pr_auc": 0.5}

    roc_auc = float(roc_auc_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def evaluate_reconstruction_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    Compute full suite of metrics: MAE, MSE, NCC, SSIM, and optional CNR & ROC-AUC.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean(np.square(y_true - y_pred)))
    ncc = compute_ncc(y_true, y_pred)
    ssim = compute_ssim_1d(y_true, y_pred)

    res = {
        "mae": mae,
        "mse": mse,
        "ncc": ncc,
        "ssim": ssim,
    }

    if mask is not None:
        # If y_true & y_pred are 3D (sY, sX, T), compute 2D error map (MAE per pixel)
        if y_true.ndim == 3:
            error_map = np.mean(np.abs(y_true - y_pred), axis=-1)
        else:
            error_map = np.abs(y_true - y_pred)
            
        res["cnr"] = compute_cnr(error_map, mask)
        auc_res = compute_pixel_auc(error_map, mask)
        res["roc_auc"] = auc_res["roc_auc"]
        res["pr_auc"] = auc_res["pr_auc"]

    return res
