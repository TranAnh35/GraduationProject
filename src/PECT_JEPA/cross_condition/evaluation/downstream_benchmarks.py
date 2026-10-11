"""
Comprehensive Downstream Benchmark Suite for Cross-Condition PECT-JEPA.

Strict Protocols:
1. Pure Frozen Foundation Representations: Model weights are 100% frozen.
2. Defect Detection: Mahalanobis anomaly detection (AUC-ROC, Average Precision).
   AP-chance = sum(y == 1) / len(y).
3. Fastener Classification: Rivet_v1 / Rivet_v2 defect classification (AUC-ROC, Average Precision).
4. Physical Sizing (Conditional Hurdle Regression): Ridge / Linear Probe on defect areas (y > 0)
   measuring calibrated defect diameter (mm): R^2 score, MAE (mm), and Pearson r.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr


def compute_anomaly_detection_metrics(
    latents: np.ndarray,
    ground_truth_mask: np.ndarray,
    sound_metal_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluates unsupervised Mahalanobis distance anomaly detection on frozen latents.
    Args:
        latents: [H, W, D] dense latent feature map.
        ground_truth_mask: [H, W] boolean / integer mask (1 = defect, 0 = sound metal).
        sound_metal_mask: optional known sound metal mask for covariance calculation.
    Returns:
        Dict with auc_roc, average_precision, ap_chance, cnr.
    """
    H, W, D = latents.shape
    flat_latents = latents.reshape(-1, D)
    flat_labels = (ground_truth_mask.reshape(-1) > 0.5).astype(np.int64)

    if sound_metal_mask is not None:
        flat_sound = (sound_metal_mask.reshape(-1) > 0.5)
    else:
        flat_sound = (flat_labels == 0)

    # Compute sound metal baseline mean and regularized covariance
    z_sound = flat_latents[flat_sound]
    mu_sound = np.mean(z_sound, axis=0)
    cov_sound = np.cov(z_sound, rowvar=False) + np.eye(D) * 1e-4
    inv_cov = np.linalg.pinv(cov_sound)

    # Mahalanobis distance: sqrt((z - mu) @ inv_cov @ (z - mu)^T)
    diff = flat_latents - mu_sound
    # Fast vectorized Mahalanobis squared: sum((diff @ inv_cov) * diff, axis=-1)
    dist_sq = np.sum((diff @ inv_cov) * diff, axis=-1)
    dist = np.sqrt(np.maximum(dist_sq, 0.0))

    auc = float(roc_auc_score(flat_labels, dist))
    ap = float(average_precision_score(flat_labels, dist))
    ap_chance = float(np.mean(flat_labels))

    # Contrast-to-Noise Ratio (CNR): |mu_def - mu_snd| / sqrt(0.5*(var_def + var_snd))
    dist_def = dist[flat_labels == 1]
    dist_snd = dist[flat_labels == 0]
    mu_d = np.mean(dist_def) if len(dist_def) > 0 else 0.0
    mu_s = np.mean(dist_snd) if len(dist_snd) > 0 else 0.0
    var_d = np.var(dist_def) if len(dist_def) > 0 else 0.0
    var_s = np.var(dist_snd) if len(dist_snd) > 0 else 0.0
    cnr = float(abs(mu_d - mu_s) / max(1e-6, np.sqrt(0.5 * (var_d + var_s))))

    return {
        "auc_roc": auc,
        "average_precision": ap,
        "ap_chance": ap_chance,
        "cnr": cnr,
        "anomaly_scores": dist.reshape(H, W),
    }


def evaluate_fastener_classification(
    anomaly_scores: np.ndarray,
    defect_mask: np.ndarray,
    fastener_collar_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates fastener collar defect discrimination (Rivet structures):
    Distinguishes real fastener flaws from undamaged fastener collar responses.
    """
    flaw_pixels = anomaly_scores[defect_mask > 0.5]
    collar_pixels = anomaly_scores[(fastener_collar_mask > 0.5) & (defect_mask < 0.5)]

    if len(flaw_pixels) == 0 or len(collar_pixels) == 0:
        return {"fastener_auc": 0.0, "fastener_ap": 0.0}

    scores = np.concatenate([flaw_pixels, collar_pixels])
    labels = np.concatenate([np.ones(len(flaw_pixels)), np.zeros(len(collar_pixels))])

    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    return {
        "fastener_auc": auc,
        "fastener_ap": ap,
    }


def evaluate_conditional_sizing(
    features: np.ndarray,
    ground_truth_diameter_map: np.ndarray,
    flaw_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Two-Stage Hurdle Protocol: Evaluates quantitative defect sizing (diameter in mm)
    conditioned strictly on defect pixels (flaw_mask > 0).
    Uses Ridge regression with 5-fold cross-validation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    H, W, D = features.shape
    flat_X = features.reshape(-1, D)
    flat_y = ground_truth_diameter_map.reshape(-1)
    flat_flaw = flaw_mask.reshape(-1) > 0.5

    X_def = flat_X[flat_flaw]
    y_def = flat_y[flat_flaw]

    if len(y_def) < 20 or np.ptp(y_def) < 1e-4:
        return {"r2": 0.0, "mae_mm": 0.0, "pearson_r": 0.0}

    # 5-fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_preds = np.zeros_like(y_def)

    for train_idx, val_idx in kf.split(X_def):
        reg = Ridge(alpha=10.0)
        reg.fit(X_def[train_idx], y_def[train_idx])
        y_preds[val_idx] = reg.predict(X_def[val_idx])

    r2 = float(r2_score(y_def, y_preds))
    mae = float(mean_absolute_error(y_def, y_preds))
    r_val, _ = pearsonr(y_def, y_preds) if np.std(y_preds) > 1e-6 else (0.0, 0.0)

    return {
        "r2": r2,
        "mae_mm": mae,
        "pearson_r": float(r_val),
    }
