"""
Test-Time Adaptation (TTA) & Self-Calibration Module for Cross-Condition PECT-JEPA.

Physical & Theoretical Foundation:
In Pulsed Eddy Current Testing (PECT), >98% of any C-scan corresponds to sound metal.
Traditional neural UDA (CORAL, DANN, CDAN) aligns marginal distributions, which causes
the 98.8% sound-metal background to dominate the alignment loss and destroy defect boundaries.

Test-Time Adaptation (TTA) solves this via Unsupervised Spatial Self-Calibration:
1. Robust Trimmed Estimation: Identifies the sound-metal background distribution without
   labels by trimming the top outlier/defect energy pixels (e.g., top 5-25%).
2. Whitening Transformation: Centers and decorrelates the test scan's background:
   z_calibrated = (z - mu_sound) @ W_sound, where W_sound = (Sigma_sound + eps * I)^(-1/2).
3. Readout:
   - Unsupervised Mahalanobis Anomaly: ||z_calibrated||_2^2
   - Calibrated Linear/MLP Probes: evaluated on z_calibrated against a canonical sound standard.
   - Calibrated Flaw Sizing: monotonic relative perturbation mapping.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import torch


def whiten_matrix(cov: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """
    Computes symmetric whitening matrix W = (cov + eps * I)^(-1/2) via eigh.
    
    Args:
        cov: [D, D] covariance matrix.
        eps: Diagonal loading shrinkage parameter to damp noisy eigen-directions.
    Returns:
        W: [D, D] whitening projection matrix (float32).
    """
    D = cov.shape[0]
    reg_cov = cov.astype(np.float64) + np.eye(D, dtype=np.float64) * eps
    vals, vecs = np.linalg.eigh(reg_cov)
    vals = np.clip(vals, 1e-6, None)
    W = (vecs / np.sqrt(vals)).dot(vecs.T).astype(np.float32)
    return W


def unsup_trimmed_bg_stats(
    flat_z: np.ndarray,
    keep_q: float = 0.75,
    eps: float = 1e-2,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unsupervised outlier-trimmed sound-metal background estimator (Zero GT labels required).
    
    Steps:
    1. Filters valid pixels (if mask provided).
    2. Computes coordinate-wise median and Median Absolute Deviation (MAD).
    3. Ranks pixels by normalized energy distance: d^2 = mean(((z - med) / mad)^2).
    4. Trims the highest anomaly energy pixels, keeping the lowest `keep_q` (default 75%).
    5. Estimates sound-metal mean mu and shrinkage whitening matrix W on the trimmed subset.
    
    Args:
        flat_z: [N, D] feature matrix of the C-scan.
        keep_q: Fraction of lowest-energy pixels retained as sound metal (0.5 to 0.95).
        eps: Diagonal loading shrinkage (default 1e-2).
        valid_mask: Optional [N] boolean mask of valid scan coordinates.
    Returns:
        mu: [D] sound-metal background mean vector.
        W:  [D, D] sound-metal whitening matrix.
    """
    if valid_mask is not None:
        zv = flat_z[valid_mask].astype(np.float64)
    else:
        zv = flat_z.astype(np.float64)

    if len(zv) < 10:
        D = flat_z.shape[-1]
        return np.zeros(D, dtype=np.float32), np.eye(D, dtype=np.float32)

    med = np.median(zv, axis=0, keepdims=True)
    mad = np.median(np.abs(zv - med), axis=0, keepdims=True) + 1e-6
    d2 = np.mean(((zv - med) / mad) ** 2, axis=1)

    cut = float(np.quantile(d2, np.clip(keep_q, 0.1, 0.99)))
    keep = d2 <= cut
    sound_subset = zv[keep]
    if len(sound_subset) < sound_subset.shape[1] + 2:
        sound_subset = zv

    mu = np.mean(sound_subset, axis=0).astype(np.float32)
    cov = np.cov(sound_subset, rowvar=False)
    W = whiten_matrix(cov, eps=eps)
    return mu, W


class TestTimeSelfCalibrator:
    """
    Test-Time Self-Calibrator for PECT-JEPA C-Scans.
    Applies unsupervised background whitening per scan at inference time.
    """
    def __init__(self, keep_q: float = 0.75, eps: float = 1e-2):
        self.keep_q = keep_q
        self.eps = eps

    def calibrate_cscan(
        self,
        cscan_latents: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calibrates a full 2D C-scan latent map.
        
        Args:
            cscan_latents: [H, W, D] or [N, D] latent features.
            valid_mask: Optional [H, W] or [N] valid mask.
        Returns:
            calibrated_z: Transformed features with background normalized to zero-mean unit covariance.
            mu: Estimated background mean.
            W: Estimated whitening matrix.
        """
        orig_shape = cscan_latents.shape
        D = orig_shape[-1]
        flat_z = cscan_latents.reshape(-1, D)
        flat_mask = valid_mask.reshape(-1) if valid_mask is not None else None

        mu, W = unsup_trimmed_bg_stats(flat_z, keep_q=self.keep_q, eps=self.eps, valid_mask=flat_mask)
        calibrated_flat = (flat_z - mu).dot(W).astype(np.float32)
        calibrated_z = calibrated_flat.reshape(orig_shape)
        return calibrated_z, mu, W

    def calibrate(
        self,
        cscan_latents: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Alias returning calibrated features directly."""
        calibrated_z, _, _ = self.calibrate_cscan(cscan_latents, valid_mask=valid_mask)
        return calibrated_z

    def compute_anomaly_map(
        self,
        cscan_latents: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes unsupervised Mahalanobis anomaly energy map: Score = ||z_calibrated||_2^2.
        Direction-agnostic, threshold-free, zero-shot.
        """
        calibrated_z, _, _ = self.calibrate_cscan(cscan_latents, valid_mask=valid_mask)
        anomaly_map = np.sum(calibrated_z ** 2, axis=-1)
        return anomaly_map

    # Alias for compatibility with evaluation routines
    compute_anomaly_scores = compute_anomaly_map

