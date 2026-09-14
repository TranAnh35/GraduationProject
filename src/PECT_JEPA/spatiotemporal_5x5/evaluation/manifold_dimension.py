"""
Manifold Intrinsic Dimension & Eigenspectrum Analysis for PECT-JEPA.

Addresses the Advisor's critique on Linear PCA vs. True Neural Representation:
1. "PCA always projects maximum variance onto PC1 (dominated by the massive sound metal baseline)."
2. "Linear Effective Rank cannot distinguish between representation collapse and sound-metal dominance."

This module implements:
1. Two-NN Non-linear Intrinsic Dimension Estimator (Facco et al., Nature Scientific Reports 2017):
   Measures the true local topology dimension of the latent manifold without any linear projection.
2. Power-Law Spectral Decay Index alpha (lambda_i ~ i^-alpha):
   Distinguishes true dimensional collapse (alpha -> inf or sharp cliff) from scale-free continuous representation.
3. Region-Partitioned SVD (Sound Metal vs. Defect Region):
   Compares the eigenspectra separately, proving mathematically whether PC1 dominance belongs to the physical specimen baseline or the model.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.stats import linregress


def estimate_twonn_dimension(X: np.ndarray, subsample: int = 5000, random_state: int = 42) -> float:
    """
    Estimates the non-linear intrinsic dimension of data manifold X using the Two-NN algorithm
    (Facco, Elena, et al. 'Estimating the intrinsic dimension of datasets by a minimal neighborhood information.'
    Scientific Reports 7.1 (2017): 12140).

    Args:
        X: [N, D] array of latent representations
        subsample: maximum number of points to use for fast estimation
    Returns:
        d_twonn: estimated intrinsic dimension (float)
    """
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    if N > subsample:
        np.random.seed(random_state)
        idx = np.random.choice(N, subsample, replace=False)
        X = X[idx]
        N = len(X)

    # Find distances to 2 nearest neighbors (k=3 includes self at index 0)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Remove zero distance duplicates
    valid = (r1 > 1e-10) & (r2 > r1)
    if np.sum(valid) < 50:
        return 1.0

    mu = r2[valid] / r1[valid]
    mu = np.sort(mu)
    N_valid = len(mu)

    # Empirical cumulative distribution F(mu) = i / N_valid
    # Theory: 1 - F(mu) = mu^(-d) => log(1 - F(mu)) = -d * log(mu)
    f_emp = np.arange(1, N_valid + 1) / (N_valid + 1)
    y = -np.log(1.0 - f_emp)
    x = np.log(mu)

    # Linear fit passing through origin: y = d * x
    # Or standard linear regression on middle 80% to avoid edge bias
    lo_idx = int(0.1 * N_valid)
    hi_idx = int(0.9 * N_valid)
    slope, _, _, _, _ = linregress(x[lo_idx:hi_idx], y[lo_idx:hi_idx])

    return float(max(1.0, slope))


def compute_eigenspectrum_decay(X: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Computes singular value spectrum and power-law decay exponent alpha (lambda_i ~ i^-alpha).
    - alpha ~ 1.0: healthy, scale-free representation (like biological neural codes)
    - alpha >> 2.0: fast spectral decay (few dominant components)
    - ERank: exponential entropy of normalized eigenvalues
    """
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    eigvals = (s ** 2) / (len(X) - 1 + 1e-8)
    p = eigvals / (np.sum(eigvals) + 1e-12)

    # Effective Rank = exp( - sum p_i log p_i )
    entropy = -np.sum(p * np.log(p + 1e-12))
    erank = float(np.exp(entropy))

    # Power law fit: log(lambda_i) = -alpha * log(i) + c
    valid_k = np.where(eigvals > 1e-8)[0]
    if len(valid_k) >= 5:
        ranks = np.arange(1, len(valid_k) + 1)
        res = linregress(np.log(ranks), np.log(eigvals[valid_k]))
        alpha = float(-res.slope)
    else:
        alpha = 0.0

    return eigvals, erank, alpha


def compare_sound_vs_defect_representation(
    feature_map: np.ndarray,  # [sY, sX, D]
    gt_mask: np.ndarray       # [sY, sX] (1: defect, 0: sound metal, -1: buffer)
) -> Dict[str, Any]:
    """
    Directly answers the advisor's question by comparing the eigenspectrum and intrinsic dimension
    of Sound Metal versus Defect regions.
    """
    sY, sX, D = feature_map.shape
    flat_feats = feature_map.reshape(-1, D)
    flat_labels = gt_mask.reshape(-1)

    defect_idx = np.where(flat_labels == 1)[0]
    sound_idx = np.where(flat_labels == 0)[0]

    X_all = flat_feats[flat_labels >= 0]
    X_sound = flat_feats[sound_idx]
    X_defect = flat_feats[defect_idx]

    # Global
    _, erank_global, alpha_global = compute_eigenspectrum_decay(X_all)
    d_twonn_global = estimate_twonn_dimension(X_all)

    # Sound Metal
    e_sound, erank_sound, alpha_sound = compute_eigenspectrum_decay(X_sound)
    d_twonn_sound = estimate_twonn_dimension(X_sound)

    # Defect Region
    e_defect, erank_defect, alpha_defect = compute_eigenspectrum_decay(X_defect)
    d_twonn_defect = estimate_twonn_dimension(X_defect)

    # Percentage of variance in PC1
    pc1_pct_all = float(e_sound[0] / (np.sum(e_sound) + 1e-8) * 100) if len(e_sound) > 0 else 0.0
    pc1_pct_defect = float(e_defect[0] / (np.sum(e_defect) + 1e-8) * 100) if len(e_defect) > 0 else 0.0

    return {
        "global": {
            "n_samples": len(X_all),
            "effective_rank": round(erank_global, 2),
            "twonn_intrinsic_dimension": round(d_twonn_global, 2),
            "alpha_decay": round(alpha_global, 2)
        },
        "sound_metal": {
            "n_samples": len(X_sound),
            "effective_rank": round(erank_sound, 2),
            "twonn_intrinsic_dimension": round(d_twonn_sound, 2),
            "alpha_decay": round(alpha_sound, 2),
            "pc1_variance_pct": round(pc1_pct_all, 2)
        },
        "defect_regions": {
            "n_samples": len(X_defect),
            "effective_rank": round(erank_defect, 2),
            "twonn_intrinsic_dimension": round(d_twonn_defect, 2),
            "alpha_decay": round(alpha_defect, 2),
            "pc1_variance_pct": round(pc1_pct_defect, 2)
        }
    }
