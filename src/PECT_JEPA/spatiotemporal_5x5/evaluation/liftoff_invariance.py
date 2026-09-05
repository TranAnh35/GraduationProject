"""
Lift-Off Invariance and Cross-Domain Probing for 5x5 PECT-JEPA.
Computes Linear CKA (Centered Kernel Alignment) and Cosine Similarity
between latent representations extracted at different lift-offs (z1 vs z2 vs z3).
"""

import numpy as np


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between feature matrices X and Y.
    Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
    
    Args:
        X: [N, D1] representations at condition 1 (e.g. z1)
        Y: [N, D2] representations at condition 2 (e.g. z3)
    Returns:
        CKA score in [0, 1] (1.0 = identical representation geometry).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
    if Y.ndim > 2:
        Y = Y.reshape(-1, Y.shape[-1])

    assert X.shape[0] == Y.shape[0], f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"

    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC(K, L) for linear kernels = ||Y^T X||_F^2
    yt_x = Y.T @ X
    hsic_xy = np.sum(yt_x ** 2)

    xt_x = X.T @ X
    hsic_xx = np.sum(xt_x ** 2)

    yt_y = Y.T @ Y
    hsic_yy = np.sum(yt_y ** 2)

    denominator = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    return float(hsic_xy / denominator)


def compute_feature_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Mean point-wise cosine similarity between paired representations X and Y at identical coordinates.
    """
    X = np.asarray(X, dtype=np.float32).reshape(-1, X.shape[-1])
    Y = np.asarray(Y, dtype=np.float32).reshape(-1, Y.shape[-1])

    norm_x = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
    norm_y = np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-8

    X_u = X / norm_x
    Y_u = Y / norm_y

    cos_sim = np.sum(X_u * Y_u, axis=-1)
    return float(np.mean(cos_sim))


def compute_effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank (Roy & Vetterli): exp(Shannon entropy of
    normalized squared singular values).
    A collapsed embedding has rank ~1.0, while a healthy embedding has rank close to D (e.g. 50-128).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 2:
        X = X.reshape(-1, X.shape[-1])
    if X.shape[0] < 2:
        return 1.0
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = (s ** 2) / max(float((s ** 2).sum()), eps)
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))

