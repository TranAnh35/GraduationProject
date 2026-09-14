"""
Unsupervised Anomaly Detection and C-Scan Heatmap Visualization for 5x5 PECT-JEPA.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, Any


class AnomalyDetector5x5:
    """
    Downstream anomaly detector on frozen 5x5 representations [sY, sX, D].
    Uses 2D spatial detrending (local baseline contrast) to eliminate low-frequency
    mechanical scanner lift-off tilt and environmental drift, isolating localized defects.
    Supports local contrast detrending, raw euclidean distance, and cosine distance.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: str = "euclidean",
        detrend: bool = True,
        detrend_sigma: float = 15.0,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.detrend = detrend
        self.detrend_sigma = detrend_sigma
        self.normal_prototype: Optional[np.ndarray] = None
        self.raw_prototype: Optional[np.ndarray] = None

    def fit(self, train_features: np.ndarray):
        """
        Fit sound metal baseline prototype from unlabelled representations.
        train_features: [N, D] or [sY, sX, D]
        """
        flat = train_features.reshape(-1, train_features.shape[-1]).astype(np.float32)

        if self.metric == "euclidean":
            # Direct MiniBatchKMeans clustering on raw representations preserves energy/magnitude
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=4096).fit(flat)
            counts = np.bincount(kmeans.labels_)
            dominant_id = int(np.argmax(counts))

            self.raw_prototype = kmeans.cluster_centers_[dominant_id]
            norm_val = np.linalg.norm(self.raw_prototype) + 1e-8
            self.normal_prototype = self.raw_prototype / norm_val
        else:
            # Cosine distance clustering on unit hypersphere
            norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
            flat_norm = flat / norms

            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=4096).fit(flat_norm)
            counts = np.bincount(kmeans.labels_)
            dominant_id = int(np.argmax(counts))

            proto = kmeans.cluster_centers_[dominant_id]
            self.normal_prototype = proto / (np.linalg.norm(proto) + 1e-8)
            mask_dominant = kmeans.labels_ == dominant_id
            if np.any(mask_dominant):
                self.raw_prototype = np.mean(flat[mask_dominant], axis=0)
            else:
                self.raw_prototype = proto

    def score_map(self, test_map: np.ndarray, detrend: Optional[bool] = None) -> np.ndarray:
        """
        Compute 2D anomaly score map for test_map [sY, sX, D].
        Returns: [sY, sX] float32 array where higher score = more anomalous.
        If detrend is True (or None with self.detrend=True), applies local baseline detrending.
        If detrend is False, evaluates pure raw latent distance from sound prototype.
        """
        do_detrend = self.detrend if detrend is None else bool(detrend)
        if test_map.ndim == 3 and do_detrend:
            sY, sX, D = test_map.shape
            if min(sY, sX) >= 5:
                # 2D Spatial Detrending: filter out low-frequency mechanical scanner tilt / lift-off drift
                sigma = float(min(self.detrend_sigma, max(2.0, min(sY, sX) / 3.0)))
                fmap_f32 = test_map.astype(np.float32)
                fmap_bg = gaussian_filter(fmap_f32, sigma=(sigma, sigma, 0), mode="nearest")
                # Feature-level local contrast: residual vector norm against local sound baseline
                diff = fmap_f32 - fmap_bg
                return np.linalg.norm(diff, axis=-1).astype(np.float32)

        assert (self.raw_prototype is not None if self.metric == "euclidean" else self.normal_prototype is not None), "Detector must be fitted first"
        sY, sX, D = test_map.shape
        flat = test_map.reshape(-1, D).astype(np.float32)

        if self.metric == "euclidean" and self.raw_prototype is not None:
            # Euclidean distance from sound metal baseline in latent space
            diff = flat - self.raw_prototype
            scores_1d = np.linalg.norm(diff, axis=-1)
        else:
            # Cosine distance: 1 - cos(theta)
            norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
            flat_norm = flat / norms
            cos_sim = np.dot(flat_norm, self.normal_prototype)
            scores_1d = 1.0 - cos_sim

        return scores_1d.reshape(sY, sX).astype(np.float32)


def compute_anomaly_metrics(score_map: np.ndarray, top_percentile: float = 90.0) -> Dict[str, float]:
    """
    Computes quantitative defect detection metrics on the 2D anomaly score map.
    Default top_percentile=90.0 (top 10%) covers the multi-defect calibration matrix (~25 corrosion spots).
    - contrast_ratio_cnr: (defect_mean - background_mean) / (background_std + 1e-8)
    - peak_contrast_ratio: (max_score - background_mean) / (background_std + 1e-8)
    """
    flat = score_map.flatten().astype(np.float64)
    threshold = np.percentile(flat, top_percentile)

    defect_pts = flat[flat >= threshold]
    bg_pts = flat[flat < threshold]

    bg_mean = float(np.mean(bg_pts))
    bg_std = float(np.std(bg_pts))
    defect_mean = float(np.mean(defect_pts))
    max_score = float(np.max(flat))

    cnr = (defect_mean - bg_mean) / (bg_std + 1e-8)
    p_cnr = (max_score - bg_mean) / (bg_std + 1e-8)

    return {
        "mean_score": float(np.mean(flat)),
        "max_score": max_score,
        "background_mean": bg_mean,
        "background_std": bg_std,
        "defect_mean": defect_mean,
        "contrast_ratio_cnr": float(cnr),
        "peak_contrast_ratio": float(p_cnr),
    }


def plot_anomaly_heatmap_5x5(
    anomaly_map: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "5x5 PECT-JEPA C-Scan Anomaly Map",
    close_fig: bool = True,
    clip_percentile: float = 99.0,
):
    """
    Save and/or return 2D C-Scan anomaly heatmap matplotlib figure.
    Uses 99th percentile clipping to prevent extreme edge/outlier artifacts from dominating the colormap.
    """
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig = plt.figure(figsize=(7, 6), dpi=150)
    vmin = float(np.min(anomaly_map))
    vmax = float(np.percentile(anomaly_map, clip_percentile))
    if vmax <= vmin:
        vmax = float(np.max(anomaly_map))
    im = plt.imshow(anomaly_map, cmap="jet", aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Anomaly Score (Local Baseline Contrast)")
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Scan X (pixels)")
    plt.ylabel("Scan Y (pixels)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if close_fig:
        plt.close(fig)
        return None
    return fig


# ==============================================================================
# Advanced Anomaly Detectors & Ground-Truth Benchmarking
# ==============================================================================

class MahalanobisDetector:
    """
    Mahalanobis Distance anomaly detector on frozen latent representations.
    Fits empirical mean and covariance on sound metal baseline (dominant cluster or normal samples).
    Score = sqrt( (z - mu)^T (Sigma + eps*I)^-1 (z - mu) ).
    """
    def __init__(self, regularize_eps: float = 1e-4):
        self.regularize_eps = regularize_eps
        self.mean: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, contamination: float = 0.05):
        flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
        # Use median / robust estimation or trimmed sound metal
        center = np.median(flat, axis=0)
        dists = np.linalg.norm(flat - center, axis=1)
        inlier_mask = dists <= np.percentile(dists, 100.0 * (1.0 - contamination))
        sound_pts = flat[inlier_mask]

        self.mean = np.mean(sound_pts, axis=0)
        cov = np.cov(sound_pts, rowvar=False)
        cov_reg = cov + np.eye(cov.shape[0]) * self.regularize_eps
        self.precision = np.linalg.inv(cov_reg)

    def score_map(self, test_map: np.ndarray) -> np.ndarray:
        sY, sX, D = test_map.shape
        flat = test_map.reshape(-1, D).astype(np.float64)
        diff = flat - self.mean
        # d^2 = sum(diff * (diff @ precision), axis=-1)
        maha_sq = np.sum((diff @ self.precision) * diff, axis=-1)
        maha_d = np.sqrt(np.maximum(0, maha_sq))
        return maha_d.reshape(sY, sX).astype(np.float32)


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector on frozen latent representations.
    Optimized for heavily imbalanced anomaly detection (< 2% defect pixels).
    """
    def __init__(self, n_estimators: int = 100, max_samples: int = 2048, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = None

    def fit(self, features: np.ndarray):
        from sklearn.ensemble import IsolationForest
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
        # Downsample if too large for fast fitting
        if len(flat) > 20000:
            np.random.seed(self.random_state)
            idx = np.random.choice(len(flat), 20000, replace=False)
            flat_fit = flat[idx]
        else:
            flat_fit = flat

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(flat_fit)),
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(flat_fit)

    def score_map(self, test_map: np.ndarray) -> np.ndarray:
        sY, sX, D = test_map.shape
        flat = test_map.reshape(-1, D).astype(np.float32)
        # IsolationForest decision_function: lower means more anomalous.
        # Negate so higher means more anomalous.
        scores = -self.model.decision_function(flat)
        return scores.reshape(sY, sX).astype(np.float32)


class OneClassSVMDetector:
    """
    One-Class SVM anomaly detector on unit-normalized latent representations.
    """
    def __init__(self, kernel: str = "rbf", gamma: str = "scale", nu: float = 0.05):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.model = None

    def fit(self, features: np.ndarray):
        from sklearn.svm import OneClassSVM
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
        flat_norm = flat / norms

        if len(flat_norm) > 10000:
            np.random.seed(42)
            idx = np.random.choice(len(flat_norm), 10000, replace=False)
            flat_fit = flat_norm[idx]
        else:
            flat_fit = flat_norm

        self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
        self.model.fit(flat_fit)

    def score_map(self, test_map: np.ndarray) -> np.ndarray:
        sY, sX, D = test_map.shape
        flat = test_map.reshape(-1, D).astype(np.float32)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
        flat_norm = flat / norms
        scores = -self.model.decision_function(flat_norm)
        return scores.reshape(sY, sX).astype(np.float32)


def evaluate_anomaly_ground_truth(
    score_map: np.ndarray,
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates an anomaly score map against the binary ground-truth mask.
    Excludes transition buffer pixels (gt_mask == -1).
    Computes:
    - AUC-ROC
    - Average Precision (PR-AUC)
    - Optimal F1-Score, Precision, and Recall at best threshold
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

    flat_scores = score_map.reshape(-1)
    flat_labels = gt_mask.reshape(-1)

    valid_idx = np.where(flat_labels >= 0)[0]
    y_true = flat_labels[valid_idx].astype(np.int64)
    y_scores = flat_scores[valid_idx]

    auc_roc = float(roc_auc_score(y_true, y_scores))
    avg_prec = float(average_precision_score(y_true, y_scores))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2.0 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))

    best_f1 = float(f1_scores[best_idx])
    best_p = float(precisions[best_idx])
    best_r = float(recalls[best_idx])
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])

    return {
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(avg_prec, 4),
        "best_f1": round(best_f1, 4),
        "precision_at_best_f1": round(best_p, 4),
        "recall_at_best_f1": round(best_r, 4),
        "optimal_threshold": round(best_thresh, 4),
        "num_defect_pixels": int(np.sum(y_true == 1)),
        "num_sound_pixels": int(np.sum(y_true == 0)),
    }

