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


def compute_anomaly_metrics(
    score_map: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    top_percentile: float = 90.0,
) -> Dict[str, Any]:
    """
    Computes quantitative defect detection metrics on the 2D anomaly score map.

    If gt_mask is provided:
        Computes TRUE label-based metrics:
        - True contrast_ratio_cnr: (defect_mean - sound_mean) / (sound_std + 1e-8)
        - True peak_contrast_ratio: (defect_max - sound_mean) / (sound_std + 1e-8)
        - auc_roc: Area Under ROC Curve
        - average_precision: Area Under Precision-Recall Curve (PR-AUC)
        - best_f1: Optimal F1-Score
        - has_ground_truth: True

    If gt_mask is None:
        Falls back to unsupervised top_percentile heuristic (default 90.0).
    """
    flat = score_map.flatten().astype(np.float64)
    max_score = float(np.max(flat)) if len(flat) > 0 else 0.0
    mean_score = float(np.mean(flat)) if len(flat) > 0 else 0.0

    if gt_mask is not None:
        min_Y = min(score_map.shape[0], gt_mask.shape[0])
        min_X = min(score_map.shape[1], gt_mask.shape[1])
        s_aligned = score_map[:min_Y, :min_X]
        g_aligned = gt_mask[:min_Y, :min_X]

        flat_s = s_aligned.reshape(-1)
        flat_g = g_aligned.reshape(-1)

        defect_pts = flat_s[flat_g == 1]
        sound_pts = flat_s[flat_g == 0]

        if len(defect_pts) > 0 and len(sound_pts) > 0:
            bg_mean = float(np.mean(sound_pts))
            bg_std = float(np.std(sound_pts))
            defect_mean = float(np.mean(defect_pts))
            defect_max = float(np.max(defect_pts))

            cnr = (defect_mean - bg_mean) / (bg_std + 1e-8)
            p_cnr = (defect_max - bg_mean) / (bg_std + 1e-8)

            gt_eval = evaluate_anomaly_ground_truth(s_aligned, g_aligned)

            return {
                "mean_score": mean_score,
                "max_score": max_score,
                "background_mean": bg_mean,
                "background_std": bg_std,
                "defect_mean": defect_mean,
                "defect_max": defect_max,
                "contrast_ratio_cnr": float(cnr),
                "peak_contrast_ratio": float(p_cnr),
                "auc_roc": gt_eval["auc_roc"],
                "average_precision": gt_eval["average_precision"],
                "best_f1": gt_eval["best_f1"],
                "optimal_threshold": gt_eval["optimal_threshold"],
                "has_ground_truth": True,
            }

    # Fallback when gt_mask is None or has no valid defect/sound split
    threshold = np.percentile(flat, top_percentile)
    defect_pts = flat[flat >= threshold]
    bg_pts = flat[flat < threshold]

    bg_mean = float(np.mean(bg_pts)) if len(bg_pts) > 0 else 0.0
    bg_std = float(np.std(bg_pts)) if len(bg_pts) > 0 else 1.0
    defect_mean = float(np.mean(defect_pts)) if len(defect_pts) > 0 else 0.0

    cnr = (defect_mean - bg_mean) / (bg_std + 1e-8)
    p_cnr = (max_score - bg_mean) / (bg_std + 1e-8)

    return {
        "mean_score": mean_score,
        "max_score": max_score,
        "background_mean": bg_mean,
        "background_std": bg_std,
        "defect_mean": defect_mean,
        "defect_max": max_score,
        "contrast_ratio_cnr": float(cnr),
        "peak_contrast_ratio": float(p_cnr),
        "auc_roc": None,
        "average_precision": None,
        "best_f1": None,
        "optimal_threshold": None,
        "has_ground_truth": False,
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


def plot_latent_representation_quality(
    feature_map: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title_prefix: str = "Latent Representation Quality",
    close_fig: bool = True,
) -> Dict[str, Any]:
    """
    Directly diagnoses the geometry and physical contrast of PECT-JEPA representations
    WITHOUT artificial spatial high-pass filtering (Spatial Residual Contrast) or downstream detectors.

    Generates a 3-panel diagnostic visualization:
    1. Latent PCA-RGB Projection:
       - Projects the high-dimensional latent space z in R^D to the top 3 principal components (PC1=R, PC2=G, PC3=B).
       - Reveals intrinsic cluster separation, flaw pop-out, and lift-off gradients.
    2. Hypersphere Angular Distance Map (1 - cos(theta)):
       - Measures angular departure on the unit hypersphere from nominal sound metal baseline.
       - Pure representation distance without spatial detrending.
    3. Authoritative CAD Ground-Truth Verification Overlay:
       - Physical CAD defect and fastener contours registered to C-scan space.

    Returns:
        dict containing:
        - 'fig': matplotlib figure (if not closed)
        - 'pca_variance_explained': list of variance explained by [PC1, PC2, PC3]
        - 'total_3pc_variance': sum of variance of top 3 PCs
        - 'angular_cnr': contrast-to-noise ratio in hypersphere angular space
        - 'angular_auc': AUC-ROC of raw angular distance vs true label (if gt_mask provided)
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score

    sY, sX, D = feature_map.shape
    flat = feature_map.reshape(-1, D).astype(np.float32)

    # 1. PCA-RGB Projection
    pca = PCA(n_components=3, random_state=42)
    pca_proj = pca.fit_transform(flat)  # [N, 3]
    var_exp = pca.explained_variance_ratio_

    rgb_map = np.zeros((sY, sX, 3), dtype=np.float32)
    for c in range(3):
        pc_c = pca_proj[:, c].reshape(sY, sX)
        p_low = np.percentile(pc_c, 1.0)
        p_high = np.percentile(pc_c, 99.0)
        if p_high > p_low:
            pc_norm = np.clip((pc_c - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            pc_norm = np.zeros_like(pc_c)
        rgb_map[:, :, c] = pc_norm

    # 2. Hypersphere Angular Distance Map (1 - cos(theta))
    norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
    flat_norm = flat / norms  # [N, D] on unit sphere

    # Estimate nominal sound metal vector as spatial median on unit sphere
    median_vec = np.median(flat_norm, axis=0, keepdims=True)
    median_vec = median_vec / (np.linalg.norm(median_vec, axis=-1, keepdims=True) + 1e-8)

    # Cosine distance: 1 - z . z_nominal
    cos_sim = np.sum(flat_norm * median_vec, axis=-1)  # [N]
    angular_dist = (1.0 - cos_sim).reshape(sY, sX)

    # Calculate metrics on angular distance
    angular_cnr = float("nan")
    angular_auc = None
    angular_ap = None

    if gt_mask is not None:
        flat_labels = gt_mask.reshape(-1)
        valid = flat_labels >= 0
        if np.any(flat_labels == 1) and np.any(flat_labels == 0):
            y_t = flat_labels[valid].astype(np.int64)
            y_s = angular_dist.reshape(-1)[valid]
            try:
                angular_auc = float(roc_auc_score(y_t, y_s))
                angular_ap = float(average_precision_score(y_t, y_s))
            except Exception:
                pass

            def_vals = angular_dist.reshape(-1)[flat_labels == 1]
            snd_vals = angular_dist.reshape(-1)[flat_labels == 0]
            if len(def_vals) > 0 and len(snd_vals) > 0:
                s_std = float(np.std(snd_vals))
                angular_cnr = float((np.mean(def_vals) - np.mean(snd_vals)) / (s_std + 1e-8))

    # 3. Create Diagnostic Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

    # Panel 1: Latent PCA-RGB Projection
    axes[0].imshow(rgb_map, origin="lower", aspect="equal")
    axes[0].set_title(
        f"1. Latent PCA-RGB Projection\n"
        f"PC1(R):{var_exp[0]*100:.1f}% | PC2(G):{var_exp[1]*100:.1f}% | PC3(B):{var_exp[2]*100:.1f}% "
        f"(\u03a3={np.sum(var_exp)*100:.1f}%)",
        fontsize=10,
        fontweight="bold"
    )
    axes[0].set_xlabel("Scan X (pixels)")
    axes[0].set_ylabel("Scan Y (pixels)")

    # Panel 2: Hypersphere Angular Distance Map
    vmax_ang = float(np.percentile(angular_dist, 99.0))
    im1 = axes[1].imshow(angular_dist, cmap="magma", origin="lower", aspect="equal", vmin=0.0, vmax=max(vmax_ang, 1e-4))
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Angular Distance (1 - cos \u03b8)")
    title_p2 = "2. Hypersphere Angular Distance\n(Pure Latent Departure from Nominal)"
    if not np.isnan(angular_cnr):
        title_p2 += f" | Raw Latent CNR: {angular_cnr:.2f}"
    axes[1].set_title(title_p2, fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Scan X (pixels)")
    axes[1].set_ylabel("Scan Y (pixels)")

    # Panel 3: CAD Ground-Truth Verification Overlay
    if gt_mask is not None:
        im2 = axes[2].imshow(angular_dist, cmap="viridis", origin="lower", aspect="equal", vmin=0.0, vmax=max(vmax_ang, 1e-4))
        # Overlay binary defect contour
        binary_defects = (gt_mask == 1).astype(np.float32)
        if np.any(binary_defects > 0):
            axes[2].contour(binary_defects, levels=[0.5], colors=["#00ffff"], linewidths=[1.5])
        title_p3 = "3. CAD Ground-Truth Alignment\n"
        if angular_auc is not None:
            title_p3 += f"Raw Latent AUC: {angular_auc:.4f} | AP: {angular_ap:.4f}"
        else:
            title_p3 += "Cyan Contours = True Physical Flaws"
        axes[2].set_title(title_p3, fontsize=10, fontweight="bold")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Angular Distance (1 - cos \u03b8)")
    else:
        # Fallback if no gt_mask: plot intensity slice or magnitude
        axes[2].imshow(np.linalg.norm(feature_map, axis=-1), cmap="viridis", origin="lower", aspect="equal")
        axes[2].set_title("3. Latent L2 Norm Distribution", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Scan X (pixels)")
        axes[2].set_ylabel("Scan Y (pixels)")

    axes[2].set_xlabel("Scan X (pixels)")
    axes[2].set_ylabel("Scan Y (pixels)")

    plt.suptitle(f"{title_prefix} — Representation Geometry & Contrast Audit", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)

    result_dict = {
        "fig": fig if not close_fig else None,
        "pca_variance_explained": [float(v) for v in var_exp],
        "total_3pc_variance": float(np.sum(var_exp)),
        "angular_cnr": angular_cnr,
        "angular_auc": angular_auc,
        "angular_ap": angular_ap,
    }

    if close_fig:
        plt.close(fig)

    return result_dict


