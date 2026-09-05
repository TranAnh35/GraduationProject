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
from typing import Optional, Dict, Any


class AnomalyDetector5x5:
    """
    Downstream anomaly detector on frozen 5x5 representations [sY, sX, D].
    Uses unsupervised clustering to automatically isolate the dominant sound metal cluster.
    """

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.normal_prototype: Optional[np.ndarray] = None

    def fit(self, train_features: np.ndarray):
        """
        Fit sound metal baseline prototype from unlabelled representations.
        train_features: [N, D] or [sY, sX, D]
        """
        flat = train_features.reshape(-1, train_features.shape[-1]).astype(np.float32)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
        flat_norm = flat / norms

        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=4096).fit(flat_norm)
        counts = np.bincount(kmeans.labels_)
        dominant_id = int(np.argmax(counts))

        proto = kmeans.cluster_centers_[dominant_id]
        self.normal_prototype = proto / (np.linalg.norm(proto) + 1e-8)

    def score_map(self, test_map: np.ndarray) -> np.ndarray:
        """
        Compute 2D anomaly score map for test_map [sY, sX, D].
        Returns: [sY, sX] float32 array where higher score = more anomalous.
        """
        assert self.normal_prototype is not None, "Detector must be fitted first"
        sY, sX, D = test_map.shape
        flat = test_map.reshape(-1, D).astype(np.float32)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8
        flat_norm = flat / norms

        # Cosine distance: 1 - cos(theta)
        cos_sim = np.dot(flat_norm, self.normal_prototype)
        scores_1d = 1.0 - cos_sim
        return scores_1d.reshape(sY, sX).astype(np.float32)


def compute_anomaly_metrics(score_map: np.ndarray, top_percentile: float = 95.0) -> Dict[str, float]:
    """
    Computes quantitative defect detection metrics on the 2D anomaly score map:
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
    close_fig: bool = True
):
    """
    Save and/or return 2D C-Scan anomaly heatmap matplotlib figure.
    """
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig = plt.figure(figsize=(7, 6), dpi=150)
    im = plt.imshow(anomaly_map, cmap="jet", aspect="equal", origin="lower")
    plt.colorbar(im, label="Anomaly Score (Cosine Distance to Sound Baseline)")
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
