"""
Anomaly detection and representation metrics on 1D waveforms.
100% self-contained for 1D Temporal PECT-JEPA.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score


class WaveformAnomalyDetector:
    """
    Downstream anomaly detector operating on frozen 1D representations [B, D].
    Supports Unsupervised Latent Clustering to automatically isolate healthy material
    without requiring clean defect-free labels.
    """
    def __init__(
        self,
        method: str = "clustering",  # 'clustering', 'median', 'prototype', 'knn'
        n_clusters: int = 2,
        k_neighbors: int = 5
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.normal_prototypes: Optional[torch.Tensor] = None

    def fit(self, normal_features: torch.Tensor):
        """
        Fit baseline normal distribution from unlabelled/in-the-wild 1D waveforms.
        Uses unsupervised clustering to automatically find the dominant healthy cluster.

        Args:
            normal_features: [N_samples, D]
        """
        if normal_features.ndim > 2:
            normal_features = normal_features.reshape(-1, normal_features.shape[-1])

        normal_features = F.normalize(normal_features.float(), p=2, dim=-1)

        if self.method == "clustering":
            # 1. Unsupervised Latent Clustering (k-Means)
            from sklearn.cluster import MiniBatchKMeans
            feats_np = normal_features.detach().cpu().numpy()
            kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=4096).fit(feats_np)
            
            # Identify dominant cluster (homogeneous healthy metal)
            labels = kmeans.labels_
            counts = np.bincount(labels)
            dominant_cluster_id = int(np.argmax(counts))
            
            dominant_center = torch.from_numpy(kmeans.cluster_centers_[dominant_cluster_id]).float().unsqueeze(0)
            self.normal_prototypes = F.normalize(dominant_center, p=2, dim=-1).to(normal_features.device)
            
        elif self.method == "median":
            # 2. Robust Coordinate-wise Median
            med = torch.median(normal_features, dim=0, keepdim=True)[0]
            self.normal_prototypes = F.normalize(med, p=2, dim=-1)
            
        elif self.method == "prototype":
            # 3. Simple Mean Prototype
            self.normal_prototypes = torch.mean(normal_features, dim=0, keepdim=True)
            self.normal_prototypes = F.normalize(self.normal_prototypes, p=2, dim=-1)
        else:
            self.normal_prototypes = normal_features

    def score(self, test_features: torch.Tensor) -> np.ndarray:
        """
        Score 1D waveform features against the normal prototype.

        Args:
            test_features: [N_test, D]

        Returns:
            scores: 1D numpy array of anomaly scores
        """
        test_features = test_features.float()
        if test_features.ndim == 1:
            test_features = test_features.unsqueeze(0)

        flat = test_features.view(-1, test_features.shape[-1])
        flat = F.normalize(flat, p=2, dim=-1)

        proto = self.normal_prototypes.to(flat.device)

        if self.method == "prototype":
            cos_sim = torch.matmul(flat, proto.t()).squeeze(-1)
            scores = 1.0 - cos_sim
        elif self.method == "knn":
            sim_matrix = torch.matmul(flat, proto.t())
            topk_sim, _ = torch.topk(sim_matrix, k=min(self.k_neighbors, sim_matrix.shape[1]), dim=-1)
            scores = 1.0 - torch.mean(topk_sim, dim=-1)
        else:
            dist = torch.cdist(flat, proto)
            scores = torch.min(dist, dim=-1)[0]

        return scores.detach().cpu().numpy()


def compute_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC."""
    y_true = y_true.ravel().astype(int)
    y_score = y_score.ravel().astype(float)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.5, "pr_auc": 0.0}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score))
    }
