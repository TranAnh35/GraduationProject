"""
Anomaly Detection Evaluation for PECT-JEPA (Section 18.1 & 19.1).
Evaluates downstream representation quality using frozen encoder features.
Preserves full spatio-temporal representations [H_t, W_t, K_t, D] and computes
anomaly maps via distance to reference normal representations (kNN / Prototype distance).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import roc_auc_score, average_precision_score


class RepresentationAnomalyDetector:
    """
    Downstream anomaly detector operating on frozen PECT-JEPA representations.
    Maintains full spatio-temporal token representation without forced loss of temporal axis.
    """
    def __init__(
        self,
        method: str = "prototype",  # 'prototype', 'knn', or 'cosine'
        k_neighbors: int = 5
    ):
        self.method = method
        self.k_neighbors = k_neighbors
        self.normal_prototypes: Optional[torch.Tensor] = None

    def fit(self, normal_features: torch.Tensor):
        """
        Fit baseline normal distribution from normal/reference acquisition features.

        Args:
            normal_features: Tensor of shape [N_samples, H_t, W_t, K_t, D] or [N_tokens, D]
        """
        if normal_features.ndim > 2:
            # Flatten spatial/temporal positions to token vectors [N_tokens, D]
            normal_features = normal_features.reshape(-1, normal_features.shape[-1])

        # Normalize features along embedding dimension D
        normal_features = F.normalize(normal_features.float(), p=2, dim=-1)

        if self.method == "prototype":
            # Compute mean normal prototype vector
            self.normal_prototypes = torch.mean(normal_features, dim=0, keepdim=True)
            self.normal_prototypes = F.normalize(self.normal_prototypes, p=2, dim=-1)
        else:
            self.normal_prototypes = normal_features

    def score(self, test_features: torch.Tensor) -> np.ndarray:
        """
        Compute spatial anomaly map for test acquisition.

        Args:
            test_features: [H_t, W_t, K_t, D] or [H_t, W_t, D]

        Returns:
            anomaly_map: 2D numpy array [H_t, W_t] of anomaly scores
        """
        test_features = test_features.float()
        if test_features.ndim == 4:
            H_t, W_t, K_t, D = test_features.shape
            # Compute distances per temporal clip and aggregate across time
            flat_feats = test_features.view(H_t * W_t * K_t, D)
            flat_feats = F.normalize(flat_feats, p=2, dim=-1)

            if self.method == "prototype":
                cos_sim = torch.matmul(flat_feats, self.normal_prototypes.t()).squeeze(-1)
                token_scores = 1.0 - cos_sim
            elif self.method == "knn":
                sim_matrix = torch.matmul(flat_feats, self.normal_prototypes.t())
                topk_sim, _ = torch.topk(sim_matrix, k=min(self.k_neighbors, sim_matrix.shape[1]), dim=-1)
                token_scores = 1.0 - torch.mean(topk_sim, dim=-1)
            else:
                dist = torch.cdist(flat_feats, self.normal_prototypes)
                token_scores = torch.min(dist, dim=-1)[0]

            # Reshape to [H_t, W_t, K_t] and take maximum or mean anomaly score across clips
            grid_scores = token_scores.view(H_t, W_t, K_t)
            spatial_anomaly_map = torch.max(grid_scores, dim=-1)[0]  # [H_t, W_t]

        else:
            H_t, W_t, D = test_features.shape
            flat_feats = test_features.view(H_t * W_t, D)
            flat_feats = F.normalize(flat_feats, p=2, dim=-1)

            if self.method == "prototype":
                cos_sim = torch.matmul(flat_feats, self.normal_prototypes.t()).squeeze(-1)
                spatial_anomaly_map = (1.0 - cos_sim).view(H_t, W_t)
            elif self.method == "knn":
                sim_matrix = torch.matmul(flat_feats, self.normal_prototypes.t())
                topk_sim, _ = torch.topk(sim_matrix, k=min(self.k_neighbors, sim_matrix.shape[1]), dim=-1)
                spatial_anomaly_map = (1.0 - torch.mean(topk_sim, dim=-1)).view(H_t, W_t)
            else:
                dist = torch.cdist(flat_feats, self.normal_prototypes)
                spatial_anomaly_map = torch.min(dist, dim=-1)[0].view(H_t, W_t)

        return spatial_anomaly_map.detach().cpu().numpy()


def compute_anomaly_metrics(
    anomaly_scores: np.ndarray,
    ground_truth_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute quantitative anomaly metrics (ROC-AUC, PR-AUC).

    Args:
        anomaly_scores: 1D array or 2D map of anomaly predictions
        ground_truth_mask: Binary ground-truth (1 = anomaly / defect, 0 = normal)

    Returns:
        Dict with 'roc_auc' and 'pr_auc'
    """
    y_true = ground_truth_mask.ravel().astype(int)
    y_score = anomaly_scores.ravel().astype(float)

    if len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.5, "pr_auc": 0.0}

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }
