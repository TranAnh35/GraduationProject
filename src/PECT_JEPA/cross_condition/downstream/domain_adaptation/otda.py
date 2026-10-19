"""
Optimal Transport Domain Adaptation (OTDA) via Entropic Sinkhorn Algorithm.

Non-alignment / metric-preserving domain adaptation:
Instead of training an adversarial discriminator (DANN) or matching second-order moments (CORAL),
OTDA directly minimizes the Kantorovich Wasserstein distance W_2^2(P_S, P_T) using fast
vectorized Sinkhorn-Knopp iterations in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def sinkhorn_knopp(cost_matrix: torch.Tensor, epsilon: float = 0.1, max_iter: int = 50) -> torch.Tensor:
    """
    Computes entropic regularized optimal transport plan via Sinkhorn-Knopp iterations.
    Args:
        cost_matrix: [B_s, B_t] ground metric cost matrix (e.g. squared Euclidean distance)
        epsilon: Entropic regularization parameter
        max_iter: Number of Sinkhorn scaling iterations
    Returns:
        gamma: [B_s, B_t] optimal transport plan where sum(gamma) = 1.0
    """
    B_s, B_t = cost_matrix.shape
    device = cost_matrix.device

    # Scaled Gibbs kernel
    K = torch.exp(-cost_matrix / max(epsilon, 1e-4))
    a = torch.ones(B_s, device=device) / B_s
    b = torch.ones(B_t, device=device) / B_t

    u = a.clone()
    for _ in range(max_iter):
        v = b / (torch.matmul(K.t(), u) + 1e-8)
        u = a / (torch.matmul(K, v) + 1e-8)

    gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
    return gamma


def wasserstein_distance(x_s: torch.Tensor, x_t: torch.Tensor, epsilon: float = 0.1, max_iter: int = 40) -> torch.Tensor:
    """
    Computes regularized 2-Wasserstein distance between source and target batches.
    """
    D = x_s.shape[-1]
    # Cost matrix: normalized squared Euclidean distance
    cost = torch.cdist(x_s, x_t, p=2.0) ** 2 / float(D)
    gamma = sinkhorn_knopp(cost, epsilon=epsilon, max_iter=max_iter)
    # Detach gamma to provide clean linear transport gradients on representations
    ot_loss = torch.sum(gamma.detach() * cost)
    return ot_loss


class OTDAModel(nn.Module):
    """
    Optimal Transport Domain Adaptation Network:
    - Feature Adapter: Linear/MLP projection f(z) -> z_adapted
    - Task Classifier: Linear classifier g(z_adapted) -> logits
    - Loss: Task Loss (Source BCE) + lambda_ot * Wasserstein(f(Z_s), f(Z_t))
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_net(x)
        logits = self.classifier(feat)
        return logits.squeeze(-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
        self.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                bx = torch.from_numpy(x[i:i+batch_size]).float().to(device)
                logits = self(bx)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs, axis=0) if probs else np.array([])
