"""
Correlation Alignment (CORAL) for Unsupervised Domain Adaptation on PECT Representations.

Reference:
- Sun, Baochen, Jiashi Feng, and Kate Saenko. "Return of frustratingly easy domain adaptation."
  AAAI 2016.
- Sun, Baochen, and Kate Saenko. "Deep CORAL: Correlation alignment for deep domain adaptation."
  ECCV Workshops 2016.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_covariance(features: torch.Tensor) -> torch.Tensor:
    """Computes the sample covariance matrix of a feature batch [N, D]."""
    n = features.size(0)
    if n <= 1:
        d = features.size(1)
        return torch.zeros((d, d), device=features.device, dtype=features.dtype)
    mean = torch.mean(features, dim=0, keepdim=True)
    centered = features - mean
    cov = torch.matmul(centered.t(), centered) / (n - 1)
    return cov


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes CORAL distance between source and target feature covariance matrices:
    L_coral = (1 / (4 * d^2)) * ||Cov_s - Cov_t||_F^2
    """
    d = source.size(1)
    cov_s = compute_covariance(source)
    cov_t = compute_covariance(target)
    loss = torch.norm(cov_s - cov_t, p="fro") ** 2
    return loss / (4.0 * (d ** 2))


class CORALAdapter(nn.Module):
    """
    Feature-level Domain Adapter with Classification Head trained with CORAL loss.

    Structure:
    - Feature Adapter: g_phi(z) = z + MLP(LayerNorm(z))  [Residual adaptation preserves foundation latents]
    - Classifier Head: h_psi(z_adapted) -> logit
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )
        # Small scale init so initially g_phi(z) approx z
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 1),
        )

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input representations into domain-aligned representations."""
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns classification logits for input representations."""
        z_adapted = self.adapt(x)
        return self.classifier(z_adapted).squeeze(-1)

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
        """Inference helper for numpy arrays."""
        self.eval()
        self.to(device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.from_numpy(x[i : i + batch_size]).float().to(device)
                logits = self.forward(batch)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0) if probs else np.array([])
