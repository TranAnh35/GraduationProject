"""
Deep Adaptation Network (DAN) with Multi-Kernel Maximum Mean Discrepancy (MK-MMD).

Reference:
- Long, Mingsheng, et al. "Learning transferable features with deep adaptation networks."
  ICML 2015.
"""

from typing import List
import torch
import torch.nn as nn
import numpy as np


class MultiKernelMMD(nn.Module):
    """
    Computes Multi-Kernel Maximum Mean Discrepancy between source and target representations
    using a mixture of multiple Gaussian RBF kernels with bandwidth multipliers.
    """
    def __init__(self, kernel_muls: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]):
        super().__init__()
        self.kernel_muls = kernel_muls

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # Pairwise L2 squared distance matrix
        dist_matrix = torch.cdist(total, total, p=2) ** 2

        # Median heuristic for base bandwidth
        median_dist = torch.median(dist_matrix)
        bandwidth = median_dist / np.log(n_samples) if median_dist > 0 else torch.tensor(1.0, device=source.device)

        # Multi-kernel sum
        kernel_val = 0.0
        for mul in self.kernel_muls:
            gamma = 1.0 / (2.0 * bandwidth * mul + 1e-8)
            kernel_val = kernel_val + torch.exp(-gamma * dist_matrix)
        return kernel_val

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_s = source.size(0)
        n_t = target.size(0)
        if n_s == 0 or n_t == 0:
            return torch.tensor(0.0, device=source.device)

        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:n_s, :n_s]
        YY = kernels[n_s:, n_s:]
        XY = kernels[:n_s, n_s:]

        mmd = torch.mean(XX) + torch.mean(YY) - 2.0 * torch.mean(XY)
        return torch.clamp(mmd, min=0.0)


class DANModel(nn.Module):
    """
    DAN Architecture:
    - Feature Adapter: g_phi(z) = z + MLP(LayerNorm(z))
    - Task Classifier: h_psi(z_adapted) -> Defect logits
    - Loss: Task Loss (source) + lambda * MK-MMD(z_s, z_t)
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
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 1),
        )
        self.mmd = MultiKernelMMD()

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_adapted = self.adapt(x)
        return self.classifier(z_adapted).squeeze(-1)

    def compute_mmd_loss(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        return self.mmd(z_s, z_t)

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
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
