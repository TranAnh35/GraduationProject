"""
Domain-Adversarial Neural Network (DANN) for Unsupervised Domain Adaptation on PECT Representations.

Reference:
- Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation."
  ICML 2015.
- Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks."
  JMLR 2016.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class ReverseGrad(Function):
    """
    Gradient Reversal Layer (GRL).
    In forward pass, it acts as an identity.
    In backward pass, it scales gradients by -lambda.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return ReverseGrad.apply(x, alpha)


class DANNModel(nn.Module):
    """
    DANN Architecture for PECT Latents:
    1. Feature Adapter: g_phi(z) -> z_adapted (Residual adaptation)
    2. Task Classifier: h_psi(z_adapted) -> Defect vs Sound Metal logits
    3. Domain Discriminator: D_omega(GRL(z_adapted)) -> Source vs Target domain logits
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim

        # 1. Feature Adapter
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        # 2. Defect Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # 3. Domain Discriminator
        self.discriminator = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """Adapts representations through residual projection."""
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns:
        - class_logits: [B] defect vs sound metal
        - domain_logits: [B] source vs target domain
        """
        z_adapted = self.adapt(x)
        class_logits = self.classifier(z_adapted).squeeze(-1)

        # Apply Gradient Reversal before Discriminator
        z_rev = grad_reverse(z_adapted, alpha=alpha)
        domain_logits = self.discriminator(z_rev).squeeze(-1)

        return class_logits, domain_logits

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
        """Inference helper for numpy arrays (computes defect probability)."""
        self.eval()
        self.to(device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.from_numpy(x[i : i + batch_size]).float().to(device)
                z_adapted = self.adapt(batch)
                logits = self.classifier(z_adapted).squeeze(-1)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0) if probs else np.array([])
