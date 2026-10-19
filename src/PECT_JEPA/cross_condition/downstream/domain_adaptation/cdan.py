"""
Conditional Domain Adversarial Network (CDAN).

Reference:
- Long, Mingsheng, et al. "Conditional adversarial domain adaptation."
  NeurIPS 2018.
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from .dann import ReverseGrad, grad_reverse


class CDANModel(nn.Module):
    """
    CDAN Architecture for PECT Latents:
    1. Feature Adapter: g_phi(z) -> z_adapted
    2. Defect Classifier: h_psi(z_adapted) -> Defect logits & probs
    3. Multilinear Conditioning: combines feature z_adapted with prediction softmax/prob
    4. Entropy-conditioned Domain Discriminator: D_omega(GRL(z_adapted (x) prob))
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

        # 3. Domain Discriminator conditioned on [feature, prob, 1-prob]
        cond_dim = in_dim + 2
        self.discriminator = nn.Sequential(
            nn.BatchNorm1d(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - class_logits: [B] defect vs sound metal
        - domain_logits: [B] source vs target domain
        - entropy_weights: [B] 1 + exp(-H(p)) certainty weighting
        """
        z_adapted = self.adapt(x)
        class_logits = self.classifier(z_adapted).squeeze(-1)
        prob = torch.sigmoid(class_logits).unsqueeze(-1)
        prob_vec = torch.cat([1.0 - prob, prob], dim=-1)  # [B, 2]

        # Multilinear concatenation conditioning
        cond_features = torch.cat([z_adapted, prob_vec], dim=-1)

        # Entropy calculation: H(p) = - sum(p * log(p))
        eps = 1e-6
        entropy = - (prob_vec * torch.log(prob_vec + eps)).sum(dim=-1)
        # Entropy weights: confident samples get higher weight in domain discriminator
        entropy_weights = 1.0 + torch.exp(-entropy)

        # Gradient reversal before discriminator
        cond_rev = grad_reverse(cond_features, alpha=alpha)
        domain_logits = self.discriminator(cond_rev).squeeze(-1)

        return class_logits, domain_logits, entropy_weights

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
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
