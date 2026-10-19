"""
Source Hypothesis Transfer (SHOT) / Information Maximization for PECT Inspection.

Reference:
- Liang, Jian, Dapeng Hu, and Jiashi Feng. "Do we really need to access the source data?
  Source hypothesis transfer for unsupervised domain adaptation."
  ICML 2020.
"""

import torch
import torch.nn as nn
import numpy as np


class SHOTModel(nn.Module):
    """
    SHOT Architecture for PECT Latents:
    1. Feature Adapter: g_phi(z) -> z_adapted
    2. Classifier: h_psi(z_adapted) -> Defect logits
    3. Target Adaptation Objective:
       L_target = L_entropy(p_t) + lambda_div * L_diversity(p_t)
       where L_entropy forces confident binary decision,
       and L_diversity enforces prior target class ratio (prevents sound-metal mode collapse).
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

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_adapted = self.adapt(x)
        return self.classifier(z_adapted).squeeze(-1)

    def compute_im_loss(self, z_target: torch.Tensor, defect_prior: float = 0.02) -> torch.Tensor:
        """
        Information Maximization loss:
        - Entropy loss: - [p log p + (1-p) log (1-p)] encourages confident predictions.
        - Prior Diversity loss: KL(mean(p_t) || prior) prevents predicting all 0.
        """
        logits_t = self.classifier(z_target).squeeze(-1)
        p = torch.sigmoid(logits_t)
        eps = 1e-6

        # 1. Entropy minimization
        entropy = - (p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))
        loss_ent = torch.mean(entropy)

        # 2. Diversity / Prior preservation
        p_mean = torch.mean(p)
        prior = torch.tensor(defect_prior, device=p.device)
        kl_div = p_mean * torch.log((p_mean + eps) / prior) + (1.0 - p_mean) * torch.log((1.0 - p_mean + eps) / (1.0 - prior))

        return loss_ent + 2.0 * kl_div

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
