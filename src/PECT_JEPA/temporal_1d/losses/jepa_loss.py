"""
JEPA Loss for 1D Temporal PECT-JEPA (Stage A4): latent prediction + anti-collapse.

    L = L_JEPA (normalized L2 between H_pred and EMA H_target)
      + var_weight  * mean max(0, gamma - std_d(H_pred))   (variance hinge)
      + cov_weight  * mean_offdiag(|Cov_d(H_pred)|)        (covariance decorrelation)

The regularization is applied on H_pred (gradient flows through predictor +
context encoder). H_target is detached (EMA target encoder), so regularizing
it would provide no learning signal.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss1D(nn.Module):
    """
    Normalized L2 latent-prediction loss + VICReg-style anti-collapse terms.
    """

    def __init__(
        self,
        loss_type: str = "normalized_l2",
        eps: float = 1e-8,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        var_gamma: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma

    def latent_prediction_loss(
        self, H_pred: torch.Tensor, H_target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "normalized_l2":
            pred = F.normalize(H_pred, p=2, dim=-1, eps=self.eps)
            tgt = F.normalize(H_target, p=2, dim=-1, eps=self.eps)
            return torch.mean(torch.sum((pred - tgt) ** 2, dim=-1))
        elif self.loss_type == "cosine":
            pred = F.normalize(H_pred, p=2, dim=-1, eps=self.eps)
            tgt = F.normalize(H_target, p=2, dim=-1, eps=self.eps)
            return torch.mean(1.0 - torch.sum(pred * tgt, dim=-1))
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(H_pred, H_target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def variance_hinge(self, H_pred: torch.Tensor) -> torch.Tensor:
        """Mean over embedding dimensions D of max(0, gamma - std_batch)."""
        B, N, D = H_pred.shape
        z = H_pred.reshape(B * N, D)
        std = torch.sqrt(z.var(dim=0) + self.eps)  # [D]
        return torch.mean(F.relu(self.var_gamma - std))

    def covariance_penalty(self, H_pred: torch.Tensor) -> torch.Tensor:
        """Mean |off-diagonal covariance| over embedding dims (batch-averaged)."""
        B, N, D = H_pred.shape
        z = H_pred.reshape(B * N, D)
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(1, z.shape[0] - 1)  # [D, D]
        off = cov - torch.diag(torch.diag(cov))
        return off.abs().mean()

    def forward(self, H_pred: torch.Tensor, H_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_pred = self.latent_prediction_loss(H_pred, H_target)
        l_var = self.variance_hinge(H_pred)
        l_cov = self.covariance_penalty(H_pred)
        total = l_pred + self.var_weight * l_var + self.cov_weight * l_cov
        return {
            "loss": total,
            "loss_pred": l_pred.detach(),
            "loss_var": l_var.detach(),
            "loss_cov": l_cov.detach(),
        }
