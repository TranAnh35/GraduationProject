"""
JEPA Loss for 5x5 PECT-JEPA: Smooth L1 Prediction Loss + VICReg Anti-Collapse.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss5x5(nn.Module):
    """
    Smooth L1 latent prediction loss + VICReg variance hinge & covariance penalty.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        eps: float = 1e-8,
        var_weight: float = 1.0,
        cov_weight: float = 0.5,
        var_gamma: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma

    def latent_prediction_loss(self, H_pred: torch.Tensor, H_target: torch.Tensor) -> torch.Tensor:
        safe_eps = max(self.eps, 1e-5)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(H_pred, H_target, beta=1.0)
        elif self.loss_type == "l1":
            return F.l1_loss(H_pred, H_target)
        elif self.loss_type == "l2":
            return F.mse_loss(H_pred, H_target)
        elif self.loss_type == "cosine":
            pred = F.normalize(H_pred, p=2, dim=-1, eps=safe_eps)
            tgt = F.normalize(H_target, p=2, dim=-1, eps=safe_eps)
            return torch.mean(1.0 - torch.sum(pred * tgt, dim=-1))
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def variance_hinge(self, H_pred: torch.Tensor) -> torch.Tensor:
        """Forces batch variance along each dimension to be >= var_gamma."""
        B, N, D = H_pred.shape
        z = H_pred.reshape(B * N, D)
        safe_eps = max(self.eps, 1e-5)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + safe_eps)  # [D]
        return torch.mean(F.relu(self.var_gamma - std))

    def covariance_penalty(self, H_pred: torch.Tensor) -> torch.Tensor:
        """Decorrelates embedding dimensions to maximize information content."""
        B, N, D = H_pred.shape
        z = H_pred.reshape(B * N, D)
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(1, z.shape[0] - 1)  # [D, D]
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D

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
