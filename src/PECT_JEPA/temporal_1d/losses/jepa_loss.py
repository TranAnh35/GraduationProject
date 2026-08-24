"""
JEPA Loss function for 1D Temporal PECT-JEPA (implement.md, Section 4.7).
Normalized L2 distance in latent space between H_pred and H_target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss1D(nn.Module):
    """
    Normalized L2 Loss between predicted and target representations.
    """
    def __init__(self, loss_type: str = "normalized_l2", eps: float = 1e-8):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, H_pred: torch.Tensor, H_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_pred: [B, N_tgt, D] predicted target representations
            H_target: [B, N_tgt, D] EMA target representations (detached)

        Returns:
            Scalar loss
        """
        if self.loss_type == "normalized_l2":
            # L2 normalize along embedding dimension D
            pred_norm = F.normalize(H_pred, p=2, dim=-1, eps=self.eps)
            target_norm = F.normalize(H_target, p=2, dim=-1, eps=self.eps)
            loss = torch.mean(torch.sum((pred_norm - target_norm) ** 2, dim=-1))
            return loss

        elif self.loss_type == "cosine":
            pred_norm = F.normalize(H_pred, p=2, dim=-1, eps=self.eps)
            target_norm = F.normalize(H_target, p=2, dim=-1, eps=self.eps)
            cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
            loss = torch.mean(1.0 - cos_sim)
            return loss

        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(H_pred, H_target)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
