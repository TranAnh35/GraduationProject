"""
JEPA Loss Functions for PECT-JEPA v0.2 (Module 6, implement.md).
Implements normalized L2 / MSE latent prediction loss, cosine distance, and smooth L1 loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """
    Module 6: JEPA Latent Prediction Loss (implement.md).
    Computes distance between predicted target latents and EMA target latents in normalized feature space.
    """
    def __init__(self, loss_type: str = "normalized_l2", eps: float = 1e-8):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        predicted_target: torch.Tensor,
        target_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_target: [B, N_tgt, D] predicted target representations from Predictor
            target_latent: [B, N_tgt, D] target representations from EMA Target Encoder

        Returns:
            Scalar loss value
        """
        assert predicted_target.shape == target_latent.shape, \
            f"Shape mismatch: {predicted_target.shape} vs {target_latent.shape}"

        if self.loss_type == "normalized_l2":
            # Normalized L2 distance along feature dimension D
            p = F.normalize(predicted_target, p=2, dim=-1, eps=self.eps)
            t = F.normalize(target_latent, p=2, dim=-1, eps=self.eps)
            loss = torch.mean((p - t) ** 2)
        elif self.loss_type == "cosine":
            p = F.normalize(predicted_target, p=2, dim=-1, eps=self.eps)
            t = F.normalize(target_latent, p=2, dim=-1, eps=self.eps)
            loss = 1.0 - torch.mean(torch.sum(p * t, dim=-1))
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted_target, target_latent)
        elif self.loss_type == "mse":
            loss = F.mse_loss(predicted_target, target_latent)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss
