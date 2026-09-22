"""
JEPA Loss for 5x5 PECT-JEPA: Smooth L1 Prediction Loss + Hypersphere Uniformity Anti-Collapse.

Replaces VICReg isotropic covariance/variance penalties with Hypersphere Uniformity
(Wang & Isola, ICML 2020), providing stable, non-isotropic representation dispersion
grounded in physical eddy current diffusion.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss5x5(nn.Module):
    """
    JEPA latent prediction loss + Hypersphere Uniformity Dispersion.
    Also supports Physical Lift-off Invariance and Phase-Depth Monotonicity alignment.
    """

    def __init__(
        self,
        loss_type: str = "l1",
        eps: float = 1e-8,
        liftoff_invar_weight: float = 0.0,
        phase_align_weight: float = 0.0,
        uniformity_weight: float = 0.05,
        uniformity_t: float = 2.0,
        uniformity_subsample: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.liftoff_invar_weight = liftoff_invar_weight
        self.phase_align_weight = phase_align_weight
        self.uniformity_weight = uniformity_weight
        self.uniformity_t = uniformity_t
        self.uniformity_subsample = uniformity_subsample

    def latent_prediction_loss(self, H_pred: torch.Tensor, H_target: torch.Tensor) -> torch.Tensor:
        safe_eps = max(self.eps, 1e-5)
        # Ensure FP32 precision and sanitize non-finite values to prevent numerical instability
        H_pred = torch.nan_to_num(H_pred.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        H_target = torch.nan_to_num(H_target.float(), nan=0.0, posinf=50.0, neginf=-50.0)
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

    def hypersphere_uniformity_loss(self, H_rep: torch.Tensor) -> torch.Tensor:
        """
        Hypersphere Uniformity Loss (Wang & Isola, ICML 2020).
        Computes the logarithm of the average pairwise Gaussian potential on the unit sphere:
            L_unif = log E_{u,v} [ exp(-t * ||u - v||^2) ]
                   = log E_{u,v} [ exp(2t * (u . v - 1)) ]
        Maintains representation dispersion without forcing artificial isotropic whitening.
        Gradients are strictly Lipschitz bounded with zero division-by-zero risk.
        """
        # Ensure FP32 and sanitize
        z = torch.nan_to_num(H_rep.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        B, N, D = z.shape
        z = z.reshape(B * N, D)
        M = z.shape[0]
        if M < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Subsample tokens for memory and compute efficiency if M > uniformity_subsample
        if self.uniformity_subsample > 0 and M > self.uniformity_subsample:
            idx = torch.randperm(M, device=z.device)[:self.uniformity_subsample]
            z = z[idx]
            M = z.shape[0]

        u = F.normalize(z, p=2, dim=-1, eps=max(self.eps, 1e-8))
        sim = torch.mm(u, u.t())  # [M, M] in [-1, 1]

        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(M, dtype=torch.bool, device=z.device)
        pair_exp = torch.exp(2.0 * self.uniformity_t * (sim - 1.0))[mask]
        unif = torch.log(torch.mean(pair_exp) + 1e-8)
        return torch.nan_to_num(unif, nan=0.0, posinf=0.0, neginf=-10.0)

    def liftoff_invariance_loss(self, H_ctx: torch.Tensor, H_ctx_pert: torch.Tensor) -> torch.Tensor:
        """
        Cosine distance between representations of original and lift-off perturbed inputs:
        L_liftoff = 1 - CosineSimilarity(H_ctx, H_ctx_pert)
        Forces representation to be invariant to probe height / lift-off fluctuations.
        """
        safe_eps = max(self.eps, 1e-6)
        h1 = F.normalize(H_ctx.float(), p=2, dim=-1, eps=safe_eps)
        h2 = F.normalize(H_ctx_pert.float(), p=2, dim=-1, eps=safe_eps)
        cos_sim = torch.sum(h1 * h2, dim=-1)
        return torch.mean(1.0 - cos_sim)

    def phase_depth_alignment_loss(self, z_depth: torch.Tensor, phase_ctx: torch.Tensor) -> torch.Tensor:
        """
        Pearson correlation loss between latent depth projection and physical fundamental harmonic phase:
        L_phase = 1 - |PearsonCorr(z_depth, phase_ctx)|
        Aligns the 1D manifold of latent features monotonically with physical penetration depth d ~ phase_1.
        """
        z_flat = z_depth.float().reshape(-1)
        p_flat = phase_ctx.float().reshape(-1)
        z_c = z_flat - z_flat.mean()
        p_c = p_flat - p_flat.mean()
        denom = (torch.sqrt(torch.sum(z_c ** 2) + 1e-8) * torch.sqrt(torch.sum(p_c ** 2) + 1e-8)) + 1e-6
        pearson_r = torch.sum(z_c * p_c) / denom
        return 1.0 - torch.abs(torch.clamp(pearson_r, min=-1.0, max=1.0))

    def forward(
        self,
        H_pred: torch.Tensor,
        H_target: torch.Tensor,
        H_ctx: Optional[torch.Tensor] = None,
        H_ctx_pert: Optional[torch.Tensor] = None,
        z_depth: Optional[torch.Tensor] = None,
        phase_ctx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_pred = self.latent_prediction_loss(H_pred, H_target)
        zero_loss = torch.tensor(0.0, device=H_pred.device, dtype=torch.float32)

        # Physical Lift-Off Invariance Loss
        l_liftoff = zero_loss
        if self.liftoff_invar_weight > 0.0 and H_ctx is not None and H_ctx_pert is not None:
            l_liftoff = self.liftoff_invariance_loss(H_ctx, H_ctx_pert)

        # Physical Phase-Depth Monotonicity Loss
        l_phase = zero_loss
        if self.phase_align_weight > 0.0 and z_depth is not None and phase_ctx is not None:
            l_phase = self.phase_depth_alignment_loss(z_depth, phase_ctx)

        # Hypersphere Uniformity Dispersion Loss
        l_unif = zero_loss
        if self.uniformity_weight > 0.0 and H_ctx is not None:
            l_unif = self.hypersphere_uniformity_loss(H_ctx)

        total = (
            l_pred
            + self.liftoff_invar_weight * l_liftoff
            + self.phase_align_weight * l_phase
            + self.uniformity_weight * l_unif
        )

        return {
            "loss": total,
            "loss_pred": l_pred.detach(),
            "loss_liftoff": l_liftoff.detach(),
            "loss_phase": l_phase.detach(),
            "loss_unif": l_unif.detach(),
        }
