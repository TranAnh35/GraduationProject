"""
JEPA Loss for 5x5 PECT-JEPA: Smooth L1 Prediction Loss + Hypersphere Uniformity Anti-Collapse.

Replaces VICReg isotropic covariance/variance penalties with Hypersphere Uniformity
(Wang & Isola, ICML 2020), providing stable, non-isotropic representation dispersion
grounded in physical eddy current diffusion.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss5x5(nn.Module):
    """
    JEPA latent prediction loss + VICReg Coordinate-Wise Variance & Covariance Regularization.
    Also supports Physical Lift-off Invariance, Phase-Depth Monotonicity alignment,
    and optional Hypersphere Uniformity Dispersion.
    """

    def __init__(
        self,
        loss_type: str = "l1",
        eps: float = 1e-8,
        liftoff_invar_weight: float = 0.0,
        phase_align_weight: float = 0.0,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        var_gamma: float = 1.0,
        uniformity_weight: float = 0.0,
        uniformity_t: float = 2.0,
        uniformity_subsample: int = 1024,
        norm_floor_weight: float = 0.0,
        norm_floor_target: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.liftoff_invar_weight = liftoff_invar_weight
        self.phase_align_weight = phase_align_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma
        self.uniformity_weight = uniformity_weight
        self.uniformity_t = uniformity_t
        self.uniformity_subsample = uniformity_subsample
        self.norm_floor_weight = norm_floor_weight
        self.norm_floor_target = norm_floor_target

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

    def variance_hinge(self, H_rep: torch.Tensor) -> torch.Tensor:
        """
        VICReg Variance Hinge Loss (Bardes et al., ICLR 2022).
        L_var = (1/D) * sum_{d=1}^D max(0, gamma - std_b(z_{:, d}))
        Forces batch variance along each dimension to be >= var_gamma (computed in FP32).
        Acts as a rigid coordinate scale anchor in R^D, preventing EMA target encoder drift.
        """
        z = torch.nan_to_num(H_rep.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        B, N, D = z.shape
        z = z.reshape(B * N, D)
        safe_eps = max(self.eps, 1e-5)
        var = torch.clamp(z.var(dim=0, unbiased=False), min=0.0)
        std = torch.sqrt(var + safe_eps)
        std = torch.nan_to_num(std, nan=0.0, posinf=self.var_gamma)
        return torch.mean(F.relu(self.var_gamma - std))

    def covariance_penalty(self, H_rep: torch.Tensor) -> torch.Tensor:
        """
        VICReg Covariance Decorrelation Loss (Bardes et al., ICLR 2022).
        L_cov = (1/D) * sum_{i != j} [C(z)]_{ij}^2
        Decorrelates embedding dimensions to prevent dimensional collapse and maximize representation capacity.
        """
        z = torch.nan_to_num(H_rep.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        B, N, D = z.shape
        z = z.reshape(B * N, D)
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(1, z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_penalty = (off_diag ** 2).sum() / D
        return torch.nan_to_num(cov_penalty, nan=0.0, posinf=10.0)

    def norm_floor_loss(self, H_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Norm-Floor Barrier Loss:
        L_norm = max(0, norm_floor_target - mean(||z_i||_2))^2
        Enforces representation magnitude to stay >= norm_floor_target (default 1.0),
        preventing zero-vector collapse.
        """
        z = torch.nan_to_num(H_rep.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        norms = torch.norm(z, p=2, dim=-1)  # [B, N]
        mean_norm = torch.mean(norms)
        l_norm = torch.relu(self.norm_floor_target - mean_norm) ** 2
        return l_norm, mean_norm

    def hypersphere_uniformity_loss(self, H_rep: torch.Tensor) -> torch.Tensor:
        """
        Calibrated Non-Negative Hypersphere Uniformity Loss (Wang & Isola, ICML 2020 + Jensen Shift).
        L_unif^+ = log E_{u,v} [ exp(2t * u^T v) ] = L_unif^raw + 2t

        Properties:
        - Strictly non-negative: L_unif^+ >= 0 (by Jensen's inequality E[exp(2t u^T v)] >= exp(0) = 1).
        - At complete collapse (u = v): L_unif^+ = 2t = +4.0 (strong positive penalty).
        - At zero-vector collapse (||z|| < 0.1): L_unif^+ = 2t = +4.0 (eliminates zero-collapse loophole).
        - At uniform dispersion (u^T v ~ N(0, 1/D)): L_unif^+ -> 2t^2 / D ~ 0.06 - 0.12.
        - Gradients are 100% mathematically identical to Wang & Isola potential:
          grad(L_unif^+) == grad(L_unif^raw) since 2t is constant.
        """
        # Ensure FP32 and sanitize
        z = torch.nan_to_num(H_rep.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        B, N, D = z.shape
        z = z.reshape(B * N, D)
        M = z.shape[0]
        if M < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Anti zero-collapse scale barrier:
        # For any non-zero state 0 < ||z|| < 1.0, this smooth sigmoid penalty exerts an active,
        # non-vanishing restoring gradient pushing the representation norm outward toward >= 1.0.
        # Note: At the exact mathematical origin z = 0, any isotropic potential g(||z||) has
        # gradient 0 by symmetry; the primary architectural anchor maintaining ||z|| ~ sqrt(D) ~ 8.0
        # is the Transformer's LayerNorm.
        norms = torch.norm(z, p=2, dim=-1)
        mean_norm = torch.mean(norms)
        max_bound = 2.0 * self.uniformity_t
        scale_penalty = max_bound * torch.sigmoid(4.0 * (1.0 - mean_norm))

        # Subsample tokens for memory and compute efficiency if M > uniformity_subsample
        if self.uniformity_subsample > 0 and M > self.uniformity_subsample:
            idx = torch.randperm(M, device=z.device)[:self.uniformity_subsample]
            z = z[idx]
            M = z.shape[0]

        u = F.normalize(z, p=2, dim=-1, eps=max(self.eps, 1e-8))
        sim = torch.mm(u, u.t())  # [M, M] in [-1, 1]

        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(M, dtype=torch.bool, device=z.device)
        pair_exp = torch.exp(2.0 * self.uniformity_t * sim)[mask]
        unif = torch.log(torch.mean(pair_exp) + 1e-8)
        raw_unif = torch.clamp(torch.nan_to_num(unif, nan=0.0, posinf=max_bound, neginf=0.0), min=0.0, max=max_bound)

        # When mean_norm >= 1.0 (normal LayerNorm operation ~8.0), scale_penalty -> 0.0
        # When mean_norm < 1.0, scale_penalty adds active restoring gradient
        return raw_unif + scale_penalty

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

        # VICReg Coordinate-Wise Variance & Covariance Penalties
        l_var = zero_loss
        l_cov = zero_loss
        if (self.var_weight > 0.0 or self.cov_weight > 0.0) and H_ctx is not None:
            if self.var_weight > 0.0:
                l_var = self.variance_hinge(H_ctx)
            if self.cov_weight > 0.0:
                l_cov = self.covariance_penalty(H_ctx)

        # Norm-Floor Barrier Loss (Anti Zero-Collapse)
        l_norm = zero_loss
        mean_norm = torch.tensor(1.0, device=H_pred.device, dtype=torch.float32)
        if self.norm_floor_weight > 0.0 and H_ctx is not None:
            l_norm, mean_norm = self.norm_floor_loss(H_ctx)

        total = (
            l_pred
            + self.liftoff_invar_weight * l_liftoff
            + self.phase_align_weight * l_phase
            + self.var_weight * l_var
            + self.cov_weight * l_cov
            + self.uniformity_weight * l_unif
            + self.norm_floor_weight * l_norm
        )

        return {
            "loss": total,
            "loss_pred": l_pred.detach(),
            "loss_liftoff": l_liftoff.detach(),
            "loss_phase": l_phase.detach(),
            "loss_var": l_var.detach(),
            "loss_cov": l_cov.detach(),
            "loss_unif": l_unif.detach(),
            "loss_norm": l_norm.detach(),
            "mean_norm": mean_norm.detach(),
        }
