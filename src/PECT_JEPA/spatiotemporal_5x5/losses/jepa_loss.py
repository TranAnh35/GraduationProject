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
        fluct_weight: float = 2.0,
        adaptive_disturbance_weight: float = 2.0,
        temporal_mono_weight: float = 0.05,
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
        self.fluct_weight = fluct_weight
        self.adaptive_disturbance_weight = adaptive_disturbance_weight
        self.temporal_mono_weight = temporal_mono_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma
        self.uniformity_weight = uniformity_weight
        self.uniformity_t = uniformity_t
        self.uniformity_subsample = uniformity_subsample
        self.norm_floor_weight = norm_floor_weight
        self.norm_floor_target = norm_floor_target

    def compute_disturbance_weights(self, x_raw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Computes batch sample weights based on spatial-temporal field disturbance:
            xi(x) = (1 / C) * sum_{c} [ StdDev_{s in 25}(x(s, :, c)) ]
        w_b = 1.0 + kappa * (xi_b - min(xi)) / (max(xi) - min(xi) + eps)
        where kappa = self.adaptive_disturbance_weight.
        Gives defect/edge samples up to (1 + kappa)x higher weight than uniform sound metal.
        """
        if self.adaptive_disturbance_weight <= 0.0 or x_raw is None:
            return None
        # x_raw shape: [B, 5, 5, C] or [B, 5, 5, T, C] or [B, 25, ...]
        B = x_raw.shape[0]
        x_flat = x_raw.reshape(B, 25, -1).float()
        spatial_std = x_flat.std(dim=1).mean(dim=-1)  # [B]
        std_min = spatial_std.min()
        std_max = spatial_std.max()
        denom = std_max - std_min
        if denom < 1e-7:
            return torch.ones(B, 1, 1, device=x_raw.device, dtype=torch.float32)
        norm_std = (spatial_std - std_min) / (denom + 1e-6)
        weights = 1.0 + self.adaptive_disturbance_weight * norm_std
        return weights.view(B, 1, 1).detach()

    def temporal_diffusion_monotonicity_loss(
        self,
        delta_pred: Optional[torch.Tensor],
        target_indices: Optional[torch.Tensor],
        margin: float = 0.05,
    ) -> torch.Tensor:
        """
        Enforces monotonic growth of representation deviation along diffusion stages:
            norm(delta_H(tau_{k+1})) >= norm(delta_H(tau_k)) + margin
        grounded in the irreversible diffusion of eddy currents through depth.
        """
        if self.temporal_mono_weight <= 0.0 or delta_pred is None or target_indices is None:
            return torch.tensor(0.0, device=delta_pred.device if delta_pred is not None else "cpu", dtype=torch.float32)

        if target_indices.max() < 50:
            return torch.tensor(0.0, device=delta_pred.device, dtype=torch.float32)

        B, N_tgt, D = delta_pred.shape
        stages = target_indices % 4  # [B, N_tgt] in {0, 1, 2, 3}
        norms = torch.norm(delta_pred.float(), p=2, dim=-1)  # [B, N_tgt]

        unique_stages = torch.unique(stages)
        if len(unique_stages) < 2:
            return torch.tensor(0.0, device=delta_pred.device, dtype=torch.float32)

        stage_means = []
        valid_stages = []
        for stg in unique_stages:
            mask = (stages == stg)  # [B, N_tgt]
            if mask.any():
                denom = mask.sum(dim=-1).clamp(min=1)  # [B]
                mean_norm = (norms * mask.float()).sum(dim=-1) / denom  # [B]
                stage_means.append(mean_norm)
                valid_stages.append(stg.item())

        if len(valid_stages) < 2:
            return torch.tensor(0.0, device=delta_pred.device, dtype=torch.float32)

        loss_mono = torch.tensor(0.0, device=delta_pred.device, dtype=torch.float32)
        count = 0
        for k in range(len(valid_stages) - 1):
            if valid_stages[k + 1] > valid_stages[k]:
                diff = stage_means[k] - stage_means[k + 1] + margin
                loss_mono = loss_mono + torch.relu(diff).mean()
                count += 1

        if count > 0:
            return loss_mono / count
        return torch.tensor(0.0, device=delta_pred.device, dtype=torch.float32)

    def fluctuation_prediction_loss(
        self,
        H_pred: torch.Tensor,
        H_target: torch.Tensor,
        target_indices: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Context-Referenced Fluctuation Prediction Loss with Adaptive Field Disturbance Weighting:
        Decomposes target predictions into DC mean background decay and AC spatial fluctuation:
            L_pred = L_mean + fluct_weight * L_fluct

        Where:
            L_mean = mean_tau || bar{H}_pred(tau) - bar{H}_tgt(tau) ||_1
            L_fluct = mean_i || (H_pred(s_i, tau_i) - bar{H}_pred(tau_i)) - (H_tgt(s_i, tau_i) - bar{H}_tgt(tau_i)) ||_1

        Optionally weights batch items by intrinsic field disturbance weights to resolve 95% sound metal imbalance.
        """
        H_pred = torch.nan_to_num(H_pred.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        H_target = torch.nan_to_num(H_target.float(), nan=0.0, posinf=50.0, neginf=-50.0)
        B, N_tgt, D = H_pred.shape

        def weighted_l1(a, b):
            diff = (a - b).abs()
            if weights is not None:
                diff = diff * weights
            return diff.mean()

        if self.fluct_weight <= 0.0:
            if weights is not None:
                l_pred = weighted_l1(H_pred, H_target)
            else:
                l_pred = self.latent_prediction_loss(H_pred, H_target)
            return l_pred, torch.tensor(0.0, device=H_pred.device, dtype=torch.float32)

        if target_indices is not None and target_indices.max() >= 50:
            stages = torch.unique(target_indices % 4)
            l_mean_list = []
            l_fluct_list = []

            for stg in stages:
                mask_stg = (target_indices % 4 == stg)  # [B, N_tgt]
                k_tokens = mask_stg[0].sum().item()
                if k_tokens < 2:
                    continue

                H_p_stg = H_pred[mask_stg].view(B, k_tokens, D)
                H_t_stg = H_target[mask_stg].view(B, k_tokens, D)

                mean_p = H_p_stg.mean(dim=1, keepdim=True)  # [B, 1, D]
                mean_t = H_t_stg.mean(dim=1, keepdim=True)  # [B, 1, D]

                l_mean_list.append(weighted_l1(mean_p, mean_t))

                fluct_p = H_p_stg - mean_p
                fluct_t = H_t_stg - mean_t
                l_fluct_list.append(weighted_l1(fluct_p, fluct_t))

            if l_mean_list:
                l_mean = torch.stack(l_mean_list).mean()
                l_fluct = torch.stack(l_fluct_list).mean()
                total = l_mean + self.fluct_weight * l_fluct
                return total, l_fluct

        mean_p = H_pred.mean(dim=1, keepdim=True)
        mean_t = H_target.mean(dim=1, keepdim=True)
        l_mean = weighted_l1(mean_p, mean_t)
        fluct_p = H_pred - mean_p
        fluct_t = H_target - mean_t
        l_fluct = weighted_l1(fluct_p, fluct_t)
        total = l_mean + self.fluct_weight * l_fluct
        return total, l_fluct

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
        target_indices: Optional[torch.Tensor] = None,
        H_ctx: Optional[torch.Tensor] = None,
        H_ctx_pert: Optional[torch.Tensor] = None,
        z_depth_tgt: Optional[torch.Tensor] = None,
        z_depth_pred: Optional[torch.Tensor] = None,
        phase_tgt: Optional[torch.Tensor] = None,
        z_depth: Optional[torch.Tensor] = None,
        phase_ctx: Optional[torch.Tensor] = None,
        x_raw: Optional[torch.Tensor] = None,
        delta_pred: Optional[torch.Tensor] = None,
        H_rep_reg: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        weights = self.compute_disturbance_weights(x_raw)
        l_pred, l_fluct = self.fluctuation_prediction_loss(H_pred, H_target, target_indices=target_indices, weights=weights)
        zero_loss = torch.tensor(0.0, device=H_pred.device, dtype=torch.float32)

        # Physical Temporal Diffusion Monotonicity Loss
        l_mono = zero_loss
        if self.temporal_mono_weight > 0.0 and delta_pred is not None and target_indices is not None:
            l_mono = self.temporal_diffusion_monotonicity_loss(delta_pred, target_indices)

        # Physical Lift-Off Invariance Loss
        l_liftoff = zero_loss
        if self.liftoff_invar_weight > 0.0 and H_ctx is not None and H_ctx_pert is not None:
            l_liftoff = self.liftoff_invariance_loss(H_ctx, H_ctx_pert)

        # Physical Phase-Depth Monotonicity Loss (Late Diffusion Stage Alignment)
        l_phase = zero_loss
        if self.phase_align_weight > 0.0:
            if z_depth_tgt is not None and phase_tgt is not None:
                l_p_tgt = self.phase_depth_alignment_loss(z_depth_tgt, phase_tgt)
                l_p_pred = self.phase_depth_alignment_loss(z_depth_pred, phase_tgt) if z_depth_pred is not None else l_p_tgt
                l_phase = 0.5 * (l_p_tgt + l_p_pred)
            elif z_depth is not None and phase_ctx is not None:
                l_phase = self.phase_depth_alignment_loss(z_depth, phase_ctx)

        # Embedding to regularize (defaults to H_rep_reg if available, else H_ctx)
        rep_reg = H_rep_reg if H_rep_reg is not None else H_ctx

        # Hypersphere Uniformity Dispersion Loss
        l_unif = zero_loss
        if self.uniformity_weight > 0.0 and rep_reg is not None:
            l_unif = self.hypersphere_uniformity_loss(rep_reg)

        # VICReg Coordinate-Wise Variance & Covariance Penalties
        l_var = zero_loss
        l_cov = zero_loss
        if (self.var_weight > 0.0 or self.cov_weight > 0.0) and rep_reg is not None:
            if self.var_weight > 0.0:
                l_var = self.variance_hinge(rep_reg)
            if self.cov_weight > 0.0:
                l_cov = self.covariance_penalty(rep_reg)

        # Norm-Floor Barrier Loss (Anti Zero-Collapse)
        l_norm = zero_loss
        mean_norm = torch.tensor(1.0, device=H_pred.device, dtype=torch.float32)
        if self.norm_floor_weight > 0.0 and rep_reg is not None:
            l_norm, mean_norm = self.norm_floor_loss(rep_reg)

        total = (
            l_pred
            + self.temporal_mono_weight * l_mono
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
            "loss_fluct": l_fluct.detach(),
            "loss_mono": l_mono.detach(),
            "loss_liftoff": l_liftoff.detach(),
            "loss_phase": l_phase.detach(),
            "loss_var": l_var.detach(),
            "loss_cov": l_cov.detach(),
            "loss_unif": l_unif.detach(),
            "loss_norm": l_norm.detach(),
            "mean_norm": mean_norm.detach(),
        }
