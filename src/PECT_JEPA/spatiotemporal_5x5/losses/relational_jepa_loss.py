"""
Cross-Waveform Relational Consistency & Factorized Anti-Collapse Loss for PECT-JEPA v2-A.

Implements:
1. Label-Free Relational Consistency Loss:
   Preserves pairwise spatial correlation distance matrices across different waveforms:
   L_rel = || D_A - D_B ||_F^2 / N^2
2. Shared/Private Subspace Orthogonality Regularization:
   L_orth = || Z_inv^T Z_meas ||_F^2 / (d_inv * d_meas)
3. Shared VICReg Variance & Covariance Anti-Collapse Anchor on z_inv
4. Preserves Core JEPA Prediction Loss for field diffusion reconstruction
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.PECT_JEPA.spatiotemporal_5x5.losses.jepa_loss import JEPALoss5x5


class RelationalJEPALoss5x5(JEPALoss5x5):
    """
    Unified Factorized JEPA Loss:
    L = L_JEPA + lambda_rel * L_rel + lambda_orth * L_orth + lambda_var * L_var + lambda_cov * L_cov
    """
    def __init__(
        self,
        relational_weight: float = 1.0,
        orthogonality_weight: float = 0.5,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        var_gamma: float = 1.0,
        **kwargs
    ):
        super().__init__(var_weight=var_weight, cov_weight=cov_weight, var_gamma=var_gamma, **kwargs)
        self.relational_weight = relational_weight
        self.orthogonality_weight = orthogonality_weight

    @staticmethod
    def compute_spatial_relational_matrix(z_tokens: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes pairwise cosine distance matrix across spatial tokens in a patch:
        Input: z_tokens [B, N, d_inv] where N=25 probes
        Output: S [B, N, N] pairwise cosine similarity matrix
        """
        z_norm = F.normalize(z_tokens.float(), p=2, dim=-1, eps=eps)  # [B, N, d_inv]
        S = torch.bmm(z_norm, z_norm.transpose(1, 2))                # [B, N, N]
        return S

    def cross_waveform_relational_loss(
        self,
        z_inv_A: torch.Tensor,
        z_inv_B: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Label-Free Spatial Relational Consistency:
        Forces the pairwise spatial geometry of invariant representations to match across waveforms A and B:
        L_rel = (1 / (B * N^2)) * sum_{b} || S_A - S_B ||_F^2
        """
        S_A = self.compute_spatial_relational_matrix(z_inv_A, eps=eps)
        S_B = self.compute_spatial_relational_matrix(z_inv_B, eps=eps)
        B, N, _ = S_A.shape
        diff_sq = (S_A - S_B) ** 2
        return torch.sum(diff_sq) / (B * (N ** 2))

    def subspace_orthogonality_loss(
        self,
        z_inv: torch.Tensor,
        z_meas: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Cross-Subspace Decorrelation Penalty:
        L_orth = || Z_inv^T Z_meas ||_F^2 / (d_inv * d_meas)
        Prevents information leakage and enforces true factorization between physical state and measurement condition.
        """
        # Flatten batch and spatial tokens: [M, d_inv], [M, d_meas]
        inv_flat = z_inv.reshape(-1, z_inv.shape[-1]).float()
        meas_flat = z_meas.reshape(-1, z_meas.shape[-1]).float()
        M, d_inv = inv_flat.shape
        d_meas = meas_flat.shape[-1]

        # Zero-center along sample dimension
        inv_c = inv_flat - inv_flat.mean(dim=0, keepdim=True)
        meas_c = meas_flat - meas_flat.mean(dim=0, keepdim=True)

        # Cross-covariance matrix: C_cross in R^{d_inv x d_meas}
        C_cross = torch.mm(inv_c.t(), meas_c) / max(M - 1, 1)
        frob_sq = torch.sum(C_cross ** 2)
        return frob_sq / (d_inv * d_meas + eps)

    def forward_factorized(
        self,
        base_jepa_loss_dict: Dict[str, torch.Tensor],
        z_inv_A: torch.Tensor,
        z_meas_A: torch.Tensor,
        z_inv_B: Optional[torch.Tensor] = None,
        file_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Combines standard JEPA diffusion loss with factorized regularization:
        - Relational consistency across paired inputs (A and B)
        - Subspace orthogonality between z_inv and z_meas
        - VICReg variance hinge and covariance decorrelation on z_inv
        """
        loss_dict = dict(base_jepa_loss_dict)
        zero_loss = torch.tensor(0.0, device=z_inv_A.device, dtype=torch.float32)

        # 1. Subspace Orthogonality Penalty
        l_orth = self.subspace_orthogonality_loss(z_inv_A, z_meas_A)

        # 2. Cross-Waveform Relational Consistency
        l_rel = zero_loss
        if z_inv_B is not None and self.relational_weight > 0.0:
            l_rel = self.cross_waveform_relational_loss(z_inv_A, z_inv_B)

        # 3. VICReg on Shared Invariant Branch z_inv
        l_var_inv = self.intra_scan_variance_hinge(z_inv_A, file_ids=file_ids) if getattr(self, "use_intra_scan_vicreg", False) and file_ids is not None else self.variance_hinge(z_inv_A, file_ids=file_ids)
        l_cov_inv = self.intra_scan_covariance_penalty(z_inv_A, file_ids=file_ids) if getattr(self, "use_intra_scan_vicreg", False) and file_ids is not None else self.covariance_penalty(z_inv_A, file_ids=file_ids)

        # Aggregate Total Factorized Loss
        total_loss = (
            loss_dict["loss"]
            + self.relational_weight * l_rel
            + self.orthogonality_weight * l_orth
            + self.var_weight * l_var_inv
            + self.cov_weight * l_cov_inv
        )

        loss_dict.update({
            "loss": total_loss,
            "loss_rel": l_rel.detach(),
            "loss_orth": l_orth.detach(),
            "loss_var_inv": l_var_inv.detach(),
            "loss_cov_inv": l_cov_inv.detach(),
        })
        return loss_dict
