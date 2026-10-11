"""
Latent Diagnostics for Cross-Condition PECT-JEPA.
Computes Singular Value Decomposition (SVD) effective rank, Condition number,
and Latent Signal-to-Noise Ratio (SNR).
"""

from typing import Dict, Tuple
import torch


@torch.no_grad()
def compute_svd_diagnostics(representations: torch.Tensor) -> Dict[str, float]:
    """
    Computes SVD spectrum decay and Roy-Vetterli effective rank:
    R_eff = exp(- sum p_i * ln(p_i)), where p_i = s_i / sum(s_j).
    """
    if representations.ndim > 2:
        representations = representations.reshape(-1, representations.shape[-1])

    # Center representations
    z_cent = representations - representations.mean(dim=0, keepdim=True)
    N, D = z_cent.shape

    # SVD on covariance matrix
    cov = (z_cent.T @ z_cent) / max(N - 1, 1)
    s = torch.linalg.eigvalsh(cov)  # Eigenvalues in ascending order
    s = torch.clamp(s, min=1e-8)
    s_desc = torch.flip(s, dims=[0])  # Descending order

    p = s_desc / s_desc.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
    eff_rank = float(torch.exp(torch.tensor(entropy)).item())

    cond_number = float((s_desc[0] / s_desc[-1]).item())
    top1_ratio = float((s_desc[0] / s_desc.sum()).item())

    return {
        "effective_rank": eff_rank,
        "condition_number": cond_number,
        "top1_ratio": top1_ratio,
        "dimension": D,
    }


@torch.no_grad()
def compute_latent_snr(
    representations: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Computes Latent Signal-to-Noise Ratio (Separability index):
    SNR = ||mu_defect - mu_sound||_2 / sqrt(0.5 * (sigma_defect^2 + sigma_sound^2))
    """
    if representations.ndim > 2:
        representations = representations.reshape(-1, representations.shape[-1])
    labels = labels.view(-1)

    defect_mask = labels > 0.5
    sound_mask = ~defect_mask

    if defect_mask.sum() < 2 or sound_mask.sum() < 2:
        return 0.0

    z_def = representations[defect_mask]
    z_snd = representations[sound_mask]

    mu_def = z_def.mean(dim=0)
    mu_snd = z_snd.mean(dim=0)

    delta_norm = torch.norm(mu_def - mu_snd, p=2).item()
    var_def = z_def.var(dim=0, unbiased=False).mean().item()
    var_snd = z_snd.var(dim=0, unbiased=False).mean().item()

    pooled_std = max(1e-6, (0.5 * (var_def + var_snd)) ** 0.5)
    return float(delta_norm / pooled_std)
