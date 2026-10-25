"""
Latent Diagnostics for Cross-Condition PECT-JEPA.
Computes Singular Value Decomposition (SVD) effective rank, Condition number,
and Latent Signal-to-Noise Ratio (SNR).

Numerical policy: diagnostics are telemetry, never a training dependency. A
single non-finite activation (AMP overflow, diverging batch) used to abort a
multi-hour run inside `torch.linalg.eigvalsh` (LAPACK info=62), so the spectrum
is computed in float64 with non-finite rows dropped, and an SVD fallback is used
when the symmetric eigensolver still refuses an ill-conditioned covariance.
"""

import math
from typing import Dict
import torch

_EIG_FLOOR = 1e-8


def _empty_diagnostics(dimension: int, n_nonfinite_rows: int) -> Dict[str, float]:
    return {
        "effective_rank": 0.0,
        "condition_number": float("inf"),
        "top1_ratio": 0.0,
        "dimension": dimension,
        "n_nonfinite_rows": n_nonfinite_rows,
    }


@torch.no_grad()
def compute_svd_diagnostics(representations: torch.Tensor) -> Dict[str, float]:
    """
    Computes SVD spectrum decay and Roy-Vetterli effective rank:
    R_eff = exp(- sum p_i * ln(p_i)), where p_i = s_i / sum(s_j).

    Never raises: an unusable spectrum (all rows non-finite, or a covariance the
    solver rejects) returns zeros/inf plus `n_nonfinite_rows`, so a training run
    is never killed by telemetry.
    """
    if representations.ndim > 2:
        representations = representations.reshape(-1, representations.shape[-1])

    # float64 on CPU: fp32 syevd fails (info=62) on an ill-conditioned
    # covariance, and a fp16 tensor left behind by autocast is not supported by
    # the CPU solver at all.
    z = representations.detach().to(device="cpu", dtype=torch.float64)
    D = int(z.shape[-1])

    finite = torch.isfinite(z).all(dim=1)
    n_bad = int((~finite).sum().item())
    if n_bad:
        z = z[finite]
    if z.shape[0] < 2:
        return _empty_diagnostics(D, n_bad)

    z_cent = z - z.mean(dim=0, keepdim=True)
    N = z_cent.shape[0]

    cov = (z_cent.T @ z_cent) / max(N - 1, 1)
    cov = 0.5 * (cov + cov.T)  # exact symmetry; eigh reads one triangle only

    try:
        s = torch.linalg.eigvalsh(cov)  # eigenvalues in ascending order
    except Exception:
        # lambda_i = s_i^2 / (N-1) of the centered data; the SVD driver
        # converges where the symmetric eigensolver gives up.
        s = torch.linalg.svdvals(z_cent) ** 2 / max(N - 1, 1)
        s, _ = torch.sort(s)  # match eigvalsh ordering

    s = torch.clamp(s, min=_EIG_FLOOR)
    if s.numel() < D:
        s = torch.cat([s, torch.full((D - s.numel(),), _EIG_FLOOR, dtype=s.dtype)])
    s_desc = torch.flip(s, dims=[0])  # descending order

    total = s_desc.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return _empty_diagnostics(D, n_bad)

    p = s_desc / total
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
    eff_rank = float(math.exp(entropy))

    cond_number = float((s_desc[0] / s_desc[-1]).item())
    top1_ratio = float((s_desc[0] / total).item())

    return {
        "effective_rank": eff_rank,
        "condition_number": cond_number,
        "top1_ratio": top1_ratio,
        "dimension": D,
        "n_nonfinite_rows": n_bad,
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
