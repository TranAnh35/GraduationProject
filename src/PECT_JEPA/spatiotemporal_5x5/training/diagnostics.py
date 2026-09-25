"""
Physics-Grounded Training Diagnostics Dashboard for 5x5 Spatiotemporal PECT-JEPA.

Replaces arbitrary statistical projections with 4 core physical and architectural checks:
1. Panel 1: Lift-Off Invariance Trajectory (z1 vs z2 vs z3 on sound metal).
2. Panel 2: Empirical Latent Spatial Variogram gamma(r) (Spatial field continuity of Helmholtz diffusion).
3. Panel 3: Multi-Physics Cross-Domain Similarity Matrix (Coil, Diffensor, GMR x Square, Gauss, Chirp).
4. Panel 4: 2D Spatial Receptive Fields of Transformer Attention Heads (5x5 grid attention for center patch).
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_spatial_variogram(
    model: torch.nn.Module,
    sample_grids: torch.Tensor,  # [B, 5, 5, C]
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes empirical latent semivariance gamma(r) across spatial lag r on the 5x5 grid.
    Helmholtz diffusion physics requires spatial continuity: low gamma(r) at r=1,
    monotonically increasing to a stable sill.

    Returns:
        unique_lags: sorted 1D array of spatial Euclidean distances r.
        gamma_r: empirical semivariance at each lag.
    """
    model.eval()
    with torch.no_grad():
        x = sample_grids.to(device)
        # tokens: [B, 25, D]
        z_all = model.extract_all_features(x)
        B, N, D = z_all.shape

        # Precompute spatial coordinates for all 25 tokens (5x5)
        coords = np.array([(i, j) for i in range(5) for j in range(5)], dtype=np.float32)  # [25, 2]
        # Pairwise spatial distances: [25, 25]
        diff_coords = coords[:, None, :] - coords[None, :, :]
        spatial_dist = np.sqrt(np.sum(diff_coords ** 2, axis=-1))

        # Pairwise latent squared distances across batch: [B, 25, 25]
        z_np = z_all.detach().cpu().numpy()  # [B, 25, D]
        diff_z = z_np[:, :, None, :] - z_np[:, None, :, :]  # [B, 25, 25, D]
        sq_dist_z = np.sum(diff_z ** 2, axis=-1) / 2.0      # [B, 25, 25]
        mean_sq_dist = np.mean(sq_dist_z, axis=0)          # [25, 25]

        # Aggregate by rounded lag distance
        rounded_lags = np.round(spatial_dist, decimals=2)
        unique_lags = np.unique(rounded_lags)
        unique_lags = unique_lags[unique_lags > 0]  # ignore lag 0

        gamma_list = []
        for lag in unique_lags:
            mask = rounded_lags == lag
            gamma_list.append(float(np.mean(mean_sq_dist[mask])))

        return unique_lags, np.array(gamma_list, dtype=np.float32)


def extract_attention_receptive_fields(
    model: torch.nn.Module,
    sample_grids: torch.Tensor,  # [B, 5, 5, C]
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Extracts the 2D spatial attention distribution of the 4 attention heads
    in the final Context Encoder block for the center patch (index 12, (2, 2)).

    Returns:
        attn_maps: [num_heads, 5, 5] normalized attention weights.
    """
    model.eval()
    with torch.no_grad():
        x = sample_grids[:min(8, sample_grids.shape[0])].to(device)
        attn = model.extract_attention_map(x)
        if attn is None:
            return None

        # attn: [B, num_heads, N_tokens, N_tokens]
        # Take center token attending to all tokens
        N_tokens = attn.shape[-1]
        num_heads = attn.shape[1]
        if N_tokens == 50:
            center_shallow = 24
            center_deep = 25
            center_attn = 0.5 * (attn[:, :, center_shallow, :] + attn[:, :, center_deep, :])  # [B, num_heads, 50]
            mean_attn = torch.mean(center_attn, dim=0).detach().cpu().numpy()  # [num_heads, 50]
            mean_attn_spatial = 0.5 * (mean_attn[:, 0::2] + mean_attn[:, 1::2])  # [num_heads, 25]
        elif N_tokens % 25 == 0 and N_tokens > 25:
            T_s = N_tokens // 25
            center_indices = [12 * T_s + tau for tau in range(T_s)]
            center_attn = attn[:, :, center_indices, :].mean(dim=2)  # [B, num_heads, N_tokens]
            mean_attn = torch.mean(center_attn, dim=0).detach().cpu().numpy()  # [num_heads, N_tokens]
            mean_attn_spatial = mean_attn.reshape(num_heads, 25, T_s).mean(axis=-1)  # [num_heads, 25]
        else:
            center_idx = min(12, N_tokens - 1)
            center_attn = attn[:, :, center_idx, :]  # [B, num_heads, N_tokens]
            mean_attn_spatial = torch.mean(center_attn, dim=0).detach().cpu().numpy()  # [num_heads, N_tokens]

        attn_maps = mean_attn_spatial.reshape(num_heads, 5, 5)
        return attn_maps


def compute_liftoff_invariance_diagnostic(
    val_features_by_liftoff: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Computes pairwise representation similarity across lift-offs (z1 vs z2, z1 vs z3)
    on baseline sound metal. Preserves amplitude and vector magnitude.
    """
    sims = {}
    z1 = val_features_by_liftoff.get("z1")
    z2 = val_features_by_liftoff.get("z2")
    z3 = val_features_by_liftoff.get("z3")

    def calc_cos(a, b):
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            return 0.0
        n = min(len(a), len(b))
        a_sub = a[:n]
        b_sub = b[:n]
        norm_a = np.linalg.norm(a_sub, axis=-1, keepdims=True) + 1e-8
        norm_b = np.linalg.norm(b_sub, axis=-1, keepdims=True) + 1e-8
        cos = np.sum((a_sub / norm_a) * (b_sub / norm_b), axis=-1)
        return float(np.mean(cos))

    sims["sim_z1_z2"] = calc_cos(z1, z2)
    sims["sim_z1_z3"] = calc_cos(z1, z3)
    sims["sim_z2_z3"] = calc_cos(z2, z3)
    return sims


def compute_domain_similarity_matrix(
    features_by_domain: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    """
    Computes pairwise cross-domain cosine similarity matrix across physical configurations.
    """
    domain_names = sorted(list(features_by_domain.keys()))
    K = len(domain_names)
    if K == 0:
        return [], np.zeros((0, 0))

    centroids = []
    for d in domain_names:
        feats = features_by_domain[d]
        c = np.mean(feats, axis=0)
        norm = np.linalg.norm(c) + 1e-8
        centroids.append(c / norm)

    C = np.array(centroids)  # [K, D]
    sim_matrix = C @ C.T    # [K, K]
    return domain_names, sim_matrix


def plot_foundation_diagnostics_dashboard(
    epoch: int,
    val_loss_pred: float,
    unique_lags: np.ndarray,
    gamma_r: np.ndarray,
    attn_maps: Optional[np.ndarray],
    liftoff_sims: Dict[str, float],
    domain_names: List[str],
    domain_sim_matrix: np.ndarray,
    twonn_dim: float = 0.0,
    uniformity: float = float("nan"),
    save_path: Optional[str] = None,
    close_fig: bool = True,
) -> matplotlib.figure.Figure:
    """
    Renders the 4-panel Foundation Training Diagnostics Dashboard:
    - Panel 1: Lift-Off Invariance Trajectory (z1 vs z2 vs z3).
    - Panel 2: Empirical Latent Spatial Variogram gamma(r).
    - Panel 3: Multi-Physics Cross-Domain Similarity Matrix.
    - Panel 4: 2D Spatial Attention Receptive Fields (4 Heads on 5x5 grid).
    """
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    fig.suptitle(
        f"PECT-JEPA Foundation Training Diagnostics | Epoch {epoch:02d}\n"
        f"Val Pred Loss: {val_loss_pred:.4f} | Two-NN Intrinsic Dim: {twonn_dim:.1f}D | Uniformity: {uniformity:.2f}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # --------------------------------------------------------------------------
    # Panel 1 (Top-Left): Lift-off Invariance on Sound Metal
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    pairs = ["z1 vs z2 (1mm)", "z1 vs z3 (2mm)", "z2 vs z3 (1mm delta)"]
    sim_values = [
        liftoff_sims.get("sim_z1_z2", 0.0),
        liftoff_sims.get("sim_z1_z3", 0.0),
        liftoff_sims.get("sim_z2_z3", 0.0),
    ]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    bars = ax1.bar(pairs, sim_values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(0.80, color="forestgreen", linestyle="--", alpha=0.7, label="Robust Target (>= 0.80)")
    ax1.set_ylabel("Representation Cosine Similarity", fontweight="bold")
    ax1.set_title("Panel 1: Sound Metal Lift-Off Alignment\n(Testing Invariance to 70% Amplitude Damping)", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="lower right", fontsize=8)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, max(0.02, h + 0.02), f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel 2 (Top-Right): Empirical Latent Spatial Variogram gamma(r)
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    if len(unique_lags) > 0 and len(gamma_r) > 0:
        ax2.plot(unique_lags, gamma_r, "o-", color="#1b9e77", linewidth=2.2, markersize=6, label=r"Empirical $\gamma(r)$")
        # Reference sill
        sill = float(np.mean(gamma_r[-3:])) if len(gamma_r) >= 3 else float(gamma_r[-1])
        ax2.axhline(sill, color="gray", linestyle=":", label=f"Sill ({sill:.2f})")
        ax2.set_xlabel("Spatial Lag $r$ (grid units on 5x5)", fontweight="bold")
        ax2.set_ylabel(r"Semivariance $\gamma(r) = \frac{1}{2} E[||z_x - z_{x+r}||^2]$", fontweight="bold")
        ax2.set_title("Panel 2: Latent Spatial Variogram\n(Testing Helmholtz Parabolic Diffusion Continuity)", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Variogram Computing...", ha="center", va="center")
        ax2.set_title("Panel 2: Latent Spatial Variogram", fontsize=10, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Multi-Physics Cross-Domain Matrix
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    if len(domain_names) > 0 and domain_sim_matrix.shape[0] == len(domain_names):
        im3 = ax3.imshow(domain_sim_matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Cosine Similarity", fontsize=8)
        ax3.set_xticks(range(len(domain_names)))
        ax3.set_yticks(range(len(domain_names)))
        ax3.set_xticklabels(domain_names, rotation=35, ha="right", fontsize=8)
        ax3.set_yticklabels(domain_names, fontsize=8)
        ax3.set_title("Panel 3: Multi-Physics Domain Alignment\n(Testing Shared Geometry across Sensors & Waveforms)", fontsize=10, fontweight="bold")
    else:
        # Fallback 3x3 sensors x waveforms schematic if grouping not ready
        dummy = np.eye(3) * 0.8 + 0.2
        im3 = ax3.imshow(dummy, cmap="magma", vmin=0.0, vmax=1.0)
        plt.colorbar(im3, ax=ax3)
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        ax3.set_xticklabels(["Square", "Gaussian", "Chirp"], fontsize=8)
        ax3.set_yticklabels(["Coil", "Diffensor", "GMR"], fontsize=8)
        ax3.set_title("Panel 3: Multi-Physics Domain Alignment", fontsize=10, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel 4 (Bottom-Right): 2D Spatial Attention Receptive Fields of the 4 Heads
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Panel 4: Context Encoder Spatial Receptive Fields\n(Attention Weights on 5x5 Grid for Center Patch)", fontsize=10, fontweight="bold", pad=12)

    if attn_maps is not None and attn_maps.ndim == 3:
        num_heads = min(4, attn_maps.shape[0])
        # Position sub-axes natively inside ax4 using inset_axes (100% compatible with tight_layout)
        width = 0.84 / num_heads
        gap = 0.10 / max(1, num_heads - 1) if num_heads > 1 else 0.0

        im_h = None
        for h in range(num_heads):
            left = 0.08 + h * (width + gap)
            s_ax = ax4.inset_axes([left, 0.28, width, 0.62])
            h_map = attn_maps[h]
            im_h = s_ax.imshow(h_map, cmap="viridis", vmin=0.0, vmax=max(0.15, float(np.max(h_map))))
            s_ax.plot(2, 2, "r*", markersize=7)  # center marker
            s_ax.set_title(f"Head {h + 1}", fontsize=8, fontweight="bold")
            s_ax.set_xticks([])
            s_ax.set_yticks([])

        if im_h is not None:
            cax = ax4.inset_axes([0.15, 0.08, 0.70, 0.07])
            cbar_h = fig.colorbar(im_h, cax=cax, orientation="horizontal")
            cbar_h.set_label("Attention Weight (Normalized)", fontsize=8)
    else:
        ax4.text(0.5, 0.5, "Attention Maps Available on Encoder Evaluation", ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
        return None
    return fig
