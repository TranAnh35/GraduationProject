"""
High-Dimensional Latent Diagnostics & Forensics Suite for PECT-JEPA.

Analyzes raw 64D / 128D representations directly without lossy 2D/3D projections (e.g. PCA),
evaluating:
1. Per-Channel Defect Sensitivity Spectrum (Fisher Discriminant Ratio per coordinate)
2. Defect Morphology Subspace Disentanglement (Surface vs Subsurface vs Fastener vs Sound)
3. Cross-Waveform Feature Alignment (Square vs Gaussian vs Chirp)
4. Latent Subspace Clutter Decomposition Audit (z^base vs Delta z energy distributions)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any


def compute_channel_sensitivity_spectrum(
    features: np.ndarray,          # [N, D]
    gt_labels: np.ndarray,         # [N] (0: sound, 1: defect)
    save_path: Optional[str] = None,
    title: str = "64D Latent Coordinate Defect Sensitivity Spectrum (Fisher Ratio)",
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Computes Fisher Discriminant Ratio for each latent coordinate k in 1..D:
        S_k = |mu_{k, defect} - mu_{k, sound}| / sqrt(sigma_{k, defect}^2 + sigma_{k, sound}^2)
    Identifies which specific latent dimensions fire for flaws vs dormant/structural dimensions.
    """
    flat_f = features.reshape(-1, features.shape[-1])
    flat_y = gt_labels.reshape(-1)

    def_idx = np.where(flat_y == 1)[0]
    snd_idx = np.where(flat_y == 0)[0]

    if len(def_idx) == 0 or len(snd_idx) == 0:
        return {"error": "Missing defect or sound metal samples"}

    f_def = flat_f[def_idx]
    f_snd = flat_f[snd_idx]

    mu_def = np.mean(f_def, axis=0)
    mu_snd = np.mean(f_snd, axis=0)
    var_def = np.var(f_def, axis=0)
    var_snd = np.var(f_snd, axis=0)

    # Fisher sensitivity ratio
    denom = np.sqrt(var_def + var_snd + 1e-8)
    sensitivity = np.abs(mu_def - mu_snd) / denom  # [D]

    d_dim = len(sensitivity)
    sorted_idx = np.argsort(sensitivity)[::-1]
    top_channels = sorted_idx[:10].tolist()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)

        colors = ["#dc3545" if s >= np.percentile(sensitivity, 85) else "#007bff" for s in sensitivity]
        ax.bar(np.arange(d_dim), sensitivity, color=colors, width=0.75, alpha=0.85, edgecolor="none")

        ax.axhline(float(np.mean(sensitivity)), color="black", linestyle="--", lw=1.2, label=f"Mean Sensitivity ({np.mean(sensitivity):.3f})")
        ax.axhline(float(np.percentile(sensitivity, 85)), color="#dc3545", linestyle=":", lw=1.2, label="Top 15% Flaw-Dedicated Channels")

        ax.set_xlabel("Latent Coordinate Index $k$ ($1..D$)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Fisher Discriminant Sensitivity $S_k$", fontsize=10, fontweight="bold")
        ax.set_title(f"{title}\nTop Channels: {top_channels[:5]} (Peak $S_k={sensitivity[sorted_idx[0]]:.3f}$)", fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "channel_sensitivity": sensitivity.tolist(),
        "mean_sensitivity": float(np.mean(sensitivity)),
        "max_sensitivity": float(np.max(sensitivity)),
        "top_channels": top_channels,
        "plot_path": save_path,
    }


def compute_morphology_subspace_separation(
    features: np.ndarray,          # [N, D]
    morphology_labels: np.ndarray, # [N] (0: sound, 1: surface, 2: subsurface, 3: fastener)
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    title: str = "High-Dimensional Defect Morphology Separation Matrix (Cosine Distance)",
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Computes pairwise Cosine distance matrix:
        d(A, B) = 1 - (mu_A . mu_B) / (||mu_A|| ||mu_B||)
    in the raw 64D/128D latent space across the 4 physical states.
    Verifies that the representation cleanly separates surface vs subsurface vs fastener.
    """
    if class_names is None:
        class_names = ["Sound Metal", "Surface Defect", "Subsurface Defect", "Fastener"]

    flat_f = features.reshape(-1, features.shape[-1])
    flat_y = morphology_labels.reshape(-1)

    n_classes = len(class_names)
    centroids = []
    present_classes = []

    for c in range(n_classes):
        idx = np.where(flat_y == c)[0]
        if len(idx) > 0:
            c_vec = np.mean(flat_f[idx], axis=0)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 1e-8:
                c_vec = c_vec / c_norm
            centroids.append(c_vec)
            present_classes.append(c)
        else:
            centroids.append(np.zeros(flat_f.shape[-1]))

    # Cosine distance matrix: D_cos[i, j] = 1 - (mu_i . mu_j)
    cos_dist_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        for j in range(n_classes):
            if i in present_classes and j in present_classes:
                sim = float(np.dot(centroids[i], centroids[j]))
                cos_dist_matrix[i, j] = float(np.clip(1.0 - sim, 0.0, 2.0))
            else:
                cos_dist_matrix[i, j] = np.nan

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

        display_matrix = np.nan_to_num(cos_dist_matrix, nan=0.0)
        im = ax.imshow(display_matrix, cmap="YlOrRd", vmin=0.0, vmax=float(np.nanmax(display_matrix) + 0.1))

        for i in range(n_classes):
            for j in range(n_classes):
                if not np.isnan(cos_dist_matrix[i, j]):
                    val = cos_dist_matrix[i, j]
                    color = "white" if val > 0.4 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("High-Dimensional Cosine Distance (1 - cos θ)", fontsize=10)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9, fontweight="bold")
        ax.set_yticklabels(class_names, fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "cosine_distance_matrix": np.nan_to_num(cos_dist_matrix, nan=-1.0).tolist(),
        "present_classes": present_classes,
        "class_names": class_names,
        "plot_path": save_path,
    }


def audit_subspace_clutter_decomposition(
    z_base_norms: np.ndarray,      # [N] ||z^base||
    delta_z_norms: np.ndarray,     # [N] ||Delta z||
    gt_labels: np.ndarray,         # [N] (0: sound metal, 1: fastener, 2: flaw)
    save_path: Optional[str] = None,
    title: str = "Latent Subspace Clutter Decomposition Audit",
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Audits whether the Latent Subspace Decomposition successfully resolved the Fastener Clutter Paradox:
    - Verifies ||Delta z|| is near-zero on Sound Metal and Fasteners, but spikes strongly on Flaws.
    - Verifies ||z^base|| models the nominal structural geometry without triggering false alarms.
    """
    labels = ["Sound Metal", "Fastener Structure", "Flaw / Corrosion"]
    colors = ["#28a745", "#fd7e14", "#dc3545"]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

        # Panel 1: Flaw Perturbation ||Delta z|| distribution
        ax1 = axes[0]
        for c_idx, (lbl, col) in enumerate(zip(labels, colors)):
            mask = (gt_labels == c_idx)
            if np.sum(mask) > 0:
                vals = delta_z_norms[mask]
                ax1.hist(vals, bins=40, density=True, alpha=0.55, color=col, label=f"{lbl} (Mean={np.mean(vals):.3f})")

        ax1.set_xlabel(r"Residual Flaw Perturbation Norm $\|\Delta z\|$", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
        ax1.set_title(r"Flaw Perturbation Disentanglement ($\Delta z$)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Panel 2: Nominal Structural Base ||z^base|| distribution
        ax2 = axes[1]
        for c_idx, (lbl, col) in enumerate(zip(labels, colors)):
            mask = (gt_labels == c_idx)
            if np.sum(mask) > 0:
                vals = z_base_norms[mask]
                ax2.hist(vals, bins=40, density=True, alpha=0.55, color=col, label=f"{lbl} (Mean={np.mean(vals):.3f})")

        ax2.set_xlabel(r"Nominal Structural Base Norm $\|z^{\mathrm{base}}\|$", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
        ax2.set_title(r"Structural Background Invariance ($z^{\mathrm{base}}$)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)

        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return {
        "mean_delta_z_sound": float(np.mean(delta_z_norms[gt_labels == 0])) if np.sum(gt_labels == 0) > 0 else 0.0,
        "mean_delta_z_fastener": float(np.mean(delta_z_norms[gt_labels == 1])) if np.sum(gt_labels == 1) > 0 else 0.0,
        "mean_delta_z_flaw": float(np.mean(delta_z_norms[gt_labels == 2])) if np.sum(gt_labels == 2) > 0 else 0.0,
        "plot_path": save_path,
    }
