"""
Visualization utilities for Cross-Condition PECT-JEPA.
Generates side-by-side C-scan heatmaps and detection/sizing evaluation plots.
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cscan_anomaly_map(
    anomaly_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    file_title: str,
    save_path: str,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Plots high-contrast C-scan anomaly heatmap alongside CAD ground-truth mask.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Anomaly map
    im0 = axes[0].imshow(anomaly_map, cmap="inferno", origin="upper")
    axes[0].set_title(f"JEPA Mahalanobis Anomaly Map\n{file_title}", fontsize=11)
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # CAD Ground Truth
    im1 = axes[1].imshow(ground_truth_mask, cmap="Blues", origin="upper", vmin=0, vmax=1)
    title_gt = f"CAD Ground Truth Mask"
    if metrics:
        title_gt += f"\nAUC: {metrics.get('auc_roc', 0.0):.4f} | AP: {metrics.get('average_precision', 0.0):.4f} | CNR: {metrics.get('cnr', 0.0):.2f}"
    axes[1].set_title(title_gt, fontsize=11)
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, list],
    save_path: str,
):
    """
    Plots loss trajectories, SVD Effective Rank, and Latent SNR across epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    epochs = range(1, len(history["total_loss"]) + 1)

    # 1. Losses
    axes[0].plot(epochs, history["total_loss"], "b-o", label="Total Loss")
    axes[0].plot(epochs, history["pred_loss"], "r--", label="Pred L1 Loss")
    axes[0].set_title("Training Loss Trajectory", fontsize=11)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. SVD Effective Rank
    if "effective_rank" in history and len(history["effective_rank"]) > 0:
        axes[1].plot(epochs, history["effective_rank"], "g-s", label="R_eff")
        axes[1].axhline(y=history["effective_rank"][0], color="gray", linestyle=":", label="Initial Rank")
        axes[1].set_title("SVD Effective Rank (Anti-Collapse)", fontsize=11)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Effective Rank (Dimensions)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    # 3. Latent SNR
    if "latent_snr" in history and len(history["latent_snr"]) > 0:
        axes[2].plot(epochs, history["latent_snr"], "m-^", label="Latent SNR")
        axes[2].set_title("Latent Flaw-to-Sound Separability (SNR)", fontsize=11)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("SNR Index")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
