"""
Visualization Suite for Cross-Condition PECT-JEPA Downstream Evaluation.

Provides publication-quality plots:
- Task 1: Anomaly Detection (Defect Probability Heatmaps, ROC & PR Curves).
- Task 2: Segmentation Contours & IoU Masks.
- Task 3: Quantitative Sizing Regression (Predicted vs CAD Ground Truth, Calibration Curves).
- Task 4: Training Loss & Intrinsic Dimension Trajectories.
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_probability_heatmap(
    prob_map: np.ndarray,
    save_path: str,
    title: str = "Defect Probability Heatmap",
    cbar_label: str = "P(Defect | z)",
    cmap: str = "inferno",
    dpi: int = 150,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)

    p_disp = np.clip(prob_map, 0.0, 1.0)
    im = ax.imshow(p_disp, cmap=cmap, aspect="equal", origin="upper", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Scan X (pixels)", fontsize=9)
    ax.set_ylabel("Scan Y (pixels)", fontsize=9)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cscan_anomaly_map(
    anomaly_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    file_title: str,
    save_path: str,
    metrics: Optional[Dict[str, float]] = None,
    dpi: int = 150,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    im0 = axes[0].imshow(anomaly_map, cmap="inferno", origin="upper")
    axes[0].set_title(f"Mahalanobis Distance Map\n{file_title}", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ground_truth_mask, cmap="Blues", origin="upper", vmin=0, vmax=1)
    title_gt = f"CAD Ground Truth Mask"
    if metrics:
        auc = metrics.get('auc_roc', 0.0)
        ap = metrics.get('average_precision', 0.0)
        cnr = metrics.get('cnr', 0.0)
        title_gt += f"\nAUC: {auc:.4f} | AP: {ap:.4f} | CNR: {cnr:.2f}"
    axes[1].set_title(title_gt, fontsize=10, fontweight="bold")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_sizing_calibration(
    true_size: np.ndarray,
    pred_size: np.ndarray,
    save_path: str,
    title: str = "Defect Sizing Calibration (mm)",
    r2: Optional[float] = None,
    mae: Optional[float] = None,
    dpi: int = 150,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=dpi)

    y_t = true_size.flatten()
    y_p = pred_size.flatten()

    ax.scatter(y_t, y_p, alpha=0.4, s=14, color="#2ca02c", edgecolors="none", label="Defect Pixels")
    max_val = max(1.0, float(np.max(y_t)), float(np.max(y_p)))
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.5, label="Ideal (y = x)")

    metrics_str = []
    if r2 is not None:
        metrics_str.append(f"R² = {r2:.4f}")
    if mae is not None:
        metrics_str.append(f"MAE = {mae:.4f} mm")

    if metrics_str:
        ax.text(
            0.05, 0.92, "\n".join(metrics_str),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#ccc")
        )

    ax.set_xlim([0, max_val * 1.05])
    ax.set_ylim([0, max_val * 1.05])
    ax.set_xlabel("Ground Truth Size (mm)", fontsize=10)
    ax.set_ylabel("Predicted Size (mm)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, list],
    save_path: str,
    dpi: int = 150,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=dpi)

    epochs = range(1, len(history["total_loss"]) + 1)

    axes[0].plot(epochs, history["total_loss"], "b-o", label="Total Loss")
    axes[0].plot(epochs, history["pred_loss"], "r--", label="Pred L1 Loss")
    axes[0].set_title("Training Loss Trajectory", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "effective_rank" in history and len(history["effective_rank"]) > 0:
        axes[1].plot(epochs, history["effective_rank"], "g-s", label="R_eff")
        axes[1].set_title("SVD Effective Rank (Anti-Collapse)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Rank (Dimensions)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    if "latent_snr" in history and len(history["latent_snr"]) > 0:
        axes[2].plot(epochs, history["latent_snr"], "m-^", label="Latent SNR")
        axes[2].set_title("Latent Flaw-to-Sound Separability", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("SNR")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_comparison(
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    file_title: str,
    save_path: str,
    metrics: Optional[Dict[str, float]] = None,
    dpi: int = 150,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=dpi)

    im0 = axes[0].imshow(prob_map, cmap="inferno", origin="upper", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Predicted Probability Map\n{file_title}", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_mask, cmap="Greens", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Binarized Segmentation Mask", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(ground_truth_mask, cmap="Blues", origin="upper", vmin=0, vmax=1)
    title_gt = "CAD Ground Truth Mask"
    if metrics:
        iou = metrics.get("iou", 0.0)
        dice = metrics.get("dice", 0.0)
        title_gt += f"\nIoU: {iou:.4f} | Dice: {dice:.4f}"
    axes[2].set_title(title_gt, fontsize=10, fontweight="bold")
    axes[2].set_xlabel("X (pixels)")
    axes[2].set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
