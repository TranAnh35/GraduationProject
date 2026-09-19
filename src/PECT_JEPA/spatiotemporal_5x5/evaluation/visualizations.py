"""
Visualization Suite for 5x5 PECT-JEPA Downstream Evaluation.

Provides publication-quality plots for all 5 NDT downstream benchmark tasks:
- Task 1: Anomaly Detection (Defect Probability Heatmaps, ROC & PR Curves)
- Task 2: Quantitative Depth Regression (CAD vs Predicted Depth Maps, Calibration Scatter Plots)
- Task 3: Severity Classification (Normalized Confusion Matrices)
- Task 4: Lift-off Invariance (Linear CKA Heatmap Matrix)
- Task 5: Representation Geometry (PCA-RGB, Angular Distance, Intrinsic Dimension)
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_probability_heatmap(
    prob_map: np.ndarray,
    save_path: str,
    title: str = "Defect Probability Heatmap",
    cbar_label: str = "Defect Probability P(Y=1 | z)",
    cmap: str = "inferno",
    dpi: int = 150,
) -> None:
    """
    Plots and saves 2D defect probability heatmap with strict [0, 1] color range.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)
    
    # Clip probabilities to [0, 1] for safety
    p_disp = np.clip(prob_map, 0.0, 1.0)
    vmax = max(0.5, float(np.percentile(p_disp, 99.9)))
    
    im = ax.imshow(p_disp, cmap=cmap, aspect="equal", origin="lower", vmin=0.0, vmax=min(1.0, vmax))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Scan X (pixels)", fontsize=9)
    ax.set_ylabel("Scan Y (pixels)", fontsize=9)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    fpr: List[float],
    tpr: List[float],
    auc_roc: float,
    precision: List[float],
    recall: List[float],
    avg_prec: float,
    save_path: str,
    title: str = "ROC and Precision-Recall Curves",
    dpi: int = 150,
) -> None:
    """
    Plots 2-panel figure showing ROC Curve (left) and PR Curve (right).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=dpi)

    # Panel 1: ROC Curve
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc_roc:.4f})")
    ax1.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--", label="Random Chance (0.50)")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Precision-Recall Curve
    ax2.plot(recall, precision, color="#d62728", lw=2, label=f"PR Curve (AP = {avg_prec:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Defect Detection Rate)", fontsize=10)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax2.set_title("Precision-Recall Curve (PR)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_depth_regression_maps(
    true_depth_map: np.ndarray,
    pred_depth_map: np.ndarray,
    save_path: str,
    title: str = "Defect Depth Sizing Regression (mm)",
    r2: Optional[float] = None,
    mae: Optional[float] = None,
    rmse: Optional[float] = None,
    dpi: int = 150,
) -> None:
    """
    Plots side-by-side comparison of True CAD Depth Map vs Model Predicted Depth Map (mm).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    vmax = max(1.0, float(np.max(true_depth_map)), float(np.max(pred_depth_map)))
    
    im1 = ax1.imshow(true_depth_map, cmap="plasma", origin="lower", vmin=0.0, vmax=vmax)
    ax1.set_title("CAD Ground Truth Depth (mm)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Scan X (pixels)", fontsize=9)
    ax1.set_ylabel("Scan Y (pixels)", fontsize=9)

    pred_clipped = np.clip(pred_depth_map, 0.0, vmax)
    im2 = ax2.imshow(pred_clipped, cmap="plasma", origin="lower", vmin=0.0, vmax=vmax)
    sub_title = "PECT-JEPA Predicted Depth (mm)"
    if r2 is not None and mae is not None:
        sub_title += f"\nR² = {r2:.3f} | MAE = {mae:.3f} mm | RMSE = {rmse:.3f} mm" if rmse is not None else f"\nR² = {r2:.3f} | MAE = {mae:.3f} mm"
    ax2.set_title(sub_title, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Scan X (pixels)", fontsize=9)
    ax2.set_ylabel("Scan Y (pixels)", fontsize=9)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.035, pad=0.04)
    cbar.set_label("Physical Depth (mm)", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_depth_calibration_scatter(
    true_depth: np.ndarray,
    pred_depth: np.ndarray,
    save_path: str,
    title: str = "Depth Calibration Curve (True vs Predicted)",
    r2: Optional[float] = None,
    mae: Optional[float] = None,
    rmse: Optional[float] = None,
    dpi: int = 150,
) -> None:
    """
    Plots scatter plot of True CAD Depth (mm) vs Model Predicted Depth (mm).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=dpi)

    y_t = true_depth.flatten()
    y_p = pred_depth.flatten()

    # Subsample sound background to prevent scatter plot from being completely dominated by (0, 0)
    def_idx = np.where(y_t > 0.0)[0]
    snd_idx = np.where(y_t == 0.0)[0]
    if len(snd_idx) > 2000:
        rng = np.random.RandomState(42)
        sub_snd = rng.choice(snd_idx, size=2000, replace=False)
        plot_idx = np.concatenate([def_idx, sub_snd])
    else:
        plot_idx = np.arange(len(y_t))

    ax.scatter(y_t[plot_idx], y_p[plot_idx], alpha=0.35, s=12, color="#2ca02c", edgecolors="none", label="Sampled Points")

    max_val = max(1.0, float(np.max(y_t)), float(np.max(y_p)))
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.5, label="Ideal Calibration (y = x)")

    metrics_str = []
    if r2 is not None:
        metrics_str.append(f"R² = {r2:.4f}")
    if mae is not None:
        metrics_str.append(f"MAE = {mae:.4f} mm")
    if rmse is not None:
        metrics_str.append(f"RMSE = {rmse:.4f} mm")

    if metrics_str:
        ax.text(
            0.05, 0.92, "\n".join(metrics_str),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#ccc")
        )

    ax.set_xlim([-0.05, max_val * 1.05])
    ax.set_ylim([-0.05, max_val * 1.05])
    ax.set_xlabel("Ground Truth CAD Depth (mm)", fontsize=10)
    ax.set_ylabel("PECT-JEPA Predicted Depth (mm)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_severity_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    title: str = "Defect Severity Classification Confusion Matrix",
    macro_f1: Optional[float] = None,
    accuracy: Optional[float] = None,
    dpi: int = 150,
) -> None:
    """
    Plots normalized confusion matrix heatmap for defect severity bins.
    """
    if class_names is None:
        class_names = ["Sound", "Shallow (<=0.2)", "Medium (0.3-0.6)", "Deep (>=0.7)"]

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}\n({count})", ha="center", va="center", color=color, fontsize=8.5)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlabel("Predicted Severity", fontsize=10)
    ax.set_ylabel("True Ground Truth Severity", fontsize=10)

    header = title
    if macro_f1 is not None and accuracy is not None:
        header += f"\nMacro F1: {macro_f1:.4f} | Accuracy: {accuracy:.1%}"
    ax.set_title(header, fontsize=11, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Class Accuracy", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_liftoff_cka_heatmap(
    cka_matrix: np.ndarray,
    labels: List[str],
    save_path: str,
    title: str = "Linear CKA Multi-Lift-Off Invariance Matrix",
    dpi: int = 150,
) -> None:
    """
    Plots heatmap matrix of Linear CKA invariance across lift-off heights.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)
    
    im = ax.imshow(cka_matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    n = len(labels)

    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA Score", fontsize=10)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
