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


def to_safe_path(path: str) -> str:
    """Ensures paths on Windows bypass the MAX_PATH (260 char) limitation using extended prefix."""
    if not path:
        return path
    abs_path = os.path.abspath(path)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def plot_probability_heatmap(
    prob_map: np.ndarray,
    save_path: str,
    title: str = "Defect Probability Heatmap",
    cbar_label: str = "Defect Probability P(Y=1 | z)",
    cmap: str = "jet",
    dpi: int = 150,
) -> None:
    """
    Plots and saves 2D defect probability heatmap with strict [0, 1] color range.
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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

    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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


def apply_spatial_coherence_filter(bin_map: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    Applies spatial physical coherence filtering:
    1. Morphological closing (3x3 kernel) to bridge coil footprint penumbra gaps.
    2. Connected-component labeling to remove isolated noise spikes (< min_area pixels).
    """
    if np.sum(bin_map) == 0 or min_area <= 1:
        return bin_map
    import scipy.ndimage as ndi
    closed = ndi.binary_closing(bin_map, structure=np.ones((3, 3)))
    labeled, num_features = ndi.label(closed)
    if num_features == 0:
        return np.zeros_like(bin_map)
    sizes = ndi.sum(closed, labeled, range(1, num_features + 1))
    cleaned = np.zeros_like(bin_map)
    for i, s in enumerate(sizes, 1):
        if s >= min_area:
            cleaned[labeled == i] = 1
    return cleaned


def plot_defect_contours_and_iou(
    prob_map: np.ndarray,
    gt_mask: np.ndarray,
    save_path: str,
    threshold: Optional[float] = None,
    min_defect_area: int = 8,
    title: str = "Defect Segmentation Bounding Boxes & Mask Comparison",
    dpi: int = 150,
    pixel_pitch_mm: float = 1.0,
) -> Dict[str, Any]:
    """
    Publication-grade 3-Panel Clean Defect Detection & Bounding Box Inspection:
    - Panel 1: Ground Truth CAD Flaws with Green Mask & Precise Bounding Boxes (no C-scan clutter)
    - Panel 2: Predicted Flaws with Magenta Coherent Mask & Bounding Boxes
    - Panel 3: Direct Overlap & Confusion Map (TP: Green, FP: Red, FN: Blue, TN: Clean Neutral)
    Calculates and returns clean IoU, Dice, Precision, Recall, and Sound Metal FPR.
    """
    from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
    import matplotlib.patches as patches
    from matplotlib.patches import Patch
    import scipy.ndimage as ndi

    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sY, sX = prob_map.shape
    p_disp = np.clip(prob_map, 0.0, 1.0)

    # Compute optimal threshold over F1 if not provided
    valid = (gt_mask.reshape(-1) >= 0)
    y_true = (gt_mask.reshape(-1)[valid] == 1).astype(int)
    p_val = p_disp.reshape(-1)[valid]

    if len(np.unique(y_true)) > 1:
        if threshold is not None:
            best_tau = float(threshold)
        else:
            best_dice = -1.0
            best_tau = 0.5
            for tau in np.linspace(0.2, 0.95, 31):
                d = f1_score(y_true, (p_val >= tau).astype(int), zero_division=0)
                if d > best_dice:
                    best_dice = d
                    best_tau = float(tau)

        y_pred_raw = (p_val >= best_tau).astype(int)
        raw_iou = float(jaccard_score(y_true, y_pred_raw, zero_division=0))
        raw_dice = float(f1_score(y_true, y_pred_raw, zero_division=0))
        raw_prec = float(precision_score(y_true, y_pred_raw, zero_division=0))
        raw_rec = float(recall_score(y_true, y_pred_raw, zero_division=0))

        raw_bin_map = (p_disp >= best_tau).astype(int)
        clean_bin_map = apply_spatial_coherence_filter(raw_bin_map, min_area=min_defect_area)

        y_pred_clean = clean_bin_map.reshape(-1)[valid]
        clean_iou = float(jaccard_score(y_true, y_pred_clean, zero_division=0))
        clean_dice = float(f1_score(y_true, y_pred_clean, zero_division=0))
        clean_prec = float(precision_score(y_true, y_pred_clean, zero_division=0))
        clean_rec = float(recall_score(y_true, y_pred_clean, zero_division=0))

        sound_mask = (y_true == 0)
        sound_fpr_raw = float(np.mean(y_pred_raw[sound_mask])) if np.sum(sound_mask) > 0 else 0.0
        sound_fpr_clean = float(np.mean(y_pred_clean[sound_mask])) if np.sum(sound_mask) > 0 else 0.0
    else:
        best_tau = 0.5
        raw_iou, raw_dice, raw_prec, raw_rec = 0.0, 0.0, 0.0, 0.0
        clean_iou, clean_dice, clean_prec, clean_rec = 0.0, 0.0, 0.0, 0.0
        sound_fpr_raw, sound_fpr_clean = 0.0, 0.0
        raw_bin_map = np.zeros_like(p_disp, dtype=int)
        clean_bin_map = np.zeros_like(p_disp, dtype=int)

    gt_core_map = (gt_mask == 1).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)

    # -------------------------------------------------------------
    # Panel 1: Ground Truth CAD Flaws & BBoxes (Clean Canvas)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#f8f9fa")
    # Show GT mask in solid green
    gt_canvas = np.zeros((sY, sX, 4), dtype=np.float32)
    gt_canvas[:, :] = [0.97, 0.98, 0.98, 1.0]  # clean light background
    gt_canvas[gt_core_map == 1] = [0.15, 0.68, 0.38, 0.85]  # emerald green
    ax1.imshow(gt_canvas, origin="lower", aspect="equal")

    # Bounding boxes for Ground Truth clusters
    gt_labeled, n_gt = ndi.label(gt_core_map)
    for i in range(1, n_gt + 1):
        pts = np.where(gt_labeled == i)
        if len(pts[0]) >= min_defect_area:
            y_min, y_max = np.min(pts[0]), np.max(pts[0])
            x_min, x_max = np.min(pts[1]), np.max(pts[1])
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            area_mm2 = len(pts[0]) * (pixel_pitch_mm ** 2)
            d_equiv = 2.0 * np.sqrt(area_mm2 / np.pi)

            rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), w, h, linewidth=1.8, edgecolor="#0e6231", facecolor="none")
            ax1.add_patch(rect)
            ax1.text(x_min, y_max + 2, f"GT #{i}: d={d_equiv:.1f}mm", color="#0e6231", fontsize=7.5, fontweight="bold",
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="#0e6231", pad=1.5))

    ax1.set_title(f"Ground Truth CAD Flaws (N={n_gt})", fontsize=11, fontweight="bold", color="#0e6231")
    ax1.set_xlabel("Scan X (pixels)", fontsize=9)
    ax1.set_ylabel("Scan Y (pixels)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 2: Predicted Detections & BBoxes (Clean Canvas)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fa")
    pred_canvas = np.zeros((sY, sX, 4), dtype=np.float32)
    pred_canvas[:, :] = [0.97, 0.98, 0.98, 1.0]
    pred_canvas[clean_bin_map == 1] = [0.85, 0.10, 0.35, 0.85]  # magenta/coral
    ax2.imshow(pred_canvas, origin="lower", aspect="equal")

    pred_labeled, n_pred = ndi.label(clean_bin_map)
    for j in range(1, n_pred + 1):
        pts = np.where(pred_labeled == j)
        if len(pts[0]) >= min_defect_area:
            y_min, y_max = np.min(pts[0]), np.max(pts[0])
            x_min, x_max = np.min(pts[1]), np.max(pts[1])
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            area_mm2 = len(pts[0]) * (pixel_pitch_mm ** 2)
            d_equiv = 2.0 * np.sqrt(area_mm2 / np.pi)

            rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), w, h, linewidth=1.8, edgecolor="#b30938", facecolor="none", linestyle="--")
            ax2.add_patch(rect)
            ax2.text(x_min, y_max + 2, f"Pred #{j}: d={d_equiv:.1f}mm", color="#b30938", fontsize=7.5, fontweight="bold",
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="#b30938", pad=1.5))

    ax2.set_title(f"Predicted Flaw Clusters (N={n_pred}, tau={best_tau:.2f})", fontsize=11, fontweight="bold", color="#b30938")
    ax2.set_xlabel("Scan X (pixels)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 3: Direct Overlap & Confusion Map
    # -------------------------------------------------------------
    ax3 = axes[2]
    # TP: GT=1 & Pred=1 (Green), FP: GT=0 & Pred=1 (Red), FN: GT=1 & Pred=0 (Blue), TN: GT=0 & Pred=0 (Gray)
    confusion_rgb = np.full((sY, sX, 3), 0.95, dtype=np.float32)  # TN off-white
    confusion_rgb[(gt_core_map == 1) & (clean_bin_map == 1)] = [0.15, 0.68, 0.38]  # TP Emerald
    confusion_rgb[(gt_core_map == 0) & (clean_bin_map == 1)] = [0.90, 0.20, 0.20]  # FP Red
    confusion_rgb[(gt_core_map == 1) & (clean_bin_map == 0)] = [0.00, 0.45, 0.85]  # FN Blue

    ax3.imshow(confusion_rgb, origin="lower", aspect="equal")
    ax3.set_title(f"Overlap Map | IoU: {clean_iou*100:.1f}% | Dice: {clean_dice*100:.1f}%", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Scan X (pixels)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.4)

    legend_elements = [
        Patch(facecolor="#002d12", edgecolor="none", label="True Positive (TP)"),
        Patch(facecolor="#dc3545", edgecolor="none", label="False Positive (FP)"),
        Patch(facecolor="#007bff", edgecolor="none", label="False Negative (FN)"),
        Patch(facecolor="#f0f0f0", edgecolor="#ccc", label="Sound Metal (TN)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"{title}\nIoU: {clean_iou*100:.1f}% | Precision: {clean_prec*100:.1f}% | Recall: {clean_rec*100:.1f}%, Sound Metal FPR: {sound_fpr_clean*100:.2f}%", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "iou": clean_iou,
        "dice": clean_dice,
        "precision": clean_prec,
        "recall": clean_rec,
        "raw_iou": raw_iou,
        "raw_dice": raw_dice,
        "raw_precision": raw_prec,
        "raw_recall": raw_rec,
        "sound_metal_fpr_raw": sound_fpr_raw,
        "sound_metal_fpr_clean": sound_fpr_clean,
        "optimal_threshold": best_tau,
        "min_defect_area": min_defect_area,
        "contour_plot_path": save_path,
        "num_gt_flaws": int(n_gt),
        "num_pred_flaws": int(n_pred),
    }


def plot_flaw_size_maps(
    true_size_map: np.ndarray,
    pred_size_map: np.ndarray,
    save_path: str,
    title: str = "Universal Flaw Sizing Map (Equivalent Diameter mm)",
    dpi: int = 150,
) -> None:
    """
    Plots dual side-by-side 2D spatial maps:
    - Left: Authoritative CAD Ground-Truth Flaw Diameter (mm)
    - Right: PECT-JEPA Predicted Flaw Diameter (mm)
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    max_size = max(10.0, float(np.max(true_size_map)), float(np.max(pred_size_map)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)

    im0 = axes[0].imshow(true_size_map, cmap="plasma", origin="lower", aspect="equal", vmin=0.0, vmax=max_size)
    axes[0].set_title("CAD Ground-Truth Flaw Diameter (mm)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Scan X (pixels)", fontsize=9)
    axes[0].set_ylabel("Scan Y (pixels)", fontsize=9)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Equivalent Diameter (mm)", fontsize=9)

    im1 = axes[1].imshow(pred_size_map, cmap="plasma", origin="lower", aspect="equal", vmin=0.0, vmax=max_size)
    axes[1].set_title("PECT-JEPA Predicted Diameter (mm)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Scan X (pixels)", fontsize=9)
    axes[1].set_ylabel("Scan Y (pixels)", fontsize=9)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Equivalent Diameter (mm)", fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_flaw_size_calibration_scatter(
    true_sizes: np.ndarray,
    pred_sizes: np.ndarray,
    save_path: str,
    title: str = "Flaw Size Calibration Scatter (Diameter mm)",
    r2: Optional[float] = None,
    mae: Optional[float] = None,
    unit: str = "mm",
    dpi: int = 150,
) -> None:
    """
    Calibration scatter plot for cluster-level or pixel-level flaw sizing (Diameter / Area).
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_t = true_sizes.reshape(-1)
    y_p = pred_sizes.reshape(-1)

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=dpi)

    ax.scatter(y_t, y_p, alpha=0.65, s=28, color="#17a2b8", edgecolors="#0f6674", label="Detected Flaws")
    max_val = max(10.0, float(np.max(y_t)) if len(y_t) > 0 else 10.0, float(np.max(y_p)) if len(y_p) > 0 else 10.0)

    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.5, label="Ideal Calibration (y = x)")

    metrics_str = []
    if r2 is not None:
        metrics_str.append(f"Size R² = {r2:.4f}")
    if mae is not None:
        metrics_str.append(f"Size MAE = {mae:.4f} {unit}")

    if metrics_str:
        ax.text(
            0.05, 0.92, "\n".join(metrics_str),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ccc")
        )

    ax.set_xlim([-0.2, max_val * 1.05])
    ax.set_ylim([-0.2, max_val * 1.05])
    ax.set_xlabel(f"True Ground Truth Size ({unit})", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"PECT-JEPA Predicted Size ({unit})", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



