"""
Visualization Suite for Cross-Condition PECT-JEPA Downstream Evaluation.

Provides publication-quality plots:
- Task 1: Anomaly Detection (Trained Linear Probe Probability Heatmaps, Mahalanobis Distance Maps) with cmap='jet'.
- Task 2: Quantitative Physical Sizing Calibration Scatter Plots & Error Maps.
- Task 3: 3-Panel Clean Defect Segmentation & Bounding Box Inspection matching spatiotemporal_5x5:
  - Panel 1: CAD Ground Truth Flaws & BBoxes (Clean Canvas)
  - Panel 2: Predicted Flaw Clusters & BBoxes (Clean Canvas)
  - Panel 3: Direct Overlap & Confusion Map (TP Green, FP Red, FN Blue, TN Light Gray)
- Training Loss & Diagnostics Curves.
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import scipy.ndimage as ndi


def to_safe_path(path: str) -> str:
    """Returns a plain absolute path on Windows.

    NOTE: The Win32 extended prefix (``\\\\?\\``) is intentionally NOT used:
    Pillow/matplotlib do not reliably support it (sporadic ``OSError:
    [Errno 22] Invalid argument`` on save, exactly the crash observed in
    long training runs). The prefix is only needed for paths > ~260 chars;
    our experiment paths are ~140 chars, so a plain abspath is safest.
    """
    if not path:
        return path
    abs_path = os.path.abspath(path)
    if os.name == "nt" and abs_path.startswith("\\\\?\\"):
        return abs_path[4:]
    return abs_path


def _safe_savefig(fig, save_path: str, **kwargs) -> None:
    """Saves a matplotlib figure robustly on Windows.

    Uses atomic write (temp file + os.replace) with retries so a transient
    file-lock / AV-scan / cloud-sync hiccup (the sporadic Errno 22 seen in
    long runs) never crashes training. Falls back to a timestamped sibling
    file if the target path stays unwritable.
    """
    save_path = to_safe_path(save_path)
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    base, ext = os.path.splitext(save_path)
    tmp_path = base + ".tmp" + (ext or ".png")
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            fig.savefig(tmp_path, **kwargs)
            os.replace(tmp_path, save_path)
            return
        except OSError as e:
            last_err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            time.sleep(0.5 * (attempt + 1))
    # Final fallback: timestamped sibling so training can continue
    try:
        base, ext = os.path.splitext(save_path)
        fallback = f"{base}_epoch{int(time.time())}{ext or '.png'}"
        fig.savefig(fallback, **kwargs)
        print(f"[WARN] Could not overwrite {save_path} ({last_err}); saved fallback {fallback}", flush=True)
    except Exception as e2:
        print(f"[WARN] Skipping plot save {save_path}: {last_err} (fallback also failed: {e2})", flush=True)


def apply_spatial_coherence_filter(bin_map: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    Applies spatial physical coherence filtering:
    1. Morphological closing (3x3 kernel) to bridge coil footprint penumbra gaps.
    2. Connected-component labeling to remove isolated noise spikes (< min_area pixels).
    """
    if np.sum(bin_map) == 0 or min_area <= 1:
        return bin_map
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


def plot_probability_heatmap(
    prob_map: np.ndarray,
    save_path: str,
    title: str = "Defect Probability Heatmap (Trained Linear Probe)",
    cbar_label: str = "Defect Probability P(Y=1 | z)",
    cmap: str = "jet",
    dpi: int = 150,
) -> None:
    """
    Plots and saves 2D defect probability heatmap with strict [0, 1] color range and jet colormap.
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=dpi)

    p_disp = np.clip(prob_map, 0.0, 1.0)
    im = ax.imshow(p_disp, cmap=cmap, aspect="equal", origin="lower", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Scan X (pixels)", fontsize=9)
    ax.set_ylabel("Scan Y (pixels)", fontsize=9)
    fig.tight_layout()
    _safe_savefig(fig, save_path, bbox_inches="tight")
    plt.close(fig)


def plot_detection_summary_map(
    prob_map: np.ndarray,
    anomaly_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    file_title: str,
    save_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    dpi: int = 150,
) -> None:
    """
    Side-by-side 3-panel figure showing:
    - Panel 1: Trained Linear Probe Probability Map (cmap='jet')
    - Panel 2: Unsupervised Mahalanobis Anomaly Distance Map (cmap='jet')
    - Panel 3: CAD Ground Truth Mask
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=dpi)

    # 1. Trained Linear Probe Probability
    p_disp = np.clip(prob_map, 0.0, 1.0)
    im0 = axes[0].imshow(p_disp, cmap="jet", origin="lower", vmin=0.0, vmax=1.0)
    t0 = "Trained Linear Probe P(Defect | z)"
    if metrics:
        lp_auc = metrics.get("linear_probe_auc", 0.0)
        lp_ap = metrics.get("linear_probe_ap", 0.0)
        t0 += f"\nAUC: {lp_auc:.4f} | AP: {lp_ap:.4f}"
    axes[0].set_title(t0, fontsize=10, fontweight="bold")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Unsupervised Mahalanobis Distance Map
    im1 = axes[1].imshow(anomaly_map, cmap="jet", origin="lower")
    t1 = "Unsupervised Mahalanobis Distance"
    if metrics:
        m_auc = metrics.get("mahalanobis_auc", 0.0)
        cnr = metrics.get("cnr", 0.0)
        t1 += f"\nAUC: {m_auc:.4f} | CNR: {cnr:.2f}"
    axes[1].set_title(t1, fontsize=10, fontweight="bold")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Ground Truth CAD Mask
    im2 = axes[2].imshow(ground_truth_mask, cmap="Blues", origin="lower", vmin=0, vmax=1)
    axes[2].set_title(f"CAD Ground Truth Mask\n{file_title}", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("X (pixels)")
    axes[2].set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    _safe_savefig(fig, save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cscan_anomaly_map(
    anomaly_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    file_title: str,
    save_path: str,
    metrics: Optional[Dict[str, float]] = None,
    dpi: int = 150,
) -> None:
    """
    Plots Unsupervised Mahalanobis Anomaly Distance map with cmap='jet'.
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    im0 = axes[0].imshow(anomaly_map, cmap="jet", origin="lower")
    axes[0].set_title(f"Mahalanobis Distance Map\n{file_title}", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ground_truth_mask, cmap="Blues", origin="lower", vmin=0, vmax=1)
    title_gt = "CAD Ground Truth Mask"
    if metrics:
        auc = metrics.get("auc_roc", metrics.get("mahalanobis_auc", 0.0))
        ap = metrics.get("average_precision", metrics.get("mahalanobis_ap", 0.0))
        cnr = metrics.get("cnr", 0.0)
        title_gt += f"\nAUC: {auc:.4f} | AP: {ap:.4f} | CNR: {cnr:.2f}"
    axes[1].set_title(title_gt, fontsize=10, fontweight="bold")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    _safe_savefig(fig, save_path, bbox_inches="tight")
    plt.close(fig)


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
    Publication-grade 3-Panel Clean Defect Detection & Bounding Box Inspection matching spatiotemporal_5x5:
    - Panel 1: Ground Truth CAD Flaws with Green Mask & Precise Bounding Boxes (no C-scan clutter)
    - Panel 2: Predicted Flaws with Magenta Coherent Mask & Bounding Boxes
    - Panel 3: Direct Overlap & Confusion Map (TP: Green, FP: Red, FN: Blue, TN: Clean Neutral)
    Calculates and returns clean IoU, Dice, Precision, Recall, and Sound Metal FPR.
    """
    from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

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
    gt_canvas = np.zeros((sY, sX, 4), dtype=np.float32)
    gt_canvas[:, :] = [0.97, 0.98, 0.98, 1.0]
    gt_canvas[gt_core_map == 1] = [0.15, 0.68, 0.38, 0.85]  # emerald green
    ax1.imshow(gt_canvas, origin="lower", aspect="equal")

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
    ax2.set_ylabel("Scan Y (pixels)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.4)

    # -------------------------------------------------------------
    # Panel 3: Direct Overlap & Confusion Map
    # -------------------------------------------------------------
    ax3 = axes[2]
    confusion_rgb = np.full((sY, sX, 3), 0.95, dtype=np.float32)  # TN off-white
    confusion_rgb[(gt_core_map == 1) & (clean_bin_map == 1)] = [0.15, 0.68, 0.38]  # TP Emerald
    confusion_rgb[(gt_core_map == 0) & (clean_bin_map == 1)] = [0.90, 0.20, 0.20]  # FP Red
    confusion_rgb[(gt_core_map == 1) & (clean_bin_map == 0)] = [0.00, 0.45, 0.85]  # FN Blue

    ax3.imshow(confusion_rgb, origin="lower", aspect="equal")
    ax3.set_title(f"Overlap Map | IoU: {clean_iou*100:.1f}% | Dice: {clean_dice*100:.1f}%", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Scan X (pixels)", fontsize=9)
    ax3.set_ylabel("Scan Y (pixels)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.4)

    legend_elements = [
        Patch(facecolor="#27ae60", edgecolor="none", label="True Positive (TP)"),
        Patch(facecolor="#e53935", edgecolor="none", label="False Positive (FP)"),
        Patch(facecolor="#1e88e5", edgecolor="none", label="False Negative (FN)"),
        Patch(facecolor="#f0f0f0", edgecolor="#ccc", label="Sound Metal (TN)"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"{title}\nIoU: {clean_iou*100:.1f}% | Precision: {clean_prec*100:.1f}% | Recall: {clean_rec*100:.1f}%, Sound Metal FPR: {sound_fpr_clean*100:.2f}%", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout()
    _safe_savefig(fig, save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "iou": clean_iou,
        "dice": clean_dice,
        "precision": clean_prec,
        "recall": clean_rec,
        "raw_iou": raw_iou,
        "raw_dice": raw_dice,
        "optimal_threshold": best_tau,
        "sound_metal_fpr": sound_fpr_clean,
    }


def plot_sizing_calibration(
    true_size: np.ndarray,
    pred_size: np.ndarray,
    save_path: str,
    title: str = "Defect Sizing Calibration (mm)",
    r2: Optional[float] = None,
    mae: Optional[float] = None,
    dpi: int = 150,
) -> None:
    """
    Plots scatter plot of True CAD Size (mm) vs Predicted Size (mm).
    """
    save_path = to_safe_path(save_path)
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
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )

    ax.set_xlim([0, max_val * 1.05])
    ax.set_ylim([0, max_val * 1.05])
    ax.set_xlabel("Ground Truth Size (mm)", fontsize=10)
    ax.set_ylabel("Predicted Size (mm)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    _safe_savefig(fig, save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, list],
    save_path: str,
    dpi: int = 150,
) -> None:
    """
    Plots training loss trajectory, effective rank, validation AUC/AP, and latent SNR.
    """
    save_path = to_safe_path(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=dpi)

    epochs = range(1, len(history["total_loss"]) + 1)

    # Panel 1: Loss (train vs val reconstruction)
    axes[0].plot(epochs, history["total_loss"], "b-o", label="Train Total Loss", markersize=3)
    axes[0].plot(epochs, history["pred_loss"], "r--", label="Train Pred L1 Loss")
    if "val_loss" in history and len(history["val_loss"]) == len(history["total_loss"]) and any(v > 0 for v in history["val_loss"]):
        axes[0].plot(epochs, history["val_loss"], "g-s", label="Val Recon Loss", markersize=3)
    axes[0].set_title("Training / Validation Loss Trajectory", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel 2: Effective Rank
    if "effective_rank" in history and len(history["effective_rank"]) > 0:
        axes[1].plot(epochs, history["effective_rank"], "g-s", label="R_eff", markersize=3)
        axes[1].set_title("SVD Effective Rank", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Rank (Dimensions)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right", fontsize=8)

    # Panel 3: Validation AUC & AP
    if "val_auc" in history and len(history["val_auc"]) > 0:
        axes[2].plot(epochs, history["val_auc"], "b-^", label="Val AUC", markersize=3)
        if "val_ap" in history and len(history["val_ap"]) > 0:
            axes[2].plot(epochs, history["val_ap"], "c--", label="Val AP")
        axes[2].set_title("Validation Anomaly Detection", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC / AP")
        axes[2].set_ylim([0.4, 1.02])
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="lower right", fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "No Validation Telemetry", ha="center", va="center", transform=axes[2].transAxes)

    # Panel 4: Latent SNR & Sound Var
    snr_data = history.get("val_snr", history.get("latent_snr", []))
    if len(snr_data) > 0:
        ax4_snr = axes[3]
        line1 = ax4_snr.plot(epochs, snr_data, "m-d", label="Latent SNR", markersize=3)
        ax4_snr.set_title("Latent SNR & Background Variance", fontsize=11, fontweight="bold")
        ax4_snr.set_xlabel("Epoch")
        ax4_snr.set_ylabel("SNR (Flaw / Noise)", color="purple")
        ax4_snr.tick_params(axis="y", labelcolor="purple")
        ax4_snr.grid(True, alpha=0.3)

        if "val_sound_var" in history and len(history["val_sound_var"]) > 0:
            ax4_var = ax4_snr.twinx()
            line2 = ax4_var.plot(epochs, history["val_sound_var"], "k:", label="Sound Var")
            ax4_var.set_ylabel("Sound Metal Variance", color="black")
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4_snr.legend(lines, labels, loc="upper left", fontsize=8)
        else:
            ax4_snr.legend(loc="upper left", fontsize=8)
    else:
        axes[3].text(0.5, 0.5, "No Latent SNR Data", ha="center", va="center", transform=axes[3].transAxes)

    plt.tight_layout()
    _safe_savefig(fig, save_path, bbox_inches="tight")
    plt.close(fig)
