"""
Visualization utilities for PECT-JEPA downstream tasks.
Saves high-resolution (300 DPI) publication-ready plots to results/plots/.
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.manifold import TSNE


def plot_anomaly_heatmap_3d(
    raw_scan: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: str,
    title: str = "PECT-JEPA Anomaly Detection",
    cmap_raw: str = "viridis",
    cmap_map: str = "hot"
):
    """
    Save side-by-side plot of Raw C-Scan vs JEPA Dense Anomaly Map (300x300).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Raw C-Scan (Peak/RMS across time)
    if raw_scan.ndim == 3:
        raw_2d = np.max(raw_scan, axis=-1)
    else:
        raw_2d = raw_scan

    im0 = axes[0].imshow(raw_2d, cmap=cmap_raw, origin="lower")
    axes[0].set_title("Raw PECT C-Scan", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. JEPA Anomaly Map
    im1 = axes[1].imshow(anomaly_map, cmap=cmap_map, origin="lower")
    axes[1].set_title("JEPA Dense Anomaly Score (300x300)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Y (mm)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str = "Anomaly Detection Performance"
):
    """
    Save ROC Curve and Precision-Recall Curve.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # ROC
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # PR
    axes[1].plot(recall, precision, color="teal", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_latent_tsne(
    features: np.ndarray,
    labels: List[str],
    save_path: str,
    title: str = "t-SNE of PECT-JEPA Latent Space"
):
    """
    Save 2D t-SNE plot of latent representations colored by category.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Subsample if too large
    max_pts = 3000
    if len(features) > max_pts:
        idx = np.random.choice(len(features), max_pts, replace=False)
        features = features[idx]
        labels = [labels[i] for i in idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(features)

    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i, lbl in enumerate(unique_labels):
        mask = [l == lbl for l in labels]
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[colors[i]], label=lbl, alpha=0.7, s=20)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(title="Domains", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
