"""
Visualization utilities for 1D Temporal PECT-JEPA downstream tasks.
Saves high-resolution (300 DPI) publication-ready plots to results/plots/.
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.manifold import TSNE


def plot_waveform_anomaly_map_1d(
    anomaly_scores: np.ndarray,
    grid_shape: tuple,
    save_path: str,
    title: str = "1D Temporal PECT-JEPA Point-Wise Anomaly Map",
    cmap: str = "hot"
):
    """
    Reshapes point-wise 1D anomaly scores into 2D raster scan (e.g. 300x300) and saves heatmap.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    H, W = grid_shape
    if len(anomaly_scores) == H * W:
        amap = anomaly_scores.reshape(H, W)
    else:
        amap = anomaly_scores

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(amap, cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Y (mm)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves_1d(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    title: str = "1D Anomaly Detection Performance"
):
    """
    Save ROC Curve and Precision-Recall Curve for 1D evaluation.
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


def plot_latent_tsne_1d(
    features: np.ndarray,
    labels: List[str],
    save_path: str,
    title: str = "t-SNE of 1D Temporal PECT-JEPA Latents"
):
    """
    Save 2D t-SNE plot of 1D latent representations colored by category.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    max_pts = 4000
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
