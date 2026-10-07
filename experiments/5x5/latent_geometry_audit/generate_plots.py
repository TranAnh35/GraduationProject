import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", "5x5", "latent_geometry_audit")
CACHE_FILE = os.path.join(OUTPUT_DIR, "extracted_features_cache.npz")

def generate_tsne_and_geometry_plots():
    print("Generating comprehensive diagnostic figures for Latent Geometry Audit...")
    cached = np.load(CACHE_FILE, allow_pickle=True)
    feat_sq = cached["feat_Square"]
    feat_ch = cached["feat_Chirp"]
    feat_ga = cached["feat_Gauss"]
    gt = cached["gt_mask"]

    is_defect = (gt == 1)
    is_normal = (gt == 0)

    # 1. Defect Delta z vectors (Spatial-local)
    defect_coords = np.argwhere(is_defect)
    sY, sX = gt.shape

    delta_pixels = {"Square": [], "Gauss": [], "Chirp": []}
    for y, x in defect_coords:
        y_min, y_max = max(0, y - 4), min(sY, y + 5)
        x_min, x_max = max(0, x - 4), min(sX, x + 5)
        local_norm_mask = is_normal[y_min:y_max, x_min:x_max]

        for w, fm in [("Square", feat_sq), ("Gauss", feat_ga), ("Chirp", feat_ch)]:
            z_p = fm[y, x]
            if np.any(local_norm_mask):
                local_bg = np.mean(fm[y_min:y_max, x_min:x_max][local_norm_mask], axis=0)
            else:
                local_bg = np.mean(fm[is_normal], axis=0)
            delta_pixels[w].append(z_p - local_bg)

    for w in delta_pixels:
        delta_pixels[w] = np.array(delta_pixels[w])

    # Subsample 400 defect points for clean t-SNE
    rng = np.random.RandomState(42)
    idx_sub = rng.choice(len(defect_coords), size=min(400, len(defect_coords)), replace=False)

    X_tsne_in = np.concatenate([
        delta_pixels["Square"][idx_sub],
        delta_pixels["Gauss"][idx_sub],
        delta_pixels["Chirp"][idx_sub],
    ], axis=0)

    # Normalize vectors to unit sphere to inspect direction
    X_tsne_norm = X_tsne_in / (np.linalg.norm(X_tsne_in, axis=-1, keepdims=True) + 1e-8)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(X_tsne_norm)

    n_pts = len(idx_sub)
    sq_emb = emb_tsne[:n_pts]
    ga_emb = emb_tsne[n_pts:2*n_pts]
    ch_emb = emb_tsne[2*n_pts:]

    # PCA 2D projection as well
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(X_tsne_norm)
    sq_pca = emb_pca[:n_pts]
    ga_pca = emb_pca[n_pts:2*n_pts]
    ch_pca = emb_pca[2*n_pts:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Subplot 1: t-SNE
    axes[0].scatter(sq_emb[:, 0], sq_emb[:, 1], c="#e74c3c", label="Square Defect Vector (Delta z)", alpha=0.7, s=25)
    axes[0].scatter(ga_emb[:, 0], ga_emb[:, 1], c="#2ecc71", label="Gaussian Defect Vector (Delta z)", alpha=0.7, s=25)
    axes[0].scatter(ch_emb[:, 0], ch_emb[:, 1], c="#3498db", label="Chirp Defect Vector (Delta z)", alpha=0.7, s=25)
    axes[0].set_title("t-SNE of Unit-Normalized Defect Vectors (Delta z_w)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: PCA
    axes[1].scatter(sq_pca[:, 0], sq_pca[:, 1], c="#e74c3c", label="Square Defect Vector", alpha=0.7, s=25)
    axes[1].scatter(ga_pca[:, 0], ga_pca[:, 1], c="#2ecc71", label="Gaussian Defect Vector", alpha=0.7, s=25)
    axes[1].scatter(ch_pca[:, 0], ch_pca[:, 1], c="#3498db", label="Chirp Defect Vector", alpha=0.7, s=25)
    axes[1].set_title(f"PCA of Defect Vectors (Var Exp: {pca.explained_variance_ratio_[0]*100:.1f}% + {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Latent Geometry Audit: Defect-Effect Manifold Separation Across Waveforms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, "exp_b_defect_vector_manifolds.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"Saved visualization to: {out_fig}")

if __name__ == "__main__":
    generate_tsne_and_geometry_plots()
