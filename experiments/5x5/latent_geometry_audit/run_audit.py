"""
Latent Geometry Audit Engine for PECT-JEPA
Checkpoint: EXP-28 (best_model_5x5.pt)
Specimen: TMR Corrosion Frontside z1 (Square, Chirp, Gauss)

Implements:
- Experiment A: Full 3x3 transfer matrix (Raw, Centered, Standardized, Whitened)
- Experiment B: Defect-effect vectors Delta z_w (Global & Spatial-local), pairwise cosine, per-pixel alignment
- Experiment C: Waveform domain classifier (Raw, Centered, Standardized)
- Experiment D: Linear vs small MLP probe cross-waveform (Square -> Chirp, Square -> Gauss)
- Experiment E: Progressive alignment (Raw -> Centered -> Standardized -> Whitened -> CORAL)
- Experiment F: Physics pulse compression / matched filtering baseline
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy import signal

ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.evaluate import load_model_from_checkpoint
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import (
    extract_full_cscan_map,
    load_cscan_from_tdms,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager

OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", "5x5", "latent_geometry_audit")
CACHE_FILE = os.path.join(OUTPUT_DIR, "extracted_features_cache.npz")


def extract_or_load_data(device="cuda"):
    if os.path.exists(CACHE_FILE):
        print(f"[1/7] Loading cached feature maps from: {CACHE_FILE}")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        data = {
            "feat_maps": {
                "Square": cached["feat_Square"],
                "Chirp": cached["feat_Chirp"],
                "Gauss": cached["feat_Gauss"],
            },
            "gt_mask": cached["gt_mask"],
            "raw_scans": {
                "Square": cached["scan_Square"],
                "Chirp": cached["scan_Chirp"],
                "Gauss": cached["scan_Gauss"],
            }
        }
        return data

    print(f"[1/7] Extracting features from EXP-28 checkpoint on {device}...")
    ckpt_path = os.path.join(ROOT_DIR, "experiments", "5x5", "exp28_full_20ep", "checkpoints", "best_model_5x5.pt")
    model = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()
    cfg = model.config

    gt_mgr = get_ground_truth_manager(data_dir=getattr(cfg, "data_dir", "data"))

    test_files = {
        "Square": r"data\TMR\Corrosion\Square\tmr_corosion_frontside_square_300x300_z1_20260126_190655.tdms",
        "Chirp":  r"data\TMR\Corrosion\Chirp\tmr_corosion_frontside_chirp_300x300_z1_20260128_015728.tdms",
        "Gauss":  r"data\TMR\Corrosion\Gauss\tmr_corosion_frontside_gaussian_300x300_z1_20260127_154450.tdms",
    }

    feat_maps = {}
    raw_scans = {}
    gt_mask_common = None

    for name, rel_path in test_files.items():
        full_path = os.path.join(ROOT_DIR, rel_path)
        print(f"  Extracting C-scan for {name}: {rel_path}...")
        grid_3d = load_cscan_from_tdms(
            full_path,
            time_samples=cfg.time_samples,
            temporal_samples=cfg.temporal_samples,
            resample_mode=cfg.resample_mode,
            normalization=cfg.normalization,
            raster_correction=cfg.raster_correction,
            crop_border=15,
            apply_lowpass=getattr(cfg, "apply_lowpass", True),
            lowpass_cutoff=getattr(cfg, "lowpass_cutoff", 2500.0),
            lowpass_order=getattr(cfg, "lowpass_order", 4),
        )
        raw_scans[name] = grid_3d

        feat_map = extract_full_cscan_map(
            model=model,
            full_cscan_3d=grid_3d,
            batch_size=512,
            device=device,
            show_pbar=False,
            return_volume_3d=False,
        )

        gt = gt_mgr.get_ground_truth_mask_for_file(full_path, aligned_scan=True)
        if gt_mask_common is None:
            min_Y = min(feat_map.shape[0], gt.shape[0])
            min_X = min(feat_map.shape[1], gt.shape[1])
            gt_mask_common = gt[:min_Y, :min_X]
        feat_maps[name] = feat_map[:min_Y, :min_X]

    # Save cache
    np.savez_compressed(
        CACHE_FILE,
        feat_Square=feat_maps["Square"],
        feat_Chirp=feat_maps["Chirp"],
        feat_Gauss=feat_maps["Gauss"],
        gt_mask=gt_mask_common,
        scan_Square=raw_scans["Square"],
        scan_Chirp=raw_scans["Chirp"],
        scan_Gauss=raw_scans["Gauss"],
    )
    print(f"  Saved cache to {CACHE_FILE}")

    return {
        "feat_maps": feat_maps,
        "gt_mask": gt_mask_common,
        "raw_scans": raw_scans,
    }


def compute_whitening(X, eps=1e-5):
    """Zero-phase component analysis (ZCA) whitening."""
    mu = np.mean(X, axis=0, keepdims=True)
    X_c = X - mu
    cov = np.cov(X_c, rowvar=False)
    U, S, Vt = np.linalg.svd(cov)
    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), Vt))
    return mu, W


def apply_whitening(X, mu, W):
    return np.dot(X - mu, W)


def compute_coral_transform(Xs, Xt, reg=1e-4):
    """
    CORAL: Correlation Alignment for Unsupervised Domain Adaptation.
    Maps source covariance to target covariance:
    A = C_s^(-1/2) * C_t^(1/2)
    """
    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + reg * np.eye(Xt.shape[1])

    # SVD for square root and inverse square root
    Us, Ss, Vts = np.linalg.svd(Cs)
    Cs_inv_sqrt = Us @ np.diag(1.0 / np.sqrt(Ss)) @ Vts

    Ut, St, Vtt = np.linalg.svd(Ct)
    Ct_sqrt = Ut @ np.diag(np.sqrt(St)) @ Vtt

    A = Cs_inv_sqrt @ Ct_sqrt
    return A


def run_experiment_a_full_matrix(data):
    """
    Experiment A: Full 3x3 transfer matrix across Square, Gaussian, Chirp.
    Report 4 latent regimes: Raw, Centered, Standardized, Whitened.
    """
    print("\n" + "="*80)
    print("EXPERIMENT A: Full 3x3 Waveform Transfer Matrix (AUC %)")
    print("="*80)

    waveforms = ["Square", "Gauss", "Chirp"]
    gt_mask = data["gt_mask"]
    labels = (gt_mask > 0).astype(int).reshape(-1)
    valid_idx = np.where(labels >= 0)[0]
    y = labels[valid_idx]

    # Subsampled balanced training split (preserve 100% defects, 10x sound background)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    train_idx = np.sort(np.concatenate([pos_idx, sub_neg]))

    results = {}
    regimes = ["Raw", "Centered", "Standardized", "Whitened"]
    for reg in regimes:
        results[reg] = np.zeros((3, 3))

    flat_feats = {}
    for w in waveforms:
        fm = data["feat_maps"][w]
        flat_feats[w] = fm.reshape(-1, fm.shape[-1])[valid_idx]

    for reg in regimes:
        transformed = {}
        if reg == "Raw":
            transformed = {w: flat_feats[w].copy() for w in waveforms}
        elif reg == "Centered":
            transformed = {w: flat_feats[w] - np.median(flat_feats[w], axis=0, keepdims=True) for w in waveforms}
        elif reg == "Standardized":
            transformed = {}
            for w in waveforms:
                sc = StandardScaler()
                transformed[w] = sc.fit_transform(flat_feats[w])
        elif reg == "Whitened":
            transformed = {}
            for w in waveforms:
                mu_w, W_w = compute_whitening(flat_feats[w])
                transformed[w] = apply_whitening(flat_feats[w], mu_w, W_w)

        for i, src in enumerate(waveforms):
            X_tr = transformed[src][train_idx]
            y_tr = y[train_idx]

            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)

            for j, tgt in enumerate(waveforms):
                X_te = transformed[tgt]
                probs = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y, probs) * 100
                results[reg][i, j] = auc

        print(f"\n--- Regime: {reg} ---")
        train_test_label = "Train \\ Test"
        header = f"{train_test_label:<14} | " + " | ".join([f"{w:<10}" for w in waveforms])
        print(header)
        print("-" * len(header))
        for i, src in enumerate(waveforms):
            row_str = f"{src:<14} | " + " | ".join([f"{results[reg][i, j]:>8.2f}%" for j in range(3)])
            print(row_str)

    # Plot transfer matrix heatmap
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=150)
    for idx, reg in enumerate(regimes):
        mat = results[reg]
        im = axes[idx].imshow(mat, cmap="viridis", vmin=40, vmax=100)
        axes[idx].set_xticks(range(3))
        axes[idx].set_yticks(range(3))
        axes[idx].set_xticklabels(waveforms)
        axes[idx].set_yticklabels(waveforms)
        for r_i in range(3):
            for c_j in range(3):
                axes[idx].text(c_j, r_i, f"{mat[r_i, c_j]:.1f}", ha="center", va="center", color="white" if mat[r_i, c_j] < 75 else "black", fontweight="bold")
        axes[idx].set_title(f"Regime: {reg}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Target Test Domain")
        if idx == 0:
            axes[idx].set_ylabel("Source Train Domain")
        if idx == 3:
            fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    plt.suptitle("Experiment A: 3x3 Waveform Transfer Matrix (AUC %)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_a_transfer_matrix.png"))
    plt.close()

    return results


def run_experiment_b_defect_effect_vectors(data):
    """
    Experiment B: Defect-effect vectors (Highest Priority).
    1. Global Delta z_w = E[z|defect, w] - E[z|normal, w]
    2. Spatial-local Delta z_w(p) = z_w(p) - E[z_w(N(p))] (5x5 neighborhood on normal pixels)
    Measure cosine similarity between Delta z of Square <-> Chirp <-> Gaussian.
    """
    print("\n" + "="*80)
    print("EXPERIMENT B: Defect-Effect Vectors Delta z_w (Priority #1)")
    print("="*80)

    waveforms = ["Square", "Gauss", "Chirp"]
    gt_mask = data["gt_mask"]
    sY, sX = gt_mask.shape
    D = data["feat_maps"]["Square"].shape[-1]

    is_defect = (gt_mask == 1)
    is_normal = (gt_mask == 0)

    # 1. Global Defect-Effect Vectors
    delta_global = {}
    for w in waveforms:
        fm = data["feat_maps"][w]
        z_def_mean = np.mean(fm[is_defect], axis=0)
        z_norm_mean = np.mean(fm[is_normal], axis=0)
        delta_global[w] = z_def_mean - z_norm_mean

    # 2. Spatial-local Defect-Effect Vectors Delta z_w(p)
    # Using local ring neighborhood around each defect pixel
    delta_local_pixels = {w: [] for w in waveforms}
    defect_coords = np.argwhere(is_defect)

    for y, x in defect_coords:
        y_min, y_max = max(0, y - 4), min(sY, y + 5)
        x_min, x_max = max(0, x - 4), min(sX, x + 5)

        for w in waveforms:
            fm = data["feat_maps"][w]
            z_p = fm[y, x]
            # Normal pixels within window
            local_norm_mask = is_normal[y_min:y_max, x_min:x_max]
            if np.any(local_norm_mask):
                local_bg = np.mean(fm[y_min:y_max, x_min:x_max][local_norm_mask], axis=0)
            else:
                local_bg = np.mean(fm[is_normal], axis=0)
            delta_local_pixels[w].append(z_p - local_bg)

    delta_local_mean = {}
    for w in waveforms:
        delta_local_pixels[w] = np.array(delta_local_pixels[w])  # [N_defect, D]
        delta_local_mean[w] = np.mean(delta_local_pixels[w], axis=0)

    # Compute Cosine Similarities
    pairs = [("Square", "Chirp"), ("Square", "Gauss"), ("Gauss", "Chirp")]

    print("\n--- 1. Global Defect-Effect Cosine Similarity ---")
    print(f"{'Pair':<20} | {'Cosine Sim':<12} | {'Angle (deg)':<12} | {'||Delta_1||':<12} | {'||Delta_2||':<12}")
    print("-" * 75)
    global_cos = {}
    for w1, w2 in pairs:
        v1, v2 = delta_global[w1], delta_global[w2]
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        ang = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        global_cos[f"{w1}_{w2}"] = float(c)
        print(f"{w1 + ' <-> ' + w2:<20} | {c:>10.4f}  | {ang:>10.1f}° | {np.linalg.norm(v1):>10.4f}   | {np.linalg.norm(v2):>10.4f}")

    print("\n--- 2. Spatial-Local Mean Defect-Effect Cosine Similarity ---")
    print(f"{'Pair':<20} | {'Cosine Sim':<12} | {'Angle (deg)':<12} | {'Mean Pixel Cosine':<18}")
    print("-" * 75)
    local_cos = {}
    for w1, w2 in pairs:
        v1, v2 = delta_local_mean[w1], delta_local_mean[w2]
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        ang = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

        # Per-pixel pairwise cosine
        p1 = delta_local_pixels[w1]
        p2 = delta_local_pixels[w2]
        dots = np.sum(p1 * p2, axis=1)
        norms = (np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1) + 1e-8)
        mean_pixel_cos = np.mean(dots / norms)

        local_cos[f"{w1}_{w2}"] = float(c)
        print(f"{w1 + ' <-> ' + w2:<20} | {c:>10.4f}  | {ang:>10.1f}° | {mean_pixel_cos:>14.4f}")

    # Plot Cosine Matrix & Vectors Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Global Cosine Matrix
    cos_mat = np.zeros((3, 3))
    for i, w1 in enumerate(waveforms):
        for j, w2 in enumerate(waveforms):
            v1, v2 = delta_global[w1], delta_global[w2]
            cos_mat[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    im_cos = axes[0].imshow(cos_mat, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(waveforms)
    axes[0].set_yticklabels(waveforms)
    for r_i in range(3):
        for c_j in range(3):
            axes[0].text(c_j, r_i, f"{cos_mat[r_i, c_j]:.4f}", ha="center", va="center", color="black", fontweight="bold")
    fig.colorbar(im_cos, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Global Defect-Effect Cosine Matrix (Delta z)", fontsize=11, fontweight="bold")

    # Per-pixel cosine distributions
    for w1, w2 in pairs:
        p1 = delta_local_pixels[w1]
        p2 = delta_local_pixels[w2]
        dots = np.sum(p1 * p2, axis=1)
        norms = (np.linalg.norm(p1, axis=1) * np.linalg.norm(p2, axis=1) + 1e-8)
        pixel_cos = dots / norms
        axes[1].hist(pixel_cos, bins=40, alpha=0.5, label=f"{w1} <-> {w2} (mean={np.mean(pixel_cos):.2f})")
    axes[1].set_title("Per-Pixel Local Defect Cosine Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Local Cosine Similarity")
    axes[1].set_ylabel("Pixel Count")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_b_defect_effect_vectors.png"))
    plt.close()

    return {
        "global_cosine": global_cos,
        "local_cosine": local_cos,
        "cos_matrix": cos_mat.tolist(),
    }


def run_experiment_c_waveform_classifier(data):
    """
    Experiment C: Waveform domain classifier.
    Train classifier z -> waveform ID (3 classes: Square=0, Gauss=1, Chirp=2).
    Evaluated on Raw, Centered, and Standardized representations.
    """
    print("\n" + "="*80)
    print("EXPERIMENT C: Waveform Domain Discriminator (z -> Waveform ID)")
    print("="*80)

    waveforms = ["Square", "Gauss", "Chirp"]
    gt_mask = data["gt_mask"]
    labels_snd = np.where(gt_mask.reshape(-1) == 0)[0]  # Sound metal only to avoid defect leakage

    # Sample 5000 points per waveform for domain classification
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(labels_snd, size=5000, replace=False)

    regimes = ["Raw", "Centered", "Standardized"]
    acc_results = {}

    for reg in regimes:
        X_all = []
        y_all = []
        for class_id, w in enumerate(waveforms):
            fm = data["feat_maps"][w]
            flat = fm.reshape(-1, fm.shape[-1])[sample_idx]
            if reg == "Raw":
                x_tf = flat
            elif reg == "Centered":
                x_tf = flat - np.median(flat, axis=0, keepdims=True)
            elif reg == "Standardized":
                sc = StandardScaler()
                x_tf = sc.fit_transform(flat)
            X_all.append(x_tf)
            y_all.append(np.full(len(x_tf), class_id))

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # Train / Test split 70 / 30
        perm = rng.permutation(len(y_all))
        n_tr = int(0.7 * len(y_all))
        tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_all[tr_idx], y_all[tr_idx])
        acc = clf.score(X_all[te_idx], y_all[te_idx]) * 100
        acc_results[reg] = acc
        print(f"Waveform ID Classification Accuracy ({reg:<12}): {acc:6.2f}% (Chance level: 33.33%)")

    return acc_results


def run_experiment_d_linear_vs_mlp_probe(data):
    """
    Experiment D: Linear vs Small MLP Probe on Cross-Waveform.
    Square -> Chirp and Square -> Gauss.
    MLP architectures: D -> 64 -> 1 and D -> 32 -> 32 -> 1.
    """
    print("\n" + "="*80)
    print("EXPERIMENT D: Linear vs Non-Linear MLP Probing Cross-Waveform (AUC %)")
    print("="*80)

    gt_mask = data["gt_mask"]
    labels = (gt_mask > 0).astype(int).reshape(-1)
    valid_idx = np.where(labels >= 0)[0]
    y = labels[valid_idx]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    train_idx = np.sort(np.concatenate([pos_idx, sub_neg]))

    fm_sq = data["feat_maps"]["Square"].reshape(-1, data["feat_maps"]["Square"].shape[-1])[valid_idx]
    fm_ch = data["feat_maps"]["Chirp"].reshape(-1, data["feat_maps"]["Chirp"].shape[-1])[valid_idx]
    fm_ga = data["feat_maps"]["Gauss"].reshape(-1, data["feat_maps"]["Gauss"].shape[-1])[valid_idx]

    # Standardize target-side (as in best self-calibration)
    sc_sq = StandardScaler()
    X_sq = sc_sq.fit_transform(fm_sq)
    sc_ch = StandardScaler()
    X_ch = sc_ch.fit_transform(fm_ch)
    sc_ga = StandardScaler()
    X_ga = sc_ga.fit_transform(fm_ga)

    models = {
        "Linear Probe (LogReg)": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "MLP 1-Hidden (D->64->1)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True),
        "MLP 2-Hidden (D->32->32->1)": MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=42, early_stopping=True),
    }

    results = {}
    for name, clf in models.items():
        print(f"Training {name} on Square...")
        clf.fit(X_sq[train_idx], y[train_idx])

        # Test on Square, Chirp, Gauss
        p_sq = clf.predict_proba(X_sq)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_sq)
        p_ch = clf.predict_proba(X_ch)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_ch)
        p_ga = clf.predict_proba(X_ga)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_ga)

        auc_sq = roc_auc_score(y, p_sq) * 100
        auc_ch = roc_auc_score(y, p_ch) * 100
        auc_ga = roc_auc_score(y, p_ga) * 100

        results[name] = {"Square": auc_sq, "Chirp": auc_ch, "Gauss": auc_ga}
        print(f"  {name:<28} | Within Square: {auc_sq:6.2f}% | X-Chirp: {auc_ch:6.2f}% | X-Gauss: {auc_ga:6.2f}%")

    return results


def run_experiment_e_progressive_alignment(data):
    """
    Experiment E: Progressive Alignment Steps (Square -> Chirp).
    Raw -> Centered -> Standardized -> Whitened -> CORAL
    Record AUC at each step.
    """
    print("\n" + "="*80)
    print("EXPERIMENT E: Progressive Latent Alignment (Square -> Chirp Transfer)")
    print("="*80)

    gt_mask = data["gt_mask"]
    labels = (gt_mask > 0).astype(int).reshape(-1)
    valid_idx = np.where(labels >= 0)[0]
    y = labels[valid_idx]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    train_idx = np.sort(np.concatenate([pos_idx, sub_neg]))

    X_src_raw = data["feat_maps"]["Square"].reshape(-1, data["feat_maps"]["Square"].shape[-1])[valid_idx]
    X_tgt_raw = data["feat_maps"]["Chirp"].reshape(-1, data["feat_maps"]["Chirp"].shape[-1])[valid_idx]

    alignment_steps = [
        ("1. Raw (No alignment)", X_src_raw, X_tgt_raw),
        ("2. Centered (z - median(z))",
         X_src_raw - np.median(X_src_raw, axis=0, keepdims=True),
         X_tgt_raw - np.median(X_tgt_raw, axis=0, keepdims=True)),
        ("3. Standardized (z - mu)/sigma",
         StandardScaler().fit_transform(X_src_raw),
         StandardScaler().fit_transform(X_tgt_raw)),
    ]

    # Whitened
    mu_s, W_s = compute_whitening(X_src_raw)
    mu_t, W_t = compute_whitening(X_tgt_raw)
    X_src_white = apply_whitening(X_src_raw, mu_s, W_s)
    X_tgt_white = apply_whitening(X_tgt_raw, mu_t, W_t)
    alignment_steps.append(("4. Whitened (ZCA decorrelated)", X_src_white, X_tgt_white))

    # CORAL (Second-order moment alignment)
    sc_s = StandardScaler()
    sc_t = StandardScaler()
    Xs_std = sc_s.fit_transform(X_src_raw)
    Xt_std = sc_t.fit_transform(X_tgt_raw)
    A_coral = compute_coral_transform(Xs_std, Xt_std)
    Xs_coral = Xs_std @ A_coral
    alignment_steps.append(("5. CORAL (Covariance Alignment)", Xs_coral, Xt_std))

    results = {}
    print(f"{'Alignment Step':<36} | {'Target Chirp AUC':<18} | {'Delta vs Raw':<14}")
    print("-" * 75)

    base_auc = None
    for step_name, Xs, Xt in alignment_steps:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(Xs[train_idx], y[train_idx])
        probs = clf.predict_proba(Xt)[:, 1]
        auc = roc_auc_score(y, probs) * 100
        if base_auc is None:
            base_auc = auc
        results[step_name] = auc
        print(f"{step_name:<36} | {auc:>16.2f}% | {auc - base_auc:>+12.2f}%")

    return results


def run_experiment_f_physics_canonicalization(data, device="cuda"):
    """
    Experiment F: Physics Canonicalization Baseline.
    Applies pulse compression / matched filtering to Chirp raw scan:
    s_matched(t) = s(t) * r(-t)
    Resamples to canonical time frame and runs through EXP-28 model.
    """
    print("\n" + "="*80)
    print("EXPERIMENT F: Physics Canonicalization via Pulse Compression / Matched Filtering")
    print("="*80)

    # 1. Reference chirp excitation kernel
    # Chirp scan: 500-1500 Hz over time duration
    scan_chirp = data["raw_scans"]["Chirp"]  # [H, W, C]
    H, W, C = scan_chirp.shape

    # Extract spatial median sound metal A-scan as matched filter reference
    gt_mask = data["gt_mask"]
    normal_mask = (gt_mask == 0)
    ref_ascan = np.median(scan_chirp[:gt_mask.shape[0], :gt_mask.shape[1]][normal_mask], axis=0)  # [C]

    # Pulse compression: Cross-correlation with nominal excitation response
    # Yields compact impulsive envelope (peak at zero lag)
    compressed_chirp = np.zeros_like(scan_chirp)
    ref_norm = ref_ascan - np.mean(ref_ascan)
    ref_norm = ref_norm / (np.linalg.norm(ref_norm) + 1e-8)

    for i in range(H):
        for j in range(W):
            sig = scan_chirp[i, j] - np.mean(scan_chirp[i, j])
            corr = signal.correlate(sig, ref_norm, mode="same")
            compressed_chirp[i, j] = corr

    # Re-normalize to match Square dynamic range
    compressed_chirp = (compressed_chirp - np.min(compressed_chirp)) / (np.max(compressed_chirp) - np.min(compressed_chirp) + 1e-8)
    compressed_chirp = (compressed_chirp * 2.0) - 1.0

    print("Running EXP-28 forward pass on pulse-compressed Chirp C-scan...")
    ckpt_path = os.path.join(ROOT_DIR, "experiments", "5x5", "exp28_full_20ep", "checkpoints", "best_model_5x5.pt")
    model = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()

    feat_comp = extract_full_cscan_map(
        model=model,
        full_cscan_3d=compressed_chirp,
        batch_size=512,
        device=device,
        show_pbar=False,
        return_volume_3d=False,
    )

    feat_comp_sub = feat_comp[:gt_mask.shape[0], :gt_mask.shape[1]]
    labels = (gt_mask > 0).astype(int).reshape(-1)
    valid_idx = np.where(labels >= 0)[0]
    y = labels[valid_idx]

    # Subsampled balanced training split
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    train_idx = np.sort(np.concatenate([pos_idx, sub_neg]))

    X_sq_std = StandardScaler().fit_transform(data["feat_maps"]["Square"][:gt_mask.shape[0], :gt_mask.shape[1]].reshape(-1, feat_comp.shape[-1])[valid_idx])
    X_ch_raw_std = StandardScaler().fit_transform(data["feat_maps"]["Chirp"][:gt_mask.shape[0], :gt_mask.shape[1]].reshape(-1, feat_comp.shape[-1])[valid_idx])
    X_ch_comp_std = StandardScaler().fit_transform(feat_comp_sub.reshape(-1, feat_comp.shape[-1])[valid_idx])

    clf_sq = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_sq.fit(X_sq_std[train_idx], y[train_idx])

    auc_raw_ch = roc_auc_score(y, clf_sq.predict_proba(X_ch_raw_std)[:, 1]) * 100
    auc_comp_ch = roc_auc_score(y, clf_sq.predict_proba(X_ch_comp_std)[:, 1]) * 100

    print(f"Square -> Chirp Raw Zero-Shot AUC:             {auc_raw_ch:6.2f}%")
    print(f"Square -> Pulse-Compressed Chirp Zero-Shot AUC: {auc_comp_ch:6.2f}% (Delta: {auc_comp_ch - auc_raw_ch:+5.2f}%)")

    return {
        "raw_chirp_auc": auc_raw_ch,
        "compressed_chirp_auc": auc_comp_ch,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Complete Latent Geometry Audit on {device}...")
    data = extract_or_load_data(device=device)

    res_a = run_experiment_a_full_matrix(data)
    res_b = run_experiment_b_defect_effect_vectors(data)
    res_c = run_experiment_c_waveform_classifier(data)
    res_d = run_experiment_d_linear_vs_mlp_probe(data)
    res_e = run_experiment_e_progressive_alignment(data)
    res_f = run_experiment_f_physics_canonicalization(data, device=device)

    audit_summary = {
        "experiment_a_transfer_matrix": {k: v.tolist() for k, v in res_a.items()},
        "experiment_b_defect_vectors": res_b,
        "experiment_c_domain_classifier_acc": res_c,
        "experiment_d_linear_vs_mlp": res_d,
        "experiment_e_progressive_alignment": res_e,
        "experiment_f_physics_canonicalization": res_f,
    }

    summary_file = os.path.join(OUTPUT_DIR, "latent_geometry_audit_summary.json")
    with open(summary_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nAll experiments complete! Summary successfully saved to: {summary_file}")


if __name__ == "__main__":
    main()
