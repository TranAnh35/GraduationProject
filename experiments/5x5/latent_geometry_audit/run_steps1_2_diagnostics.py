"""
Diagnostic Follow-up for Latent Geometry Audit:
1. Step 1: Polarity Inversion Diagnostic (AUC_pm = max(AUC, 1 - AUC)) on 3x3 Transfer Matrix.
2. Step 2: Paired Latent Transformation on Sound Metal Baseline (Orthogonal Procrustes & Ridge Regression).
   Can a measurement-specific linear transformation T_{A -> B} bridge the cross-waveform transfer?
"""

import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes

ROOT_DIR = r"E:\Project_On_Lab\Research\GraduationProject"
OUTPUT_DIR = os.path.join(ROOT_DIR, "experiments", "5x5", "latent_geometry_audit")
CACHE_FILE = os.path.join(OUTPUT_DIR, "extracted_features_cache.npz")

def run_diagnostics():
    print(f"Loading cached C-scan features from: {CACHE_FILE}")
    cached = np.load(CACHE_FILE, allow_pickle=True)
    feat_sq = cached["feat_Square"]
    feat_ch = cached["feat_Chirp"]
    feat_ga = cached["feat_Gauss"]
    gt = cached["gt_mask"]

    waveforms = ["Square", "Gauss", "Chirp"]
    feats = {"Square": feat_sq, "Gauss": feat_ga, "Chirp": feat_ch}

    labels = (gt > 0).astype(int).reshape(-1)
    valid_idx = np.where(labels >= 0)[0]
    y = labels[valid_idx]

    # Subsampled balanced training split (preserve 100% defects, 10x sound background)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    train_idx = np.sort(np.concatenate([pos_idx, sub_neg]))

    flat_feats = {w: feats[w].reshape(-1, feats[w].shape[-1])[valid_idx] for w in waveforms}

    # =========================================================================
    # STEP 1: Polarity Inversion Diagnostic (AUC_pm = max(AUC, 1 - AUC))
    # =========================================================================
    print("\n" + "="*85)
    print("STEP 1: POLARITY INVERSION DIAGNOSTIC [AUC vs AUC_pm = max(AUC, 1 - AUC)]")
    print("="*85)

    regimes = ["Raw", "Standardized"]
    step1_results = {}

    for reg in regimes:
        step1_results[reg] = {}
        if reg == "Raw":
            tf_feats = {w: flat_feats[w].copy() for w in waveforms}
        else:
            tf_feats = {w: StandardScaler().fit_transform(flat_feats[w]) for w in waveforms}

        label_col = f"Train \\ Test ({reg})"
        header = f"{label_col:<25} | " + " | ".join([f"{w + ' (AUC / AUC_pm)':<24}" for w in waveforms])
        print(header)
        print("-" * len(header))

        for src in waveforms:
            X_tr, y_tr = tf_feats[src][train_idx], y[train_idx]
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)

            row_entries = []
            for tgt in waveforms:
                X_te = tf_feats[tgt]
                probs = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y, probs) * 100
                auc_pm = max(auc, 100.0 - auc)
                row_entries.append(f"{auc:>6.2f}% -> {auc_pm:>6.2f}%")
                step1_results[reg][f"{src}_{tgt}"] = {"auc": float(auc), "auc_pm": float(auc_pm)}
            print(f"{src:<25} | " + " | ".join(row_entries))
        print()

    # =========================================================================
    # STEP 2: Paired Latent Transformation on Healthy Metal Baseline
    # =========================================================================
    print("="*85)
    print("STEP 2: PAIRED LATENT TRANSFORMATION FITTED ON SOUND METAL BASELINE")
    print("Can a linear transformation T_{tgt -> src} or T_{src -> tgt} bridge the cross-waveform transfer?")
    print("="*85)

    # Coordinated sound metal locations (100% healthy metal, zero defect leakage)
    sound_coords_idx = np.where(y == 0)[0]
    # Sample 10,000 paired healthy points to fit coordinate transform T
    rng = np.random.RandomState(42)
    fit_sound_idx = rng.choice(sound_coords_idx, size=min(10000, len(sound_coords_idx)), replace=False)

    pairs = [
        ("Square", "Chirp"),
        ("Square", "Gauss"),
        ("Gauss", "Chirp"),
    ]

    print(f"{'Transfer Pair (Src -> Tgt)':<28} | {'Standardized Zero-Shot':<22} | {'Orthogonal Procrustes':<22} | {'Ridge Regression (T)':<20}")
    print("-" * 100)

    step2_results = {}

    for src, tgt in pairs:
        # Standardized representations per domain
        sc_src = StandardScaler()
        sc_tgt = StandardScaler()
        Z_src = sc_src.fit_transform(flat_feats[src])
        Z_tgt = sc_tgt.fit_transform(flat_feats[tgt])

        # 1. Baseline Standardized Classifier trained on Source
        clf_src = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf_src.fit(Z_src[train_idx], y[train_idx])
        base_auc = roc_auc_score(y, clf_src.predict_proba(Z_tgt)[:, 1]) * 100

        # 2. Method A: Orthogonal Procrustes fitted on healthy metal
        # Target: R* = argmin_R || Z_tgt_sound @ R - Z_src_sound ||_F s.t. R^T R = I
        Z_tgt_snd = Z_tgt[fit_sound_idx]
        Z_src_snd = Z_src[fit_sound_idx]
        R, sc_scale = orthogonal_procrustes(Z_tgt_snd, Z_src_snd)
        Z_tgt_proc = Z_tgt @ R
        proc_auc = roc_auc_score(y, clf_src.predict_proba(Z_tgt_proc)[:, 1]) * 100

        # 3. Method B: Ridge Linear Regression fitted on healthy metal
        # T* = argmin_T || Z_tgt_sound @ T - Z_src_sound ||^2 + alpha ||T||^2
        ridge = Ridge(alpha=10.0, fit_intercept=False, random_state=42)
        ridge.fit(Z_tgt_snd, Z_src_snd)
        Z_tgt_ridge = ridge.predict(Z_tgt)
        ridge_auc = roc_auc_score(y, clf_src.predict_proba(Z_tgt_ridge)[:, 1]) * 100

        step2_results[f"{src}_{tgt}"] = {
            "baseline_auc": float(base_auc),
            "procrustes_auc": float(proc_auc),
            "ridge_auc": float(ridge_auc),
        }

        print(f"{src + ' -> ' + tgt:<28} | {base_auc:>20.2f}% | {proc_auc:>20.2f}% | {ridge_auc:>18.2f}%")

    out_file = os.path.join(OUTPUT_DIR, "diagnostics_steps1_2_summary.json")
    with open(out_file, "w") as f:
        json.dump({"step1_polarity": step1_results, "step2_paired_transform": step2_results}, f, indent=2)
    print(f"\nStep 1 & Step 2 diagnostics successfully saved to: {out_file}")

if __name__ == "__main__":
    run_diagnostics()
