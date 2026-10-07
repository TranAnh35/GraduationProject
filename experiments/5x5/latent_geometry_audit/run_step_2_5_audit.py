"""
Step 2.5: Transform Generalization & Cycle Consistency Audit
Evaluates:
1. Cross-File / Cross-Scan Transform Generalization:
   Fit T^{(Scan A)}_{G -> C} on healthy points of Scan A (e.g., z1 or Half-Scan A).
   Apply directly to Scan B without refitting -> Measure target Chirp defect AUC.
   Does the transform generalize across files (z1 -> z2) or cross-halves?
2. Defect-Vector Realignment:
   Compute cos(Delta z_G @ R_{G -> C}, Delta z_C).
   Does orthogonal Procrustes directly align the defect perturbation vectors?
3. Cycle Consistency:
   Compute || R_{G -> C} @ R_{C -> G} - I ||_F and || T_{G -> C} @ T_{C -> G} - I ||_F.
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes

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
CACHE_FILE_Z1 = os.path.join(OUTPUT_DIR, "extracted_features_cache.npz")
CACHE_FILE_Z2 = os.path.join(OUTPUT_DIR, "extracted_features_cache_z2.npz")


def get_z2_data(device="cuda"):
    if os.path.exists(CACHE_FILE_Z2):
        print(f"Loading cached z2 features from: {CACHE_FILE_Z2}")
        cached = np.load(CACHE_FILE_Z2, allow_pickle=True)
        return {
            "Square": cached["feat_Square"],
            "Gauss": cached["feat_Gauss"],
            "Chirp": cached["feat_Chirp"],
            "gt_mask": cached["gt_mask"]
        }

    print("Extracting z2 C-scan features from EXP-28 checkpoint...")
    ckpt_path = os.path.join(ROOT_DIR, "experiments", "5x5", "exp28_full_20ep", "checkpoints", "best_model_5x5.pt")
    model = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()
    cfg = model.config

    gt_mgr = get_ground_truth_manager(data_dir=getattr(cfg, "data_dir", "data"))

    files_z2 = {
        "Square": r"data\TMR\Corrosion\Square\tmr_corosion_frontside_square_300x300_z2_20260127_085536.tdms",
        "Gauss":  r"data\TMR\Corrosion\Gauss\tmr_corosion_frontside_gaussian_300x300_z2_20260127_190822.tdms",
        "Chirp":  r"data\TMR\Corrosion\Chirp\tmr_corosion_frontside_chirp_300x300_z2_20260128_052059.tdms",
    }

    feat_maps = {}
    gt_mask_common = None

    for name, rel_path in files_z2.items():
        full_path = os.path.join(ROOT_DIR, rel_path)
        print(f"  Extracting z2 {name}: {rel_path}...")
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

    np.savez_compressed(
        CACHE_FILE_Z2,
        feat_Square=feat_maps["Square"],
        feat_Gauss=feat_maps["Gauss"],
        feat_Chirp=feat_maps["Chirp"],
        gt_mask=gt_mask_common,
    )
    print(f"Saved z2 cache to: {CACHE_FILE_Z2}")
    return feat_maps, gt_mask_common


def run_step_2_5_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*85)
    print("STEP 2.5: TRANSFORM GENERALIZATION & CYCLE CONSISTENCY AUDIT")
    print("="*85)

    # 1. Load z1 data
    cached_z1 = np.load(CACHE_FILE_Z1, allow_pickle=True)
    feats_z1 = {
        "Square": cached_z1["feat_Square"],
        "Gauss": cached_z1["feat_Gauss"],
        "Chirp": cached_z1["feat_Chirp"],
    }
    gt_z1 = cached_z1["gt_mask"]

    # 2. Load or extract z2 data
    res_z2 = get_z2_data(device=device)
    if isinstance(res_z2, tuple):
        feats_z2, gt_z2 = res_z2
    else:
        feats_z2 = {"Square": res_z2["Square"], "Gauss": res_z2["Gauss"], "Chirp": res_z2["Chirp"]}
        gt_z2 = res_z2["gt_mask"]

    # Labels and indices
    labels_z1 = (gt_z1 > 0).astype(int).reshape(-1)
    valid_z1 = np.where(labels_z1 >= 0)[0]
    y_z1 = labels_z1[valid_z1]

    labels_z2 = (gt_z2 > 0).astype(int).reshape(-1)
    valid_z2 = np.where(labels_z2 >= 0)[0]
    y_z2 = labels_z2[valid_z2]

    # Subsampled balanced training split for Classifier
    pos_z1 = np.where(y_z1 == 1)[0]
    neg_z1 = np.where(y_z1 == 0)[0]
    rng = np.random.RandomState(42)
    sub_neg_z1 = rng.choice(neg_z1, size=min(len(neg_z1), len(pos_z1) * 10), replace=False)
    train_idx_z1 = np.sort(np.concatenate([pos_z1, sub_neg_z1]))

    # Flatten and Standardize
    flat_z1 = {w: feats_z1[w].reshape(-1, feats_z1[w].shape[-1])[valid_z1] for w in feats_z1}
    flat_z2 = {w: feats_z2[w].reshape(-1, feats_z2[w].shape[-1])[valid_z2] for w in feats_z2}

    sc_z1 = {w: StandardScaler() for w in feats_z1}
    Z1 = {w: sc_z1[w].fit_transform(flat_z1[w]) for w in feats_z1}

    sc_z2 = {w: StandardScaler() for w in feats_z2}
    Z2 = {w: sc_z2[w].fit_transform(flat_z2[w]) for w in feats_z2}

    # =========================================================================
    # PART 1: Defect-Vector Realignment Under Procrustes
    # cos(Delta z_G @ R_{G -> C}, Delta z_C)
    # =========================================================================
    print("\n--- 1. Defect-Vector Realignment Under Orthogonal Transformation ---")
    snd_idx_z1 = np.where(y_z1 == 0)[0]
    fit_snd_z1 = rng.choice(snd_idx_z1, size=10000, replace=False)

    # Learn R_{C -> G}: Z_C @ R approx Z_G (maps Chirp coordinates to Gauss)
    R_C2G, _ = orthogonal_procrustes(Z1["Chirp"][fit_snd_z1], Z1["Gauss"][fit_snd_z1])
    # Learn R_{G -> C}: Z_G @ R approx Z_C (maps Gauss coordinates to Chirp)
    R_G2C, _ = orthogonal_procrustes(Z1["Gauss"][fit_snd_z1], Z1["Chirp"][fit_snd_z1])

    # Defect effect vectors on z1
    is_def_z1 = (y_z1 == 1)
    is_snd_z1 = (y_z1 == 0)

    delta_G = np.mean(Z1["Gauss"][is_def_z1], axis=0) - np.mean(Z1["Gauss"][is_snd_z1], axis=0)
    delta_C = np.mean(Z1["Chirp"][is_def_z1], axis=0) - np.mean(Z1["Chirp"][is_snd_z1], axis=0)
    delta_S = np.mean(Z1["Square"][is_def_z1], axis=0) - np.mean(Z1["Square"][is_snd_z1], axis=0)

    cos_raw_GC = np.dot(delta_G, delta_C) / (np.linalg.norm(delta_G) * np.linalg.norm(delta_C) + 1e-8)

    # Realigned defect vectors
    delta_C_mapped = delta_C @ R_C2G  # Chirp defect vector mapped into Gauss frame
    cos_realigned_GC = np.dot(delta_C_mapped, delta_G) / (np.linalg.norm(delta_C_mapped) * np.linalg.norm(delta_G) + 1e-8)

    print(f"Raw Cosine (Delta z_G, Delta z_C):              {cos_raw_GC:>8.4f} (Angle: {np.degrees(np.arccos(np.clip(cos_raw_GC, -1, 1))):.1f}°)")
    print(f"Realigned Cosine (Delta z_C @ R_{{C->G}}, Delta z_G): {cos_realigned_GC:>8.4f} (Angle: {np.degrees(np.arccos(np.clip(cos_realigned_GC, -1, 1))):.1f}°)")
    print(f"Cosine Gain: {cos_realigned_GC - cos_raw_GC:>+8.4f}")

    # =========================================================================
    # PART 2: Cycle Consistency of Coordinate Transformations
    # =========================================================================
    print("\n--- 2. Cycle Consistency Diagnostics ---")
    D = Z1["Gauss"].shape[-1]
    I_D = np.eye(D)

    # Procrustes Cycle Error: || R_{G->C} @ R_{C->G} - I ||_F / sqrt(D)
    cycle_R = R_G2C @ R_C2G
    cycle_err_procrustes = np.linalg.norm(cycle_R - I_D, "fro") / np.sqrt(D)

    # Ridge Regression Transformations
    ridge_C2G = Ridge(alpha=10.0, fit_intercept=False, random_state=42).fit(Z1["Chirp"][fit_snd_z1], Z1["Gauss"][fit_snd_z1])
    ridge_G2C = Ridge(alpha=10.0, fit_intercept=False, random_state=42).fit(Z1["Gauss"][fit_snd_z1], Z1["Chirp"][fit_snd_z1])
    T_C2G = ridge_C2G.coef_.T
    T_G2C = ridge_G2C.coef_.T

    cycle_T = T_G2C @ T_C2G
    cycle_err_ridge = np.linalg.norm(cycle_T - I_D, "fro") / np.sqrt(D)

    print(f"Procrustes Cycle Inversion Error (||R_{{G->C}} R_{{C->G}} - I||_F / sqrt(D)): {cycle_err_procrustes:8.4f}")
    print(f"Ridge Cycle Inversion Error      (||T_{{G->C}} T_{{C->G}} - I||_F / sqrt(D)): {cycle_err_ridge:8.4f}")

    # =========================================================================
    # PART 3: Cross-File / Cross-Scan Transform Generalization Audit
    # Fit T on z1 healthy metal -> Apply strictly to z2 test data!
    # =========================================================================
    print("\n--- 3. Cross-Scan Transform Generalization Audit (T fitted on z1 -> Tested on z2) ---")
    # Train Classifier on Gauss z1
    clf_G1 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_G1.fit(Z1["Gauss"][train_idx_z1], y_z1[train_idx_z1])

    # Baseline 1: In-domain Gauss z1 -> Gauss z2
    auc_G1_to_G2 = roc_auc_score(y_z2, clf_G1.predict_proba(Z2["Gauss"])[:, 1]) * 100

    # Baseline 2: Cross-waveform z1 -> z1 without transform
    auc_G1_to_C1_raw = roc_auc_score(y_z1, clf_G1.predict_proba(Z1["Chirp"])[:, 1]) * 100
    # In-Scan Transformed (Transductive): T fitted on z1 tested on z1
    auc_G1_to_C1_proc = roc_auc_score(y_z1, clf_G1.predict_proba(Z1["Chirp"] @ R_C2G)[:, 1]) * 100

    # Strict Cross-Scan Transfer:
    # 1. Raw z1 Gauss -> z2 Chirp (no transform)
    auc_G1_to_C2_raw = roc_auc_score(y_z2, clf_G1.predict_proba(Z2["Chirp"])[:, 1]) * 100
    # 2. Generalization of z1 Transform: Apply R_{C->G}^{(z1)} directly to Z2 Chirp!
    Z2_C_proc_gen = Z2["Chirp"] @ R_C2G
    auc_G1_to_C2_proc_gen = roc_auc_score(y_z2, clf_G1.predict_proba(Z2_C_proc_gen)[:, 1]) * 100
    # 3. Target-refitted z2 transform (Transductive z2 baseline)
    snd_idx_z2 = np.where(y_z2 == 0)[0]
    fit_snd_z2 = rng.choice(snd_idx_z2, size=10000, replace=False)
    R_C2G_z2, _ = orthogonal_procrustes(Z2["Chirp"][fit_snd_z2], Z2["Gauss"][fit_snd_z2])
    auc_G1_to_C2_proc_refit = roc_auc_score(y_z2, clf_G1.predict_proba(Z2["Chirp"] @ R_C2G_z2)[:, 1]) * 100

    print(f"Gauss z1 -> Gauss z2 (In-Domain Baseline):            {auc_G1_to_G2:>6.2f}%")
    print(f"Gauss z1 -> Chirp z1 (Raw, no transform):             {auc_G1_to_C1_raw:>6.2f}%")
    print(f"Gauss z1 -> Chirp z1 (Procrustes on z1):              {auc_G1_to_C1_proc:>6.2f}%")
    print("-" * 65)
    print(f"Gauss z1 -> Chirp z2 (Raw Cross-Scan, no transform):  {auc_G1_to_C2_raw:>6.2f}%")
    print(f"Gauss z1 -> Chirp z2 (USING z1 TRANSFORM T^(z1)):     {auc_G1_to_C2_proc_gen:>6.2f}% (Generalization Test)")
    print(f"Gauss z1 -> Chirp z2 (Refit transform on z2):         {auc_G1_to_C2_proc_refit:>6.2f}% (Transductive Ceiling)")

    # Also test Square -> Chirp generalization
    clf_S1 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_S1.fit(Z1["Square"][train_idx_z1], y_z1[train_idx_z1])

    ridge_C2S_z1 = Ridge(alpha=10.0, fit_intercept=False, random_state=42).fit(Z1["Chirp"][fit_snd_z1], Z1["Square"][fit_snd_z1])
    T_C2S_z1 = ridge_C2S_z1.coef_.T

    auc_S1_to_C1_ridge = roc_auc_score(y_z1, clf_S1.predict_proba(Z1["Chirp"] @ T_C2S_z1)[:, 1]) * 100
    auc_S1_to_C2_raw = roc_auc_score(y_z2, clf_S1.predict_proba(Z2["Chirp"])[:, 1]) * 100
    auc_S1_to_C2_ridge_gen = roc_auc_score(y_z2, clf_S1.predict_proba(Z2["Chirp"] @ T_C2S_z1)[:, 1]) * 100

    print("\n--- Square -> Chirp Generalization Audit ---")
    print(f"Square z1 -> Chirp z1 (Ridge on z1):                  {auc_S1_to_C1_ridge:>6.2f}%")
    print(f"Square z1 -> Chirp z2 (Raw Cross-Scan):               {auc_S1_to_C2_raw:>6.2f}%")
    print(f"Square z1 -> Chirp z2 (USING z1 RIDGE TRANSFORM):     {auc_S1_to_C2_ridge_gen:>6.2f}% (Generalization Test)")

    summary_step25 = {
        "defect_vector_realignment": {
            "cos_raw_GC": float(cos_raw_GC),
            "cos_realigned_GC": float(cos_realigned_GC),
            "gain": float(cos_realigned_GC - cos_raw_GC),
        },
        "cycle_consistency": {
            "procrustes_cycle_error": float(cycle_err_procrustes),
            "ridge_cycle_error": float(cycle_err_ridge),
        },
        "cross_scan_generalization": {
            "Gauss_z1_to_Gauss_z2": float(auc_G1_to_G2),
            "Gauss_z1_to_Chirp_z1_proc": float(auc_G1_to_C1_proc),
            "Gauss_z1_to_Chirp_z2_raw": float(auc_G1_to_C2_raw),
            "Gauss_z1_to_Chirp_z2_proc_generalized": float(auc_G1_to_C2_proc_gen),
            "Gauss_z1_to_Chirp_z2_proc_refit": float(auc_G1_to_C2_proc_refit),
            "Square_z1_to_Chirp_z1_ridge": float(auc_S1_to_C1_ridge),
            "Square_z1_to_Chirp_z2_raw": float(auc_S1_to_C2_raw),
            "Square_z1_to_Chirp_z2_ridge_generalized": float(auc_S1_to_C2_ridge_gen),
        }
    }

    out_file = os.path.join(OUTPUT_DIR, "step_2_5_audit_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary_step25, f, indent=2)
    print(f"\nStep 2.5 audit successfully saved to: {out_file}")


if __name__ == "__main__":
    run_step_2_5_audit()
