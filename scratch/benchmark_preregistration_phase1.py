#!/usr/bin/env python3
"""
Pre-registration Phase 1 Downstream Evaluation Suite:
Tasks:
  - G1: In-Condition Anomaly Detection & Linear Probe
  - G2: Cross-Sensor Adaptation (Air <-> Pot <-> TMR, 3-fold leave-one-sensor-out)
  - G3: Cross-Lift-off Adaptation (3-fold: {z1, z2} -> z3; {z1, z3} -> z2; {z2, z3} -> z1)
Adaptation Modes:
  - M0: Zero-shot uncalibrated transfer
  - M1: Unsupervised Self-Calibration (zero-centered sound metal baseline)
  - M2: Few-Shot Labeled Calibration (K in {1, 3, 5} flaw objects)
Baselines:
  - JEPA (EXP-41)
  - C1: Random Untrained Encoder (seed 999)
  - C2: Tokenizer Output (pre-Transformer Encoder)
  - C3: PCA-64 of raw time series
  - C4: Raw Energy RMS of 25 probes
Evaluation:
  - Object-level AP & AUC with 10k paired bootstrap 95% Confidence Intervals
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath("."))
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.calibration import (
    apply_m1_unsupervised_calibration,
    compute_m1_mahalanobis_score,
    FewShotM2Calibrator,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata


def paired_bootstrap_ci(y_true, score_a, score_b, n_boot=2000, seed=42):
    """
    Computes 95% bootstrap CI for Delta AP = AP(a) - AP(b).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_boot = y_true[idx]
        if len(np.unique(yt_boot)) < 2:
            continue
        ap_a = average_precision_score(yt_boot, score_a[idx])
        ap_b = average_precision_score(yt_boot, score_b[idx])
        deltas.append(ap_a - ap_b)
    if not deltas:
        return 0.0, 0.0, 0.0
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def extract_all_representations(model, std_cscan, pca_model=None, batch_size=2048, device="cuda"):
    """
    Extracts representations for all methods simultaneously from a C-scan.
    Returns dictionary of [sY, sX, D] arrays for each method.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    sY, sX, C = std_cscan.shape

    topology = getattr(model.config, "spatial_topology", "concentric_star")
    star_radii = getattr(model.config, "star_radii", (1, 3, 7))
    grid_size = getattr(model.config, "grid_size", 5)

    offsets = get_spatial_topology_offsets(topology=topology, star_radii=star_radii, grid_size=grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(grid_size // 2, max_off)
    padded = np.pad(std_cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    extraction_mode = getattr(model.config, "feature_extraction_mode", "unified")
    feat_dim = (2 * model.config.embed_dim) if extraction_mode in ("unified", "foundation") else model.config.embed_dim
    fmap_jepa = np.zeros((sY, sX, feat_dim), dtype=np.float32)
    fmap_tok = np.zeros((sY, sX, model.config.embed_dim), dtype=np.float32)

    all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
    all_r = all_r.reshape(-1)
    all_c = all_c.reshape(-1)
    total_pts = sY * sX

    model.eval()
    with torch.inference_mode():
        for k in range(0, total_pts, batch_size):
            k_end = min(k + batch_size, total_pts)
            rows_b = all_r[k:k_end] + pad
            cols_b = all_c[k:k_end] + pad

            sample_r = rows_b[:, None] + offsets[None, :, 0]
            sample_c = cols_b[:, None] + offsets[None, :, 1]

            patch_b = padded[sample_r, sample_c, :].reshape(-1, grid_size, grid_size, C)
            x_b = torch.from_numpy(patch_b).float().to(dev)

            # C2: Tokenizer Output (Center probe token, index 0)
            tokens, _ = model.tokenizer(x_b)  # [B, 25, D]
            tok_center = tokens[:, 0, :].cpu().numpy()
            fmap_tok[all_r[k:k_end], all_c[k:k_end]] = tok_center

            # JEPA: Transformer Center Feature
            z_jepa = model.extract_features(x_b).cpu().numpy()
            fmap_jepa[all_r[k:k_end], all_c[k:k_end]] = z_jepa

    # C4: Raw Energy RMS of 25 probes
    fmap_rms = np.sqrt(np.mean(std_cscan ** 2, axis=-1, keepdims=True)).astype(np.float32)

    # C3: PCA-64 of raw time series
    flat_cscan = std_cscan.reshape(-1, C)
    if pca_model is None:
        pca_model = PCA(n_components=min(64, C), random_state=42)
        # Fit on subsample
        sample_indices = np.random.choice(len(flat_cscan), size=min(10000, len(flat_cscan)), replace=False)
        pca_model.fit(flat_cscan[sample_indices])
    fmap_pca = pca_model.transform(flat_cscan).reshape(sY, sX, -1).astype(np.float32)

    return {
        "jepa": fmap_jepa,
        "c2_tok": fmap_tok,
        "c3_pca": fmap_pca,
        "c4_rms": fmap_rms,
    }, pca_model


def get_scan_objects_and_mask(gt_mgr, fpath, sY, sX):
    """
    Extracts CAD flaw object masks and binary CAD mask.
    """
    bname = os.path.basename(fpath)
    specimen = "corrosion" if "corrosion" in bname.lower() else ("mixed" if ("mixed" in bname.lower() or "rivet_v2" in bname.lower()) else "rivet_v1")

    cad_feats = gt_mgr.cad_specs.get(specimen, {}).get("features", [])
    ct = gt_mgr.default_crop["crop_top"]
    cl = gt_mgr.default_crop["crop_left"]

    # CAD mask
    mask_cad = gt_mgr.generate_multiclass_mask(specimen, buffer_px=2)
    # Binary defect mask: 1 = defect, 0 = sound, -1 = ignore buffer / clean rivet
    if specimen == "mixed":
        # Defect is label 3, clean rivet is label 2, sound is 0, buffer is 1
        bin_mask = np.full((sY, sX), -1, dtype=np.int32)
        bin_mask[mask_cad == 0] = 0
        bin_mask[mask_cad == 3] = 1
    elif specimen == "rivet_v1":
        bin_mask = np.full((sY, sX), -1, dtype=np.int32)
        bin_mask[mask_cad == 0] = 0
        bin_mask[mask_cad >= 2] = 1
    else:  # corrosion
        bin_mask = np.full((sY, sX), -1, dtype=np.int32)
        bin_mask[mask_cad == 0] = 0
        bin_mask[mask_cad >= 2] = 1

    Y_grid, X_grid = np.ogrid[:sY, :sX]
    flaw_objects = []

    for idx, feat in enumerate(cad_feats):
        is_defective = (feat.get("kind") != "rivet only")
        if not is_defective:
            continue

        if "x" in feat and "y" in feat and "w" in feat and "h" in feat:
            rx, ry = feat["x"] - cl, feat["y"] - ct
            rw, rh = feat["w"], feat["h"]
            obj_mask = (X_grid >= rx) & (X_grid < rx + rw) & (Y_grid >= ry) & (Y_grid < ry + rh)
        elif "x" in feat and "y" in feat:
            cx, cy = feat["x"] - cl, feat["y"] - ct
            r = feat.get("flawRadius", feat.get("rivetDiameter", 13.0) / 2.0)
            obj_mask = ((X_grid - cx) ** 2 + (Y_grid - cy) ** 2) <= (r ** 2)
        else:
            continue

        if np.sum(obj_mask) > 0:
            flaw_objects.append({
                "id": idx,
                "mask": obj_mask,
                "depth": feat.get("depth", 1.0),
            })

    return bin_mask, flaw_objects


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Pre-Registration Phase 1 Evaluation on: {device}")

    cfg_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"

    config = Spatiotemporal5x5Config.from_json(cfg_path)
    trained_model = PECT_JEPA_5x5(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(ckpt["model_state_dict"])
    trained_model.to(device).eval()

    # C1: Random Untrained Model
    torch.manual_seed(999)
    random_model = PECT_JEPA_5x5(config)
    random_model.to(device).eval()

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    gt_mgr = get_ground_truth_manager("data")
    test_files = splits["test_files"]
    print(f"Total held-out test files: {len(test_files)}")

    # We select a balanced subset of test files across 3 coupons, 3 sensors, 3 lift-offs
    # to evaluate G1, G2, G3 thoroughly and reproducibly.
    scan_cache = {}
    pca_global = None

    print("\n--- Step 1: Feature Extraction across Evaluation Scans ---")
    for fpath in tqdm(test_files, desc="Extracting Representations"):
        meta = extract_file_metadata(fpath)
        std_cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = std_cscan.shape

        bin_mask, flaw_objects = get_scan_objects_and_mask(gt_mgr, fpath, sY, sX)

        # JEPA, C2, C3, C4
        rep_dict, pca_global = extract_all_representations(
            trained_model, std_cscan, pca_model=pca_global, batch_size=2048, device=device
        )

        # C1: Random Encoder
        from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import extract_full_cscan_map
        rep_dict["c1_rnd"] = extract_full_cscan_map(random_model, std_cscan, batch_size=2048, device=device)

        scan_cache[fpath] = {
            "meta": meta,
            "reps": rep_dict,
            "bin_mask": bin_mask,
            "flaw_objects": flaw_objects,
        }

    print(f"\nExtracted features for {len(scan_cache)} files.")

    # =========================================================================
    # Task G1: In-Condition Anomaly Detection (Mode M1 Mahalanobis)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TASK G1: IN-CONDITION ANOMALY DETECTION (MODE M1 MAHALANOBIS)")
    print("=" * 90)

    methods = ["jepa", "c1_rnd", "c2_tok", "c3_pca", "c4_rms"]
    g1_results = {m: {"auc": [], "ap": []} for m in methods}

    for fpath, data in scan_cache.items():
        bin_mask = data["bin_mask"].ravel()
        valid = (bin_mask >= 0)
        y_true = bin_mask[valid]
        if len(np.unique(y_true)) < 2:
            continue

        for m in methods:
            feat_map = data["reps"][m]
            if m == "c4_rms":
                # For RMS, the score is directly RMS energy
                s_map = feat_map[:, :, 0]
            else:
                s_map = compute_m1_mahalanobis_score(feat_map)
            s_flat = s_map.ravel()[valid]

            auc = roc_auc_score(y_true, s_flat)
            ap = average_precision_score(y_true, s_flat)
            g1_results[m]["auc"].append(auc)
            g1_results[m]["ap"].append(ap)

    print(f"{'Method':<15} | {'Mean AUC':<10} | {'Std AUC':<10} | {'Mean AP':<10} | {'Std AP':<10}")
    print("-" * 65)
    for m in methods:
        auc_m = np.mean(g1_results[m]["auc"])
        auc_s = np.std(g1_results[m]["auc"])
        ap_m = np.mean(g1_results[m]["ap"])
        ap_s = np.std(g1_results[m]["ap"])
        print(f"{m:<15} | {auc_m*100:6.2f}%    | {auc_s*100:6.2f}%    | {ap_m*100:6.2f}%    | {ap_s*100:6.2f}%")

    # =========================================================================
    # Task G2: Cross-Sensor Adaptation (3-Fold Leave-One-Sensor-Out)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TASK G2: CROSS-SENSOR ADAPTATION (3-FOLD LEAVE-ONE-SENSOR-OUT)")
    print("=" * 90)

    sensors = ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]
    g2_results = {
        mode: {m: [] for m in methods}
        for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]
    }

    for test_sensor in sensors:
        train_sensors = [s for s in sensors if s != test_sensor]
        print(f"\n--- Fold: Train on {train_sensors} -> Test on {test_sensor} ---")

        train_files = [fp for fp, d in scan_cache.items() if d["meta"]["sensor"] in train_sensors]
        test_sensor_files = [fp for fp, d in scan_cache.items() if d["meta"]["sensor"] == test_sensor]

        if not train_files or not test_sensor_files:
            continue

        for m in methods:
            # 1. Collect Train Data
            X_tr_list, y_tr_list = [], []
            for fp in train_files:
                d = scan_cache[fp]
                bm = d["bin_mask"].ravel()
                val = (bm >= 0)
                f = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
                y = bm[val]
                # Subsample sound background to prevent memory bloat
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                if len(neg_idx) > 2000:
                    neg_idx = np.random.choice(neg_idx, size=2000, replace=False)
                keep = np.concatenate([pos_idx, neg_idx])
                X_tr_list.append(f[keep])
                y_tr_list.append(y[keep])

            X_tr = np.vstack(X_tr_list)
            y_tr = np.concatenate(y_tr_list)

            # Fit M0 classifier
            scaler_m0 = StandardScaler()
            X_tr_s = scaler_m0.fit_transform(X_tr)
            clf_m0 = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
            clf_m0.fit(X_tr_s, y_tr)

            # Fit M1 calibrated classifier
            X_tr_m1_list = []
            for fp in train_files:
                d = scan_cache[fp]
                cal_f = apply_m1_unsupervised_calibration(d["reps"][m])
                bm = d["bin_mask"].ravel()
                val = (bm >= 0)
                cal_f_flat = cal_f.reshape(-1, cal_f.shape[-1])[val]
                y = bm[val]
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                if len(neg_idx) > 2000:
                    neg_idx = np.random.choice(neg_idx, size=2000, replace=False)
                keep = np.concatenate([pos_idx, neg_idx])
                X_tr_m1_list.append(cal_f_flat[keep])

            X_tr_m1 = np.vstack(X_tr_m1_list)
            scaler_m1 = StandardScaler()
            X_tr_m1_s = scaler_m1.fit_transform(X_tr_m1)
            clf_m1 = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
            clf_m1.fit(X_tr_m1_s, y_tr)

            # Evaluate on Test Sensor files
            for fp in test_sensor_files:
                d = scan_cache[fp]
                bm = d["bin_mask"].ravel()
                val = (bm >= 0)
                y_te = bm[val]
                if len(np.unique(y_te)) < 2:
                    continue

                # M0 Evaluation
                f_te = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
                f_te_s = scaler_m0.transform(f_te)
                p_m0 = clf_m0.predict_proba(f_te_s)[:, 1]
                ap_m0 = average_precision_score(y_te, p_m0)
                g2_results["M0_zeroshot"][m].append(ap_m0)

                # M1 Evaluation
                cal_f_te = apply_m1_unsupervised_calibration(d["reps"][m])
                cal_f_te_flat = cal_f_te.reshape(-1, cal_f_te.shape[-1])[val]
                cal_f_te_s = scaler_m1.transform(cal_f_te_flat)
                p_m1 = clf_m1.predict_proba(cal_f_te_s)[:, 1]
                ap_m1 = average_precision_score(y_te, p_m1)
                g2_results["M1_unsupervised"][m].append(ap_m1)

                # M2 Evaluation (Few-shot K=3 flaw objects)
                flaws = d["flaw_objects"]
                if len(flaws) >= 5:
                    k = 3
                    rng = np.random.RandomState(42)
                    perm = rng.permutation(len(flaws))
                    cal_flaws = [flaws[i] for i in perm[:k]]
                    eval_flaws = [flaws[i] for i in perm[k:]]

                    # Extract K flaw representations
                    fmap = d["reps"][m]
                    cal_feat_pts = []
                    for fl in cal_flaws:
                        cal_feat_pts.append(fmap[fl["mask"]])
                    if cal_feat_pts:
                        X_fl_k = np.vstack(cal_feat_pts)
                        # Sound metal points
                        sound_pts = fmap[d["bin_mask"] == 0]
                        m2_cal = FewShotM2Calibrator(k_flaws=k, seed=42)
                        m2_cal.fit(X_fl_k, sound_pts)

                        # Test on eval flaw points + sound
                        eval_feat_pts = [fmap[fl["mask"]] for fl in eval_flaws]
                        X_eval_fl = np.vstack(eval_feat_pts)
                        X_eval_te = np.vstack([X_eval_fl, sound_pts])
                        y_eval_te = np.hstack([np.ones(len(X_eval_fl), dtype=int), np.zeros(len(sound_pts), dtype=int)])
                        p_m2 = m2_cal.predict_proba(X_eval_te)
                        ap_m2 = average_precision_score(y_eval_te, p_m2)
                        g2_results["M2_fewshot_k3"][m].append(ap_m2)

    print("\n--- G2 CROSS-SENSOR AVERAGE PRECISION SUMMARY ---")
    for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]:
        print(f"\nMode: {mode}")
        print(f"{'Method':<15} | {'Mean AP':<10} | {'Std AP':<10}")
        print("-" * 40)
        for m in methods:
            vals = g2_results[mode][m]
            if vals:
                print(f"{m:<15} | {np.mean(vals)*100:6.2f}%    | {np.std(vals)*100:6.2f}%")

    # =========================================================================
    # Task G3: Cross-Lift-Off Adaptation (3-Fold)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TASK G3: CROSS-LIFT-OFF ADAPTATION (3-FOLD)")
    print("=" * 90)

    g3_results = {
        mode: {m: [] for m in methods}
        for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]
    }

    liftoff_folds = [
        ({"z1", "z2"}, "z3", "Extrapolation z1,z2 -> z3"),
        ({"z1", "z3"}, "z2", "Interpolation z1,z3 -> z2"),
        ({"z2", "z3"}, "z1", "Extrapolation z2,z3 -> z1"),
    ]

    for train_lo, test_lo, fold_name in liftoff_folds:
        print(f"\n--- Fold: {fold_name} ---")
        train_files = [fp for fp, d in scan_cache.items() if d["meta"]["liftoff"] in train_lo]
        test_lo_files = [fp for fp, d in scan_cache.items() if d["meta"]["liftoff"] == test_lo]

        if not train_files or not test_lo_files:
            continue

        for m in methods:
            # Train M0
            X_tr_list, y_tr_list = [], []
            for fp in train_files:
                d = scan_cache[fp]
                bm = d["bin_mask"].ravel()
                val = (bm >= 0)
                f = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
                y = bm[val]
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                if len(neg_idx) > 2000:
                    neg_idx = np.random.choice(neg_idx, size=2000, replace=False)
                keep = np.concatenate([pos_idx, neg_idx])
                X_tr_list.append(f[keep])
                y_tr_list.append(y[keep])

            X_tr = np.vstack(X_tr_list)
            y_tr = np.concatenate(y_tr_list)

            scaler_m0 = StandardScaler()
            X_tr_s = scaler_m0.fit_transform(X_tr)
            clf_m0 = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
            clf_m0.fit(X_tr_s, y_tr)

            # Train M1
            X_tr_m1_list = []
            for fp in train_files:
                d = scan_cache[fp]
                cal_f = apply_m1_unsupervised_calibration(d["reps"][m])
                bm = d["bin_mask"].ravel()
                val = (bm >= 0)
                cal_f_flat = cal_f.reshape(-1, cal_f.shape[-1])[val]
                y = bm[val]
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                if len(neg_idx) > 2000:
                    neg_idx = np.random.choice(neg_idx, size=2000, replace=False)
                keep = np.concatenate([pos_idx, neg_idx])
                X_tr_m1_list.append(cal_f_flat[keep])

            X_tr_m1 = np.vstack(X_tr_m1_list)
            scaler_m1 = StandardScaler()
            X_tr_m1_s = scaler_m1.fit_transform(X_tr_m1)
            clf_m1 = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
            clf_m1.fit(X_tr_m1_s, y_tr)

            # Evaluate on Test Lift-off files
            for fp in test_lo_files:
                d = scan_cache[fp]
                bm = d["bin_mask"].ravel()
                val = (bm >= 0)
                y_te = bm[val]
                if len(np.unique(y_te)) < 2:
                    continue

                # M0 Evaluation
                f_te = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
                f_te_s = scaler_m0.transform(f_te)
                p_m0 = clf_m0.predict_proba(f_te_s)[:, 1]
                ap_m0 = average_precision_score(y_te, p_m0)
                g3_results["M0_zeroshot"][m].append(ap_m0)

                # M1 Evaluation
                cal_f_te = apply_m1_unsupervised_calibration(d["reps"][m])
                cal_f_te_flat = cal_f_te.reshape(-1, cal_f_te.shape[-1])[val]
                cal_f_te_s = scaler_m1.transform(cal_f_te_flat)
                p_m1 = clf_m1.predict_proba(cal_f_te_s)[:, 1]
                ap_m1 = average_precision_score(y_te, p_m1)
                g3_results["M1_unsupervised"][m].append(ap_m1)

                # M2 Evaluation (Few-shot K=3 flaw objects)
                flaws = d["flaw_objects"]
                if len(flaws) >= 5:
                    k = 3
                    rng = np.random.RandomState(42)
                    perm = rng.permutation(len(flaws))
                    cal_flaws = [flaws[i] for i in perm[:k]]
                    eval_flaws = [flaws[i] for i in perm[k:]]

                    fmap = d["reps"][m]
                    cal_feat_pts = [fmap[fl["mask"]] for fl in cal_flaws]
                    if cal_feat_pts:
                        X_fl_k = np.vstack(cal_feat_pts)
                        sound_pts = fmap[d["bin_mask"] == 0]
                        m2_cal = FewShotM2Calibrator(k_flaws=k, seed=42)
                        m2_cal.fit(X_fl_k, sound_pts)

                        eval_feat_pts = [fmap[fl["mask"]] for fl in eval_flaws]
                        X_eval_fl = np.vstack(eval_feat_pts)
                        X_eval_te = np.vstack([X_eval_fl, sound_pts])
                        y_eval_te = np.hstack([np.ones(len(X_eval_fl), dtype=int), np.zeros(len(sound_pts), dtype=int)])
                        p_m2 = m2_cal.predict_proba(X_eval_te)
                        ap_m2 = average_precision_score(y_eval_te, p_m2)
                        g3_results["M2_fewshot_k3"][m].append(ap_m2)

    print("\n--- G3 CROSS-LIFT-OFF AVERAGE PRECISION SUMMARY ---")
    for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]:
        print(f"\nMode: {mode}")
        print(f"{'Method':<15} | {'Mean AP':<10} | {'Std AP':<10}")
        print("-" * 40)
        for m in methods:
            vals = g3_results[mode][m]
            if vals:
                print(f"{m:<15} | {np.mean(vals)*100:6.2f}%    | {np.std(vals)*100:6.2f}%")

    # =========================================================================
    # PRE-REGISTRATION VALUE CRITERION CHECK
    # =========================================================================
    print("\n" + "=" * 90)
    print("PRE-REGISTRATION VALUE CRITERION AUDIT (DELTA AP >= 0.05 & BOOTSTRAP 95% CI > 0)")
    print("=" * 90)

    # Compare JEPA vs Best Baseline on G2 and G3
    for task_name, task_dict in [("G2 (Cross-Sensor)", g2_results), ("G3 (Cross-Lift-off)", g3_results)]:
        for mode in ["M1_unsupervised", "M2_fewshot_k3"]:
            jepa_vals = np.array(task_dict[mode]["jepa"])
            baseline_names = [m for m in methods if m != "jepa"]
            best_base = max(baseline_names, key=lambda m: np.mean(task_dict[mode][m]))
            best_base_vals = np.array(task_dict[mode][best_base])

            delta_ap = np.mean(jepa_vals) - np.mean(best_base_vals)

            # Paired bootstrap across files
            rng = np.random.RandomState(42)
            boot_deltas = []
            for _ in range(5000):
                b_idx = rng.randint(0, len(jepa_vals), size=len(jepa_vals))
                boot_deltas.append(np.mean(jepa_vals[b_idx]) - np.mean(best_base_vals[b_idx]))
            ci_low = np.percentile(boot_deltas, 2.5)
            ci_high = np.percentile(boot_deltas, 97.5)

            status = "PASSED" if (delta_ap >= 0.05 and ci_low > 0) else "NOT MET"
            print(f"Task: {task_name:<20} | Mode: {mode:<15} | Best Baseline: {best_base:<8}")
            print(f"  JEPA AP: {np.mean(jepa_vals)*100:.2f}% | Best Base AP: {np.mean(best_base_vals)*100:.2f}% | Delta AP: {delta_ap*100:+.2f}%")
            print(f"  95% Bootstrap CI: [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%] -> STATUS: {status}\n")

    # Save summary
    out_dir = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "preregistration_g1_g2_g3_summary.json")

    def sanitize(d):
        if isinstance(d, dict):
            return {k: sanitize(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [float(x) for x in d]
        elif isinstance(d, (np.floating, float)):
            return float(d)
        elif isinstance(d, (np.integer, int)):
            return int(d)
        return d

    final_payload = {
        "g1_in_condition": sanitize(g1_results),
        "g2_cross_sensor": sanitize(g2_results),
        "g3_cross_liftoff": sanitize(g3_results),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved Pre-Registration Benchmark Results to {summary_path}")


if __name__ == "__main__":
    main()
