#!/usr/bin/env python3
"""
Diagnostic Script for Stage 1: C3 & C4 Audit.
- C3: Residual Delta H' and defect content under spatial block CV
- C4: Residual scores and object-level clean vs corroded rivet discrimination (N=31)
"""

import os
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata


def evaluate_spatial_block_cv(X, y, coords, H=270, W=270, buffer_px=14, seed=42):
    r_mid = H // 2
    c_mid = W // 2
    r_coords = coords[:, 0]
    c_coords = coords[:, 1]
    
    is_buffer = (np.abs(r_coords - r_mid) <= buffer_px) | (np.abs(c_coords - c_mid) <= buffer_px)
    valid_pts = ~is_buffer

    quadrants = np.zeros(len(coords), dtype=int)
    quadrants[(r_coords < r_mid) & (c_coords < c_mid)] = 0
    quadrants[(r_coords < r_mid) & (c_coords >= c_mid)] = 1
    quadrants[(r_coords >= r_mid) & (c_coords < c_mid)] = 2
    quadrants[(r_coords >= r_mid) & (c_coords >= c_mid)] = 3

    aucs, aps = [], []
    for test_quad in range(4):
        te_mask = (quadrants == test_quad) & valid_pts
        tr_mask = (quadrants != test_quad) & valid_pts

        y_te = y[te_mask]
        y_tr = y[tr_mask]
        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            continue

        X_tr = X[tr_mask]
        X_te = X[te_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, random_state=seed)
        clf.fit(X_tr_s, y_tr)
        probs = clf.predict_proba(X_te_s)[:, 1]
        aucs.append(roc_auc_score(y_te, probs))
        aps.append(average_precision_score(y_te, probs))

    if len(aucs) == 0:
        return np.nan, np.nan
    return float(np.mean(aucs)), float(np.mean(aps))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== C3 & C4 FORENSIC AUDIT (DEVICE: {device}) ===")

    cfg41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"
    cfg40_path = "experiments/5x5/exp40_stabilized_deconv/checkpoints/config_5x5.json"
    ckpt40_path = "experiments/5x5/exp40_stabilized_deconv/checkpoints/best_model_5x5.pt"

    cfg41 = Spatiotemporal5x5Config.from_json(cfg41_path)
    cfg40 = Spatiotemporal5x5Config.from_json(cfg40_path)

    m41_tr = PECT_JEPA_5x5(cfg41).to(device).eval()
    ckpt41 = torch.load(ckpt41_path, map_location=device)
    m41_tr.load_state_dict(ckpt41["model_state_dict"])

    m40_tr = PECT_JEPA_5x5(cfg40).to(device).eval()
    ckpt40 = torch.load(ckpt40_path, map_location=device)
    m40_tr.load_state_dict(ckpt40["model_state_dict"])

    with open(split41_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_files = splits["train_files"]
    test_files = splits["test_files"]
    gt_mgr = get_ground_truth_manager("data")

    offsets = get_spatial_topology_offsets(cfg41.spatial_topology, tuple(cfg41.star_radii), cfg41.grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(cfg41.grid_size // 2, max_off)

    # -------------------------------------------------------------------------
    # 1. COMPUTE TRAIN MEAN TABLE mu_i [9, 64] and VARIANCES sigma_i [9, 64]
    # -------------------------------------------------------------------------
    print("\n[Step 1] Computing position lookup table mu_i across train files...")
    train_dH_list = []
    for fpath in train_files[:10]:
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        rng = np.random.RandomState(42)
        rows = rng.randint(0, sY, size=400)
        cols = rng.randint(0, sX, size=400)
        patches = []
        for r, c in zip(rows, cols):
            sample_r = r + pad + offsets[:, 0]
            sample_c = c + pad + offsets[:, 1]
            p = padded[sample_r, sample_c, :].reshape(cfg41.grid_size, cfg41.grid_size, C)
            patches.append(p)
        X_batch = torch.from_numpy(np.array(patches, dtype=np.float32)).to(device)
        with torch.no_grad():
            out = m41_tr(X_batch)
            dH = out["H_target_for_loss"]  # [B, 9, 64]
            train_dH_list.append(dH.cpu().numpy())

    all_train_dH = np.vstack(train_dH_list)  # [4000, 9, 64]
    mu_pos = np.mean(all_train_dH, axis=0)   # [9, 64]
    var_pos = np.var(all_train_dH, axis=0)   # [9, 64]
    std_pos = np.std(all_train_dH, axis=0)
    print(f"Table mu_pos computed: shape={mu_pos.shape}, mean_norm={np.linalg.norm(mu_pos, axis=-1).mean():.4f}")

    # -------------------------------------------------------------------------
    # 2. C3: EXTRACT PATCHES WITH DEFECT LABELS (TRAIN & TEST)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Extracting train and test samples for Delta H' defect R^2 analysis...")
    def sample_labeled_patches(file_list, n_per_file=300, seed=42):
        patches, labels, metas = [], [], []
        rng = np.random.RandomState(seed)
        for fpath in file_list:
            meta = extract_file_metadata(fpath)
            cscan = load_cscan_from_tdms(fpath)
            sY, sX, C = cscan.shape
            mask = gt_mgr.get_ground_truth_mask_for_file(fpath, aligned_scan=True)
            H = min(mask.shape[0], sY)
            W = min(mask.shape[1], sX)
            mask = mask[:H, :W]
            cscan = cscan[:H, :W]

            padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            pos_r, pos_c = np.where(mask == 1)
            neg_r, neg_c = np.where(mask == 0)

            n_pos = min(len(pos_r), n_per_file // 2)
            n_neg = n_per_file - n_pos

            selected_r, selected_c, selected_y = [], [], []
            if n_pos > 0:
                idx_p = rng.choice(len(pos_r), size=n_pos, replace=False)
                selected_r.extend(pos_r[idx_p])
                selected_c.extend(pos_c[idx_p])
                selected_y.extend([1] * n_pos)
            if len(neg_r) > 0:
                idx_n = rng.choice(len(neg_r), size=min(n_neg, len(neg_r)), replace=False)
                selected_r.extend(neg_r[idx_n])
                selected_c.extend(neg_c[idx_n])
                selected_y.extend([0] * len(idx_n))

            for r, c, y in zip(selected_r, selected_c, selected_y):
                sample_r = r + pad + offsets[:, 0]
                sample_c = c + pad + offsets[:, 1]
                p = padded[sample_r, sample_c, :].reshape(cfg41.grid_size, cfg41.grid_size, C)
                patches.append(p)
                labels.append(y)
                metas.append(meta)

        return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int32), metas

    X_tr_p, y_tr, meta_tr = sample_labeled_patches(train_files[:6], n_per_file=400, seed=123)
    X_te_p, y_te, meta_te = sample_labeled_patches(test_files[:10], n_per_file=400, seed=456)

    # Compute Delta H and Delta H'
    with torch.no_grad():
        dH_tr = m41_tr(torch.from_numpy(X_tr_p).to(device))["H_target_for_loss"].cpu().numpy()
        dH_te = m41_tr(torch.from_numpy(X_te_p).to(device))["H_target_for_loss"].cpu().numpy()
        pred_te = m41_tr(torch.from_numpy(X_te_p).to(device))["H_pred_for_loss"].cpu().numpy()

    # Delta H' = Delta H - mu_pos
    dH_prime_tr = dH_tr - mu_pos[None, :, :]  # [N, 9, 64]
    dH_prime_te = dH_te - mu_pos[None, :, :]  # [N, 9, 64]

    # Center probe Delta H' (index 0) and flattened [9*64]
    dH_prime_tr_center = dH_prime_tr[:, 0, :]
    dH_prime_te_center = dH_prime_te[:, 0, :]
    dH_prime_tr_all = dH_prime_tr.reshape(len(dH_prime_tr), -1)
    dH_prime_te_all = dH_prime_te.reshape(len(dH_prime_te), -1)

    # Linear Regression R^2 on Defect Label
    ridge_cen = Ridge(alpha=1.0).fit(dH_prime_tr_center, y_tr)
    r2_tr_cen = ridge_cen.score(dH_prime_tr_center, y_tr)
    r2_te_cen = ridge_cen.score(dH_prime_te_center, y_te)

    ridge_all = Ridge(alpha=1.0).fit(dH_prime_tr_all, y_tr)
    r2_tr_all = ridge_all.score(dH_prime_tr_all, y_tr)
    r2_te_all = ridge_all.score(dH_prime_te_all, y_te)

    print(f"\nLinear Regression R^2 on Defect Label (from Delta H'):")
    print(f"  Center Probe (64D):   Train R^2 = {r2_tr_cen:.4f}, Test R^2 = {r2_te_cen:.4f}")
    print(f"  All 9 Probes (576D):  Train R^2 = {r2_tr_all:.4f}, Test R^2 = {r2_te_all:.4f}")

    # ANOVA / Variance decomposition on Delta H'
    df_anova = pd.DataFrame([
        {
            "file": m["filename"],
            "sensor": m["sensor"],
            "liftoff": m["liftoff"],
            "norm_dH_prime": np.linalg.norm(dH_prime_te[i]),
        }
        for i, m in enumerate(meta_te)
    ])
    # Add representations
    dH_flat = dH_prime_te.reshape(len(dH_prime_te), -1)
    total_ss = np.sum((dH_flat - np.mean(dH_flat, axis=0)) ** 2)

    # File SS
    file_means = df_anova.groupby("file").indices
    file_ss = sum(len(idx) * np.sum((np.mean(dH_flat[idx], axis=0) - np.mean(dH_flat, axis=0)) ** 2) for idx in file_means.values())
    r2_file = file_ss / total_ss

    # Sensor SS
    sensor_means = df_anova.groupby("sensor").indices
    sensor_ss = sum(len(idx) * np.sum((np.mean(dH_flat[idx], axis=0) - np.mean(dH_flat, axis=0)) ** 2) for idx in sensor_means.values())
    r2_sensor = sensor_ss / total_ss

    # Liftoff SS
    lo_means = df_anova.groupby("liftoff").indices
    lo_ss = sum(len(idx) * np.sum((np.mean(dH_flat[idx], axis=0) - np.mean(dH_flat, axis=0)) ** 2) for idx in lo_means.values())
    r2_lo = lo_ss / total_ss

    # Additive model (sensor + liftoff)
    cell_means = df_anova.groupby(["sensor", "liftoff"]).indices
    cell_ss = sum(len(idx) * np.sum((np.mean(dH_flat[idx], axis=0) - np.mean(dH_flat, axis=0)) ** 2) for idx in cell_means.values())
    r2_sensor_lo = cell_ss / total_ss

    print(f"\nANOVA Variance Decomposition on Delta H':")
    print(f"  R^2(File):              {r2_file * 100:.2f}%")
    print(f"  R^2(Sensor):            {r2_sensor * 100:.2f}%")
    print(f"  R^2(Lift-off):          {r2_lo * 100:.2f}%")
    print(f"  R^2(Sensor + Lift-off): {r2_sensor_lo * 100:.2f}%")
    print(f"  Sum R^2(S) + R^2(L):    {(r2_sensor + r2_lo) * 100:.2f}%")

    # Channel-normalized Predictor Loss
    mse_pred = np.mean((pred_te - dH_te) ** 2, axis=0)  # [9, 64]
    loss_norm_channel = float(np.mean(mse_pred / (var_pos + 1e-6)))
    print(f"\nChannel-Normalized Predictor Loss on Test (MSE / Var_channel): {loss_norm_channel:.4f}")

    # -------------------------------------------------------------------------
    # 3. SPATIAL BLOCK CV (14mm buffer) ON TEST FILES
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running Spatial Block CV (14mm buffer) on representative test files...")
    # Pick 1 Corrosion, 1 Rivet_v1, 1 Mixed
    te_corr = [f for f in test_files if "corosion" in f.lower() or "corrosion" in f.lower()][0]
    te_rivet = [f for f in test_files if "rivet" in f.lower() and "mixed" not in f.lower()][0]
    te_mixed = [f for f in test_files if "mixed" in f.lower() or "rivet_v2" in f.lower()][0]

    block_eval_files = [("Corrosion", te_corr), ("Rivet_v1", te_rivet), ("Mixed", te_mixed)]
    seeds_rnd = [42, 100, 2024, 777, 999]

    block_results = []
    for spec_name, fpath in block_eval_files:
        cscan = load_cscan_from_tdms(fpath)
        mask = gt_mgr.get_ground_truth_mask_for_file(fpath, aligned_scan=True)
        H = min(mask.shape[0], cscan.shape[0])
        W = min(mask.shape[1], cscan.shape[1])
        mask = mask[:H, :W]
        cscan = cscan[:H, :W]

        # Valid non-buffer pixels
        valid_r, valid_c = np.where(mask >= 0)
        y = mask[valid_r, valid_c].astype(int)
        coords = np.stack([valid_r, valid_c], axis=1)

        # 1. EXP-41 JEPA
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        patches = []
        for r, c in zip(valid_r, valid_c):
            sr = r + pad + offsets[:, 0]
            sc = c + pad + offsets[:, 1]
            p = padded[sr, sc, :].reshape(cfg41.grid_size, cfg41.grid_size, -1)
            patches.append(p)
        patches_np = np.array(patches, dtype=np.float32)

        def batched_extract(model, p_arr, b_sz=2048):
            f_list = []
            with torch.no_grad():
                for k in range(0, len(p_arr), b_sz):
                    x_k = torch.from_numpy(p_arr[k:k+b_sz]).to(device)
                    f_list.append(model.extract_features(x_k).cpu().numpy())
            return np.vstack(f_list)

        f_jepa = batched_extract(m41_tr, patches_np)
        f_t0 = batched_extract(m40_tr, patches_np)

        auc_jepa, ap_jepa = evaluate_spatial_block_cv(f_jepa, y, coords, H=H, W=W, buffer_px=14)
        auc_t0, ap_t0 = evaluate_spatial_block_cv(f_t0, y, coords, H=H, W=W, buffer_px=14)

        # 5 seeds of Random Encoder
        rnd_aucs, rnd_aps = [], []
        for s in seeds_rnd:
            torch.manual_seed(s)
            m_rnd = PECT_JEPA_5x5(cfg41).to(device).eval()
            f_rnd = batched_extract(m_rnd, patches_np)
            a_s, p_s = evaluate_spatial_block_cv(f_rnd, y, coords, H=H, W=W, buffer_px=14, seed=s)
            rnd_aucs.append(a_s)
            rnd_aps.append(p_s)

        block_results.append({
            "specimen": spec_name,
            "jepa_auc": auc_jepa, "jepa_ap": ap_jepa,
            "t0_auc": auc_t0, "t0_ap": ap_t0,
            "rnd_auc_mean": np.mean(rnd_aucs), "rnd_auc_std": np.std(rnd_aucs),
            "rnd_ap_mean": np.mean(rnd_aps), "rnd_ap_std": np.std(rnd_aps),
        })

    print("\nSpatial Block CV (14mm Buffer) Results:")
    df_blk = pd.DataFrame(block_results)
    print(df_blk.to_string())

    # -------------------------------------------------------------------------
    # 4. C4: RESIDUAL SCORES & FASTENER DISCRIMINATION (N=31) ON MIXED
    # -------------------------------------------------------------------------
    print("\n[Step 4] C4: Evaluating s_pred, s_table, Random 5 seeds, T0, Raw RMS on Mixed (N=31)...")
    mixed_test_files = [f for f in test_files if "mixed" in f.lower() or "rivet_v2" in f.lower() or "rivet2" in f.lower()]
    cad_mixed = gt_mgr.cad_specs.get("mixed", {}).get("features", [])
    ct = gt_mgr.default_crop["crop_top"]
    cl = gt_mgr.default_crop["crop_left"]

    # Collect rivet features across all 19 Mixed test files
    rivet_scores_pred = [[] for _ in range(31)]
    rivet_scores_table = [[] for _ in range(31)]
    rivet_scores_rms = [[] for _ in range(31)]
    rivet_scores_t0 = [[] for _ in range(31)]
    rivet_scores_rnd = {s: [[] for _ in range(31)] for s in seeds_rnd}
    rivet_is_defect = []

    for idx, feat in enumerate(cad_mixed):
        is_def = (feat.get("kind") != "rivet only")
        rivet_is_defect.append(1 if is_def else 0)

    # Pre-instantiate random models once
    m_rnd_dict = {}
    for s in seeds_rnd:
        torch.manual_seed(s)
        m_rnd_dict[s] = PECT_JEPA_5x5(cfg41).to(device).eval()

    # Evaluate on a representative sample of Mixed test files (5 files)
    for fpath in mixed_test_files[:5]:
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        Y_grid, X_grid = np.ogrid[:sY, :sX]

        # Raw RMS
        rms_map = np.sqrt(np.mean(cscan ** 2, axis=-1))

        # Extract disk points for each rivet
        for r_idx, feat in enumerate(cad_mixed):
            rx = feat.get("x")
            ry = feat.get("y")
            rd = feat.get("rivetDiameter", 13.0)
            if rx is None or ry is None:
                continue
            cx = rx - cl
            cy = ry - ct
            r_disk = rd / 2.0
            disk_m = ((X_grid - cx) ** 2 + (Y_grid - cy) ** 2) <= (r_disk ** 2)
            d_rows, d_cols = np.where(disk_m)
            if len(d_rows) == 0:
                continue

            # Patches inside rivet disk
            patches_r = []
            for r, c in zip(d_rows, d_cols):
                sr = r + pad + offsets[:, 0]
                sc = c + pad + offsets[:, 1]
                p = padded[sr, sc, :].reshape(cfg41.grid_size, cfg41.grid_size, -1)
                patches_r.append(p)
            X_r = torch.from_numpy(np.array(patches_r, dtype=np.float32)).to(device)

            with torch.no_grad():
                out41 = m41_tr(X_r)
                dH_r = out41["H_target_for_loss"].cpu().numpy()
                pred_r = out41["H_pred_for_loss"].cpu().numpy()
                s_p = np.mean(np.sum((pred_r - dH_r) ** 2, axis=(1, 2)))
                s_t = np.mean(np.sum((mu_pos[None, :, :] - dH_r) ** 2, axis=(1, 2)))

                # T0 anomaly score
                out40 = m40_tr(X_r)
                dH_r0 = out40["H_target_for_loss"].cpu().numpy()
                pred_r0 = out40["H_pred_for_loss"].cpu().numpy()
                s_0 = np.mean(np.sum((pred_r0 - dH_r0) ** 2, axis=(1, 2)))

            s_rms = float(np.mean(rms_map[disk_m]))

            rivet_scores_pred[r_idx].append(s_p)
            rivet_scores_table[r_idx].append(s_t)
            rivet_scores_t0[r_idx].append(s_0)
            rivet_scores_rms[r_idx].append(s_rms)

            # Random models
            for s, m_rnd in m_rnd_dict.items():
                with torch.no_grad():
                    out_rnd = m_rnd(X_r)
                    dH_rnd = out_rnd["H_target_for_loss"].cpu().numpy()
                    pred_rnd = out_rnd["H_pred_for_loss"].cpu().numpy()
                    s_rnd_val = np.mean(np.sum((pred_rnd - dH_rnd) ** 2, axis=(1, 2)))
                rivet_scores_rnd[s][r_idx].append(s_rnd_val)

    # Average score per rivet across files
    y_rivets = np.array(rivet_is_defect)  # 25 ones, 6 zeros
    sc_pred = np.array([np.mean(scores) for scores in rivet_scores_pred])
    sc_table = np.array([np.mean(scores) for scores in rivet_scores_table])
    sc_t0 = np.array([np.mean(scores) for scores in rivet_scores_t0])
    sc_rms = np.array([np.mean(scores) for scores in rivet_scores_rms])
    sc_rnd = {s: np.array([np.mean(scores) for scores in rivet_scores_rnd[s]]) for s in seeds_rnd}

    # Cluster Bootstrap (5,000 resamples over the 31 rivets)
    def bootstrap_auc_ci(y_true, scores, n_boot=5000, seed=42):
        rng = np.random.RandomState(seed)
        n = len(y_true)
        boot_aucs = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            yb = y_true[idx]
            if len(np.unique(yb)) < 2:
                continue
            boot_aucs.append(roc_auc_score(yb, scores[idx]))
        return float(np.mean(boot_aucs)), float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))

    auc_p, ci_l_p, ci_u_p = bootstrap_auc_ci(y_rivets, sc_pred)
    auc_t, ci_l_t, ci_u_t = bootstrap_auc_ci(y_rivets, sc_table)
    auc_0, ci_l_0, ci_u_0 = bootstrap_auc_ci(y_rivets, sc_t0)
    auc_rms, ci_l_rms, ci_u_rms = bootstrap_auc_ci(y_rivets, sc_rms)

    rnd_boot_results = [bootstrap_auc_ci(y_rivets, sc_rnd[s]) for s in seeds_rnd]
    auc_rnd_mean = np.mean([r[0] for r in rnd_boot_results])
    ci_l_rnd = np.mean([r[1] for r in rnd_boot_results])
    ci_u_rnd = np.mean([r[2] for r in rnd_boot_results])

    print("\nObject-Level Fastener Discrimination on Mixed (N=31, 25 Corroded vs 6 Clean):")
    print(f"  s_pred:       AUC = {auc_p*100:.2f}% (95% CI: [{ci_l_p*100:.2f}%, {ci_u_p*100:.2f}%])")
    print(f"  s_table:      AUC = {auc_t*100:.2f}% (95% CI: [{ci_l_t*100:.2f}%, {ci_u_t*100:.2f}%])")
    print(f"  T0 (EXP-40):  AUC = {auc_0*100:.2f}% (95% CI: [{ci_l_0*100:.2f}%, {ci_u_t*100:.2f}%])")
    print(f"  Raw RMS:      AUC = {auc_rms*100:.2f}% (95% CI: [{ci_l_rms*100:.2f}%, {ci_u_rms*100:.2f}%])")
    print(f"  Random (5s):  AUC = {auc_rnd_mean*100:.2f}% (95% CI: [{ci_l_rnd*100:.2f}%, {ci_u_rnd*100:.2f}%])")

    results_out = {
        "c3": {
            "r2_defect_center": {"train": r2_tr_cen, "test": r2_te_cen},
            "r2_defect_all": {"train": r2_tr_all, "test": r2_te_all},
            "r2_file": r2_file,
            "r2_sensor": r2_sensor,
            "r2_liftoff": r2_lo,
            "r2_sensor_liftoff_additive": r2_sensor_lo,
            "loss_normalized_channel": loss_norm_channel,
            "block_cv": block_results,
        },
        "c4": {
            "fastener_discrimination_mixed": {
                "s_pred": {"auc": auc_p, "ci_95": [ci_l_p, ci_u_p]},
                "s_table": {"auc": auc_t, "ci_95": [ci_l_t, ci_u_t]},
                "t0": {"auc": auc_0, "ci_95": [ci_l_0, ci_u_0]},
                "raw_rms": {"auc": auc_rms, "ci_95": [ci_l_rms, ci_u_rms]},
                "random_5seeds": {"auc": auc_rnd_mean, "ci_95": [ci_l_rnd, ci_u_rnd]},
            }
        }
    }

    out_file = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results/stage1_audit_c3_c4.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print(f"\n[Saved C3 & C4 Results to {out_file}]")


if __name__ == "__main__":
    main()
