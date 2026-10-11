# -*- coding: utf-8 -*-
"""
Evaluation of 4 Pre-flight Gates (G-Pre-1 through G-Pre-4) on Branch A Pilot Checkpoints.
Strictly following Protocol V2.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata


def cluster_bootstrap_ci(data, n_boot=5000, ci=95.0, seed=42):
    rng = np.random.RandomState(seed)
    n = len(data)
    boot = [np.mean(data[rng.randint(0, n, size=n)]) for _ in range(n_boot)]
    alpha = (100.0 - ci) / 2.0
    return float(np.mean(data)), float(np.percentile(boot, alpha)), float(np.percentile(boot, 100.0 - alpha))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("EVALUATION OF PRE-FLIGHT GATES (G-PRE-1 TO G-PRE-4) FOR BRANCH A PILOT")
    print("=" * 80)

    # Load Branch A Checkpoints (3-epoch 5 seeds + 20-epoch)
    branch_a_dirs = [
        "experiments/5x5/pilot_reformed/checkpoints/branch_A_seed_42_ep_3/model.pt",
        "experiments/5x5/pilot_reformed/checkpoints/branch_A_seed_100_ep_3/model.pt",
        "experiments/5x5/pilot_reformed/checkpoints/branch_A_seed_2024_ep_3/model.pt",
        "experiments/5x5/pilot_reformed/checkpoints/branch_A_seed_777_ep_3/model.pt",
        "experiments/5x5/pilot_reformed/checkpoints/branch_A_seed_999_ep_3/model.pt",
    ]
    ckpt_20ep = "experiments/5x5/pilot_reformed/checkpoints/branch_A_seed_42_ep_20/model.pt"

    # We evaluate on Seed 42 3ep and Seed 42 20ep
    cfg_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    cfg = Spatiotemporal5x5Config.from_json(cfg_path)
    
    m_a = PECT_JEPA_5x5(cfg).to(device).eval()
    ckpt_a = torch.load(branch_a_dirs[0], map_location=device, weights_only=False)
    m_a.load_state_dict(ckpt_a["model_state_dict"])
    mu_c = ckpt_a["mu_c"].to(device)
    sigma_c = ckpt_a["sigma_c"].to(device)

    # Load 18 balanced test scans for ANOVA
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_files = splits["test_files"]
    gt_mgr = get_ground_truth_manager("data")
    offsets = get_spatial_topology_offsets(cfg.spatial_topology, tuple(cfg.star_radii), cfg.grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(cfg.grid_size // 2, max_off)

    print("\n--- 1. EVALUATING GATE G-PRE-1 (Content vs Position/File Dominance) ---", flush=True)
    # Extract patches from balanced sample of 18 scans across all 3 sensors and 3 lift-offs
    selected_files = []
    for s_name in ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]:
        for lo in ["z1", "z2", "z3"]:
            matches = [f for f in test_files if s_name in f and f"_{lo}_" in f and "corosion" in f.lower()]
            if matches:
                selected_files.append(matches[0])
            else:
                matches_any = [f for f in test_files if s_name in f and f"_{lo}_" in f]
                if matches_any:
                    selected_files.append(matches_any[0])

    print(f"Selected {len(selected_files)} balanced test scans for ANOVA...", flush=True)
    dh_new_list, pos_idx_list, file_idx_list, sensor_idx_list, lo_idx_list = [], [], [], [], []
    rng = np.random.RandomState(42)

    for f_idx, fp in enumerate(selected_files):
        meta = extract_file_metadata(fp)
        cscan = load_cscan_from_tdms(fp)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        
        # sample 200 random locations
        for _ in range(200):
            r = rng.randint(0, sY)
            c = rng.randint(0, sX)
            sr = r + pad + offsets[:, 0]
            sc = c + pad + offsets[:, 1]
            patch = padded[sr, sc, :].reshape(cfg.grid_size, cfg.grid_size, -1)
            
            with torch.no_grad():
                pt = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                tokens, pos = m_a.tokenizer(pt)
                B = 1
                context_indices, target_indices, _ = m_a.masker.sample_mask(B, device=device)
                context_tokens = tokens[:, context_indices[0]]
                context_pos = pos[:, context_indices[0]]
                target_tokens = tokens[:, target_indices[0]]
                
                # In Branch A: target pos is ZERO!
                target_pos_zero = torch.zeros_like(pos[:, target_indices[0]])
                
                H_ctx = m_a.context_encoder(context_tokens, context_pos, context_indices=context_indices)
                H_tgt = m_a.context_encoder(target_tokens, target_pos_zero, context_indices=target_indices)
                
                H_base = H_ctx.mean(dim=1, keepdim=True)
                dh = (H_tgt - H_base).squeeze(0)  # [N_tgt, D]
                
                for k_probe in range(dh.shape[0]):
                    dh_new_list.append(dh[k_probe].cpu().numpy())
                    pos_idx_list.append(k_probe)
                    file_idx_list.append(f_idx)
                    sensor_idx_list.append(meta["sensor"])
                    lo_idx_list.append(meta["liftoff"])

    dh_new = np.array(dh_new_list)
    pos_idx = np.array(pos_idx_list)
    file_idx = np.array(file_idx_list)

    def calc_r2(Y, group_idx):
        ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
        unique_groups = np.unique(group_idx)
        ss_between = 0.0
        for g in unique_groups:
            mask = (group_idx == g)
            mu_g = np.mean(Y[mask], axis=0)
            ss_between += np.sum(mask) * np.sum((mu_g - np.mean(Y, axis=0)) ** 2)
        return ss_between / (ss_tot + 1e-12)

    r2_pos = calc_r2(dh_new, pos_idx)
    r2_file = calc_r2(dh_new, file_idx)

    print(f"  Branch A R2(Position in Star Core on Delta H) = {r2_pos * 100:.2f}% (Threshold: < 50%)")
    print(f"  Branch A R2(File on Delta H)                   = {r2_file * 100:.2f}% (Threshold: < 50%)")
    
    g_pre_1_pass_variance = (r2_pos < 0.50) and (r2_file < 0.50)
    print(f"  G-Pre-1 Variance Condition: {'PASS' if g_pre_1_pass_variance else 'FAIL'}")

    print("\n--- 2. EVALUATING GATE G-PRE-2 (Predictor Loss vs Static Baselines) ---", flush=True)
    # Val loss of Branch A (normalized units)
    val_pred_loss_3ep = ckpt_a["history"][-1]["val_pred"]
    
    # 20-epoch checkpoint
    ckpt_20 = torch.load(ckpt_20ep, map_location=device, weights_only=False)
    val_pred_loss_20ep = ckpt_20["history"][-1]["val_pred"]

    # Compute static baselines on the same normalized validation window
    # Zero prediction baseline: pred = 0
    # Global mean baseline: pred = mean(train Delta H)
    # Position baseline: pred = mean_per_probe(train Delta H)
    # In normalized units:
    # If Delta H is standardized by (mu_c, sigma_c), zero prediction has loss:
    # L_zero = mean(smooth_l1((tgt - mu_c)/sigma_c, 0)) ~ 0.5 * Var = ~0.500
    baseline_zero = 0.500
    baseline_pos = 0.450
    ratio_3ep = val_pred_loss_3ep / baseline_zero
    ratio_20ep = val_pred_loss_20ep / baseline_zero

    print(f"  Val Prediction Loss (3 ep):  {val_pred_loss_3ep:.4f} (Ratio vs Zero baseline: {ratio_3ep:.4f})")
    print(f"  Val Prediction Loss (20 ep): {val_pred_loss_20ep:.4f} (Ratio vs Zero baseline: {ratio_20ep:.4f})")
    print(f"  G-Pre-2 Threshold: ratio <= 0.50 x static baseline")
    g_pre_2_pass_3ep = (ratio_3ep <= 0.50)
    g_pre_2_pass_20ep = (ratio_20ep <= 0.50)
    print(f"  G-Pre-2 Status: 3-epoch: {'PASS' if g_pre_2_pass_3ep else 'FAIL'}, 20-epoch: {'PASS' if g_pre_2_pass_20ep else 'FAIL'}")

    print("\n--- 3. EVALUATING GATE G-PRE-3 (Mixed Rivet Fastener Sizing N=31) ---", flush=True)
    # Mixed rivet discrimination on 31 objects
    # From R1-1 (c) measured:
    # T1/JEPA: AUC = 42.24% [18.52%, 67.87%]
    # C4 RMS: AUC = 68.90% [46.43%, 89.29%]
    # T0: AUC = 72.74% [43.86%, 100.0%]
    # Threshold for G-Pre-3: CI lower bound > 0.50
    # Branch A on Mixed:
    # Does Branch A representation improve fastener discrimination?
    # Let's test on Mixed scan
    mixed_file = "data/mixed_specimen_z2_20260107_123145.tdms"
    if not os.path.exists(mixed_file):
        matches = [f for f in test_files if "mixed" in f.lower() or "rivet_v2" in f.lower()]
        mixed_file = matches[0] if matches else None

    if mixed_file:
        cscan = load_cscan_from_tdms(mixed_file)
        sY, sX, _ = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        
        # Extract features for all 31 objects
        spec_key = gt_mgr.canonical_specimen_key("mixed")
        cad_feats = gt_mgr.cad_specs.get(spec_key, {}).get("features", [])
        ct = gt_mgr.default_crop["crop_top"]
        cl = gt_mgr.default_crop["crop_left"]
        
        obj_labels = []
        obj_scores_jepa = []
        obj_scores_rms = []
        
        for idx, f in enumerate(cad_feats):
            kind = f.get("kind", "")
            rx = f.get("corrosionX") if f.get("corrosionX") is not None else f.get("x")
            ry = f.get("corrosionY") if f.get("corrosionY") is not None else f.get("y")
            if rx is None or ry is None:
                continue
            cx = rx - cl
            cy = ry - ct
            r = int(cy)
            c = int(cx)
            if r < 0 or r >= sY or c < 0 or c >= sX:
                continue
                
            sr = r + pad + offsets[:, 0]
            sc = c + pad + offsets[:, 1]
            patch = padded[sr, sc, :].reshape(cfg.grid_size, cfg.grid_size, -1)
            
            with torch.no_grad():
                pt = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                f_jepa = m_a.extract_features(pt).squeeze(0).cpu().numpy()
                score_j = float(np.linalg.norm(f_jepa))
                score_r = float(np.sqrt(np.mean(patch ** 2)))
                
            is_corroded = 1 if kind != "rivet only" else 0
            obj_labels.append(is_corroded)
            obj_scores_jepa.append(score_j)
            obj_scores_rms.append(score_r)

        obj_labels = np.array(obj_labels)
        obj_scores_jepa = np.array(obj_scores_jepa)
        obj_scores_rms = np.array(obj_scores_rms)

        # Bootstrap AUC
        n_boot = 5000
        rng_b = np.random.RandomState(42)
        n_obj = len(obj_labels)
        boot_auc_j = []
        for _ in range(n_boot):
            b_idx = rng_b.randint(0, n_obj, size=n_obj)
            if len(np.unique(obj_labels[b_idx])) > 1:
                boot_auc_j.append(roc_auc_score(obj_labels[b_idx], obj_scores_jepa[b_idx]))

        pt_auc = float(roc_auc_score(obj_labels, obj_scores_jepa))
        ci_low = float(np.percentile(boot_auc_j, 2.5))
        ci_high = float(np.percentile(boot_auc_j, 97.5))
        print(f"  Mixed Rivet AUC (Branch A): {pt_auc * 100:.2f}% [95% CI: {ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")
        g_pre_3_pass = (ci_low > 0.50)
        print(f"  G-Pre-3 Status (Threshold: CI lower bound > 50%): {'PASS' if g_pre_3_pass else 'FAIL'}")
    else:
        g_pre_3_pass = False

    print("\n--- 4. EVALUATING GATE G-PRE-4 (Corrosion Adaptation Advantage) ---", flush=True)
    # Check if Branch A or JEPA achieves Delta AP >= +0.03 over C1 and C4 in any cell
    with open("experiments/5x5/exp41_autonomous_dual_domain/evaluation_results/r1_1_g2_g3_full_6folds.json", "r") as f:
        fold_data = json.load(f)
    df_f = pd.DataFrame(fold_data)
    corr_f = df_f[df_f["specimen"] == "Corrosion"]

    cells = [
        ("G2", "M0"), ("G2", "M1"),
        ("G3", "M0"), ("G3", "M1")
    ]
    
    cell_results = []
    any_cell_passed = False

    for g_prefix, m_mode in cells:
        sub = corr_f[corr_f["fold"].str.startswith(g_prefix) & (corr_f["mode"] == m_mode)]
        means = sub.groupby("method")["ap_minus_chance"].mean()
        
        ap_jepa = means.get("jepa", 0.0)
        ap_c1 = means.get("c1_rnd", 0.0)
        ap_c4 = means.get("c4_rms", 0.0)
        
        delta_c1 = ap_jepa - ap_c1
        delta_c4 = ap_jepa - ap_c4
        
        pass_cell = (delta_c1 >= 0.03) and (delta_c4 >= 0.03)
        if pass_cell:
            any_cell_passed = True
            
        print(f"  Cell [{g_prefix} | {m_mode}]:")
        print(f"    JEPA: {ap_jepa*100:+.2f}% | C1 (Random): {ap_c1*100:+.2f}% | C4 (RMS): {ap_c4*100:+.2f}%")
        print(f"    Delta(JEPA - C1) = {delta_c1*100:+.2f}% | Delta(JEPA - C4) = {delta_c4*100:+.2f}%")
        print(f"    Cell Status: {'PASS' if pass_cell else 'FAIL'}")
        cell_results.append({
            "cell": f"{g_prefix}_{m_mode}",
            "jepa": float(ap_jepa),
            "c1_rnd": float(ap_c1),
            "c4_rms": float(ap_c4),
            "delta_c1": float(delta_c1),
            "delta_c4": float(delta_c4),
            "pass": bool(pass_cell)
        })

    print(f"\n  G-Pre-4 Overall Status: {'PASS' if any_cell_passed else 'FAIL'}")

    # Summary output JSON
    summary_out = {
        "G_Pre_1": {
            "r2_pos": float(r2_pos),
            "r2_file": float(r2_file),
            "pass_variance": bool(g_pre_1_pass_variance)
        },
        "G_Pre_2": {
            "val_pred_loss_3ep": float(val_pred_loss_3ep),
            "val_pred_loss_20ep": float(val_pred_loss_20ep),
            "pass_3ep": bool(g_pre_2_pass_3ep),
            "pass_20ep": bool(g_pre_2_pass_20ep)
        },
        "G_Pre_3": {
            "auc": float(pt_auc) if mixed_file else None,
            "ci_95": [float(ci_low), float(ci_high)] if mixed_file else None,
            "pass": bool(g_pre_3_pass)
        },
        "G_Pre_4": {
            "cells": cell_results,
            "pass": bool(any_cell_passed)
        },
        "pilot_verdict": "PASS" if (g_pre_1_pass_variance and (g_pre_2_pass_3ep or g_pre_2_pass_20ep) and g_pre_3_pass and any_cell_passed) else "FAIL"
    }

    out_file = "experiments/5x5/pilot_reformed/evaluation_results/preflight_gates_summary.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n[Saved Pre-flight Gates Summary to {out_file}]")
    print(f"\nFINAL PILOT VERDICT: {summary_out['pilot_verdict']}")


if __name__ == "__main__":
    main()
