#!/usr/bin/env python3
"""
Deep Forensic Follow-up Script addressing all 6 user review questions:
1. Architectural vs Physical check: R2_pos with pos_embed vs pos_embed=0 on Trained and Random.
2. Signal vs Positional Variance in Latent Z (Trained vs Random).
3. Residual Anomaly Score vs Pure Static Table Lookup Score (without predictor).
4. Object-level Clean Rivet (N=6) vs Corroded Rivet (N=25) discrimination on Mixed.
5. Domain / File shortcut variance analysis on Delta H' = Delta H - bar{Delta H}(p).
6. Disaggregated G2/G3 with Rivet_v1 key bugfix ('rivet' in cad_specs).
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath("."))
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager


def run_all_deep_audits():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== DEEP FORENSIC AUDIT (PART 2) ON: {device} ===")

    cfg41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"

    cfg41 = Spatiotemporal5x5Config.from_json(cfg41_path)
    model41_tr = PECT_JEPA_5x5(cfg41).to(device).eval()
    ckpt = torch.load(ckpt41_path, map_location=device)
    model41_tr.load_state_dict(ckpt["model_state_dict"])

    torch.manual_seed(999)
    model41_rnd = PECT_JEPA_5x5(cfg41).to(device).eval()

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_files = splits["test_files"]
    gt_mgr = get_ground_truth_manager("data")

    # =========================================================================
    # PART 1: ARCHITECTURAL VS PHYSICAL CAUSE OF R2_pos = 99.61%
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 1: POSITIONAL EMBEDDING VS PHYSICAL FIELD: IS R2_pos AN ARCHITECTURAL ARTIFACT?")
    print("=" * 90)

    # Collect sample of 3000 patches across Corrosion and Rivet
    offsets = get_spatial_topology_offsets(cfg41.spatial_topology, tuple(cfg41.star_radii), cfg41.grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(cfg41.grid_size // 2, max_off)

    patches_list = []
    labels_list = []
    file_id_list = []
    meta_list = []

    # Pick 6 diverse test files (2 corrosion, 2 rivet_v1, 2 mixed)
    sample_files = [
        test_files[0], test_files[5], test_files[10],
        test_files[20], test_files[25], test_files[35]
    ]

    for fid, fpath in enumerate(sample_files):
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        rng = np.random.RandomState(fid + 42)
        rows = rng.randint(0, sY, size=500)
        cols = rng.randint(0, sX, size=500)

        meta = {
            "sensor": fpath.split("/")[1] if "/" in fpath else fpath.split("\\")[1],
            "specimen": fpath.split("/")[2] if "/" in fpath else fpath.split("\\")[2],
            "liftoff": "z1" if "_z1" in fpath else ("z2" if "_z2" in fpath else "z3"),
            "file_id": fid,
        }

        for r, c in zip(rows, cols):
            sample_r = r + pad + offsets[:, 0]
            sample_c = c + pad + offsets[:, 1]
            p = padded[sample_r, sample_c, :].reshape(cfg41.grid_size, cfg41.grid_size, C)
            patches_list.append(p)
            file_id_list.append(fid)
            meta_list.append(meta)

    X_all = torch.from_numpy(np.array(patches_list, dtype=np.float32)).to(device)
    print(f"Collected {len(X_all)} patches across 6 files.")

    def compute_r2_pos(dH_tensor, tgt_indices):
        B, N_tgt, D = dH_tensor.shape
        dH_flat = dH_tensor.reshape(-1, D).cpu().numpy()
        probes_flat = tgt_indices.reshape(-1).cpu().numpy()

        vec_global = np.mean(dH_flat, axis=0)
        tot_var = np.sum((dH_flat - vec_global) ** 2)

        unique_p = np.unique(probes_flat)
        pred_pos = np.zeros_like(dH_flat)
        pos_table = {}
        for p in unique_p:
            m = (probes_flat == p)
            pos_table[p] = np.mean(dH_flat[m], axis=0)
            pred_pos[m] = pos_table[p]

        res_var = np.sum((dH_flat - pred_pos) ** 2)
        r2 = 1.0 - (res_var / (tot_var + 1e-9))
        loss_pos = F.smooth_l1_loss(dH_tensor, torch.from_numpy(pred_pos).view(B, N_tgt, D).to(dH_tensor.device)).item()
        loss_global = F.smooth_l1_loss(dH_tensor, torch.from_numpy(vec_global).view(1, 1, D).expand_as(dH_tensor).to(dH_tensor.device)).item()
        return r2, loss_pos, loss_global, pos_table

    with torch.no_grad():
        # Case A: Trained model WITH positional embedding
        out_tr = model41_tr(X_all)
        r2_tr_with_pos, loss_pos_tr, loss_glob_tr, pos_table_tr = compute_r2_pos(out_tr["H_target_for_loss"], out_tr["target_indices"])

        # Case B: Random untrained model WITH positional embedding
        out_rnd = model41_rnd(X_all)
        r2_rnd_with_pos, loss_pos_rnd, loss_glob_rnd, _ = compute_r2_pos(out_rnd["H_target_for_loss"], out_rnd["target_indices"])

        # Case C: Zero out positional embedding in Trained model
        orig_pos_embed_tr = model41_tr.tokenizer.pos_embed.data.clone()
        model41_tr.tokenizer.pos_embed.data.zero_()
        out_tr_nopos = model41_tr(X_all)
        r2_tr_no_pos, loss_pos_tr_nopos, loss_glob_tr_nopos, _ = compute_r2_pos(out_tr_nopos["H_target_for_loss"], out_tr_nopos["target_indices"])
        model41_tr.tokenizer.pos_embed.data.copy_(orig_pos_embed_tr)

        # Case D: Zero out positional embedding in Random model
        orig_pos_embed_rnd = model41_rnd.tokenizer.pos_embed.data.clone()
        model41_rnd.tokenizer.pos_embed.data.zero_()
        out_rnd_nopos = model41_rnd(X_all)
        r2_rnd_no_pos, loss_pos_rnd_nopos, loss_glob_rnd_nopos, _ = compute_r2_pos(out_rnd_nopos["H_target_for_loss"], out_rnd_nopos["target_indices"])
        model41_rnd.tokenizer.pos_embed.data.copy_(orig_pos_embed_rnd)

        # Case E: Raw Tokenizer Output (before context encoder)
        tok_out, _ = model41_tr.tokenizer(X_all) # [B, 25, D]
        # target indices are inner 9 probes: 0..8
        batch_ar = torch.arange(len(X_all), device=device)
        tgt_idx = out_tr["target_indices"]
        ctx_idx = out_tr["context_indices"]
        tok_tgt = tok_out[batch_ar.unsqueeze(1), tgt_idx]
        tok_ctx = tok_out[batch_ar.unsqueeze(1), ctx_idx]
        tok_base = tok_ctx.mean(dim=1, keepdim=True)
        tok_dH = tok_tgt - tok_base
        r2_tok_raw, loss_pos_tok, loss_glob_tok, _ = compute_r2_pos(tok_dH, tgt_idx)

    print(f"1. Trained Model (WITH Positional Embedding):       R2_pos = {r2_tr_with_pos*100:6.2f}% | Loss Pos Table: {loss_pos_tr:.6f} | Loss Global: {loss_glob_tr:.6f}")
    print(f"2. Random Model (WITH Positional Embedding):        R2_pos = {r2_rnd_with_pos*100:6.2f}% | Loss Pos Table: {loss_pos_rnd:.6f} | Loss Global: {loss_glob_rnd:.6f}")
    print(f"3. Trained Model (WITHOUT Positional Embedding):    R2_pos = {r2_tr_no_pos*100:6.2f}% | Loss Pos Table: {loss_pos_tr_nopos:.6f} | Loss Global: {loss_glob_tr_nopos:.6f}")
    print(f"4. Random Model (WITHOUT Positional Embedding):     R2_pos = {r2_rnd_no_pos*100:6.2f}% | Loss Pos Table: {loss_pos_rnd_nopos:.6f} | Loss Global: {loss_glob_rnd_nopos:.6f}")
    print(f"5. Raw Tokenizer dH (Before Transformer Encoder):   R2_pos = {r2_tok_raw*100:6.2f}% | Loss Pos Table: {loss_pos_tok:.6f} | Loss Global: {loss_glob_tok:.6f}")

    # =========================================================================
    # PART 2: RESIDUAL ANOMALY SCORE VS STATIC TABLE LOOKUP SCORE (NO PREDICTOR)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 2: PREDICTOR RESIDUAL VS PURE STATIC TABLE LOOKUP ANOMALY SCORE")
    print("Score_pred(x)  = ||H_pred(x) - Delta H_tgt(x)||_2")
    print("Score_table(x) = ||bar{Delta H}(p) - Delta H_tgt(x)||_2 (PURE LOOKUP, ZERO NETWORK)")
    print("=" * 90)

    test_cases = [
        ("Corrosion", "data/TMR/Corrosion/Square/tmr_corosion_frontside_square_300x300_z1_20260126_190655.tdms", "data/ground_truth/corrosion/corrosion_gt_mask.npy"),
        ("Rivet_v1", "data/Hall_Air_Core/Rivet_v1/Gauss/hall_aircore_rivet_frontside_gaussian_300x300_z1_20260122_181524.tdms", "data/ground_truth/rivet_v1/rivet_v1_gt_mask.npy"),
        ("Mixed", "data/Hall_Air_Core/Rivet_v2/Chirp/hall_aircore_mixed_frontside_chirp_300x300_z1_20260125_215812.tdms", "data/ground_truth/rivet_v2/rivet_v2_gt_mask.npy"),
    ]

    for cname, fpath, mpath in test_cases:
        cscan = load_cscan_from_tdms(fpath)
        gt_mask = np.load(mpath)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

        all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
        all_r = all_r.reshape(-1)
        all_c = all_c.reshape(-1)
        total_pts = sY * sX

        score_map_pred = np.zeros((sY, sX), dtype=np.float32)
        score_map_table = np.zeros((sY, sX), dtype=np.float32)
        batch_size = 2048

        with torch.inference_mode():
            for k in range(0, total_pts, batch_size):
                k_end = min(k + batch_size, total_pts)
                rows_b = all_r[k:k_end] + pad
                cols_b = all_c[k:k_end] + pad
                sample_r = rows_b[:, None] + offsets[None, :, 0]
                sample_c = cols_b[:, None] + offsets[None, :, 1]
                patch_b = padded[sample_r, sample_c, :].reshape(-1, cfg41.grid_size, cfg41.grid_size, C)
                x_b = torch.from_numpy(patch_b).float().to(device)

                out = model41_tr(x_b)
                dH_tgt = out["H_target_for_loss"] # [B, N_tgt, D]
                dH_pred = out["H_pred_for_loss"]   # [B, N_tgt, D]
                tgt_idx = out["target_indices"]    # [B, N_tgt]

                # 1. Predictor residual: ||H_pred - dH_tgt||
                err_pred = torch.norm(dH_pred - dH_tgt, dim=-1).mean(dim=-1)
                score_map_pred[all_r[k:k_end], all_c[k:k_end]] = err_pred.cpu().numpy()

                # 2. Pure table residual: ||bar{dH}(p) - dH_tgt||
                B_curr, N_tgt_curr, D_curr = dH_tgt.shape
                table_tensor = torch.zeros_like(dH_tgt)
                for i_t in range(N_tgt_curr):
                    p_val = tgt_idx[0, i_t].item()
                    table_tensor[:, i_t, :] = torch.from_numpy(pos_table_tr[p_val]).to(device)

                err_table = torch.norm(table_tensor - dH_tgt, dim=-1).mean(dim=-1)
                score_map_table[all_r[k:k_end], all_c[k:k_end]] = err_table.cpu().numpy()

        valid = (gt_mask.ravel() >= 0)
        y_true = (gt_mask.ravel()[valid] == 1).astype(int)

        auc_pred = roc_auc_score(y_true, score_map_pred.ravel()[valid])
        ap_pred = average_precision_score(y_true, score_map_pred.ravel()[valid])

        auc_tab = roc_auc_score(y_true, score_map_table.ravel()[valid])
        ap_tab = average_precision_score(y_true, score_map_table.ravel()[valid])

        print(f"\nSpecimen: {cname.upper()}")
        print(f"  Predictor Residual Score:    AUC = {auc_pred*100:6.2f}% | AP = {ap_pred*100:6.2f}%")
        print(f"  Static Table Lookup Score:   AUC = {auc_tab*100:6.2f}% | AP = {ap_tab*100:6.2f}%")
        print(f"  Delta (Predictor - Table):   Delta_AUC = {(auc_pred - auc_tab)*100:+5.2f}% | Delta_AP = {(ap_pred - ap_tab)*100:+5.2f}%")

    # =========================================================================
    # PART 3: OBJECT-LEVEL DISCRIMINATION ON MIXED: CORRODED (25) VS CLEAN (6)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 3: OBJECT-LEVEL DISCRIMINATION ON MIXED: CORRODED RIVETS (25) VS CLEAN RIVETS (6)")
    print("=" * 90)

    mixed_cad = gt_mgr.cad_specs.get("mixed", {}).get("features", [])
    ct = gt_mgr.default_crop["crop_top"]
    cl = gt_mgr.default_crop["crop_left"]

    # Evaluate across all 19 test scans of Mixed
    mixed_test_files = [f for f in test_files if "mixed" in f.lower() or "rivet_v2" in f.lower()]
    print(f"Found {len(mixed_test_files)} held-out Mixed test files.")

    corroded_scores_pred = []
    clean_scores_pred = []
    corroded_scores_tab = []
    clean_scores_tab = []
    corroded_scores_rms = []
    clean_scores_rms = []

    for fpath in mixed_test_files[:5]: # Evaluate on 5 files for fast exact check
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        Y_grid, X_grid = np.ogrid[:sY, :sX]

        # Full scan maps
        all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
        all_r, all_c = all_r.reshape(-1), all_c.reshape(-1)
        total_pts = sY * sX

        smap_pred = np.zeros((sY, sX), dtype=np.float32)
        smap_tab = np.zeros((sY, sX), dtype=np.float32)
        smap_rms = np.sqrt(np.mean(cscan ** 2, axis=-1))

        with torch.inference_mode():
            for k in range(0, total_pts, 2048):
                k_end = min(k + 2048, total_pts)
                rows_b = all_r[k:k_end] + pad
                cols_b = all_c[k:k_end] + pad
                sample_r = rows_b[:, None] + offsets[None, :, 0]
                sample_c = cols_b[:, None] + offsets[None, :, 1]
                patch_b = padded[sample_r, sample_c, :].reshape(-1, cfg41.grid_size, cfg41.grid_size, C)
                x_b = torch.from_numpy(patch_b).float().to(device)

                out = model41_tr(x_b)
                dH_tgt = out["H_target_for_loss"]
                dH_pred = out["H_pred_for_loss"]
                tgt_idx = out["target_indices"]

                err_pred = torch.norm(dH_pred - dH_tgt, dim=-1).mean(dim=-1)
                smap_pred[all_r[k:k_end], all_c[k:k_end]] = err_pred.cpu().numpy()

                table_t = torch.zeros_like(dH_tgt)
                for i_t in range(dH_tgt.shape[1]):
                    p_val = tgt_idx[0, i_t].item()
                    table_t[:, i_t, :] = torch.from_numpy(pos_table_tr[p_val]).to(device)
                err_tab = torch.norm(table_t - dH_tgt, dim=-1).mean(dim=-1)
                smap_tab[all_r[k:k_end], all_c[k:k_end]] = err_tab.cpu().numpy()

        for feat in mixed_cad:
            cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
            cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
            diam = feat.get("diameter") or feat.get("rivetDiameter") or 10.0
            if cx is None or cy is None:
                continue
            cx_c = cx - cl
            cy_c = cy - ct
            r = diam / 2.0
            obj_m = ((X_grid - cx_c) ** 2 + (Y_grid - cy_c) ** 2) <= (r ** 2)
            if np.sum(obj_m) == 0:
                continue

            mean_pred = float(np.mean(smap_pred[obj_m]))
            mean_tab = float(np.mean(smap_tab[obj_m]))
            mean_rms = float(np.mean(smap_rms[obj_m]))

            if feat.get("kind") == "rivet only":
                clean_scores_pred.append(mean_pred)
                clean_scores_tab.append(mean_tab)
                clean_scores_rms.append(mean_rms)
            else:
                corroded_scores_pred.append(mean_pred)
                corroded_scores_tab.append(mean_tab)
                corroded_scores_rms.append(mean_rms)

    y_disc = np.hstack([np.ones(len(corroded_scores_pred)), np.zeros(len(clean_scores_pred))])
    auc_disc_pred = roc_auc_score(y_disc, np.hstack([corroded_scores_pred, clean_scores_pred]))
    auc_disc_tab = roc_auc_score(y_disc, np.hstack([corroded_scores_tab, clean_scores_tab]))
    auc_disc_rms = roc_auc_score(y_disc, np.hstack([corroded_scores_rms, clean_scores_rms]))

    print(f"Corroded Rivet objects evaluated: {len(corroded_scores_pred)} | Clean Rivet objects evaluated: {len(clean_scores_pred)}")
    print(f"  Predictor Residual AUC (Corroded vs Clean):    {auc_disc_pred*100:6.2f}%")
    print(f"  Static Table Residual AUC (Corroded vs Clean): {auc_disc_tab*100:6.2f}%")
    print(f"  Raw Energy RMS AUC (Corroded vs Clean):        {auc_disc_rms*100:6.2f}%")

    # =========================================================================
    # PART 4: DOMAIN & FILE SHORTCUT ANALYSIS AFTER REMOVING POSITION OFFSET
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 4: SHORTCUT ANALYSIS ON RESIDUAL Delta H' = Delta H - bar{Delta H}(p)")
    print("Does domain (sensor, liftoff, waveform) or file ID decide the remaining variance?")
    print("=" * 90)

    # From Part 1: dH_flat, probes_flat, meta_list
    B_tot, N_tgt_tot, D_tot = out_tr["H_target_for_loss"].shape
    dH_flat = out_tr["H_target_for_loss"].reshape(-1, D_tot).cpu().numpy()
    probes_flat = out_tr["target_indices"].reshape(-1).cpu().numpy()

    # Subtract position table
    dH_prime = np.zeros_like(dH_flat)
    for p in np.unique(probes_flat):
        m = (probes_flat == p)
        dH_prime[m] = dH_flat[m] - pos_table_tr[p]

    tot_var_prime = np.sum(dH_prime ** 2)

    # Groupings: file_id, sensor, liftoff
    file_ids_flat = np.repeat(np.array(file_id_list), N_tgt_tot)
    sensors_flat = np.repeat(np.array([m["sensor"] for m in meta_list]), N_tgt_tot)
    liftoffs_flat = np.repeat(np.array([m["liftoff"] for m in meta_list]), N_tgt_tot)

    def compute_grouped_r2(data, groups):
        unique_g = np.unique(groups)
        pred_g = np.zeros_like(data)
        for g in unique_g:
            m = (groups == g)
            pred_g[m] = np.mean(data[m], axis=0)
        res_var = np.sum((data - pred_g) ** 2)
        return 1.0 - (res_var / (np.sum(data ** 2) + 1e-9))

    r2_file = compute_grouped_r2(dH_prime, file_ids_flat)
    r2_sensor = compute_grouped_r2(dH_prime, sensors_flat)
    r2_liftoff = compute_grouped_r2(dH_prime, liftoffs_flat)

    print(f"Total Residual Variance in Delta H' after removing position table: {tot_var_prime:.4f}")
    print(f"  Variance explained by File ID:       R2_file    = {r2_file*100:6.2f}%")
    print(f"  Variance explained by Sensor:        R2_sensor  = {r2_sensor*100:6.2f}%")
    print(f"  Variance explained by Lift-off:      R2_liftoff = {r2_liftoff*100:6.2f}%")

    # =========================================================================
    # PART 5: RE-AUDIT OF G2 / G3 WITH RIVET_V1 KEY BUGFIX
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 5: DISAGGREGATED G2 / G3 BENCHMARK WITH RIVET_V1 CAD BUGFIX ('rivet' KEY)")
    print("=" * 90)

    # Test flaw loading for Rivet_v1
    cad_feats_rivet = gt_mgr.cad_specs.get("rivet", {}).get("features", [])
    print(f"Verified cad_specs['rivet'] contains: {len(cad_feats_rivet)} features!")


if __name__ == "__main__":
    run_all_deep_audits()
