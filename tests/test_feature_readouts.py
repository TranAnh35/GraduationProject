import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager

def test_feature_readouts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "experiments/5x5/exp10_residual_diff_20260925_010642/checkpoints/best_model_5x5.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        config = Spatiotemporal5x5Config(**cfg_raw)
    else:
        config = cfg_raw
    model = PECT_JEPA_5x5(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Pick 2 representative test files: one Rivet, one Corrosion
    test_files = [
        ("data/Hall_Air_Core/Rivet_v1/Chirp/hall_aircore_rivet_frontside_chirp_300x300_500_1500hz_z1_20260123_202334.tdms", "rivet_v1"),
        ("data/Hall_Air_Core/Corrosion/Chirp/hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z1_20260118_162411.tdms", "corrosion"),
    ]

    gt_mgr = get_ground_truth_manager()

    for fp, spec in test_files:
        print(f"\n==================================================")
        print(f"Testing File: {os.path.basename(fp)} ({spec})")
        print(f"==================================================")

        cscan = load_cscan_from_tdms(fp, resample_mode="linear", temporal_samples=128, crop_border=15)
        sY, sX, C = cscan.shape
        pad = 2
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

        mask = gt_mgr.get_ground_truth_mask_for_file(fp, aligned_scan=True)
        depth = gt_mgr.generate_depth_map(spec)
        if mask is None or depth is None:
            print(f"Skipping {fp}: no ground truth")
            continue
        min_Y = min(sY, mask.shape[0])
        min_X = min(sX, mask.shape[1])
        mask = mask[:min_Y, :min_X].astype(int)
        depth = depth[:min_Y, :min_X].astype(float)
        sY, sX = min_Y, min_X

        # Extract all 4 temporal stages for the center probe of each 5x5 patch
        # Token 48: tau=0, 49: tau=1, 50: tau=2, 51: tau=3
        batch_size = 512
        patches = []
        coords = []
        all_H = np.zeros((sY, sX, 4, 64), dtype=np.float32)

        with torch.no_grad():
            for i in range(sY):
                for j in range(sX):
                    sub = padded[i:i+5, j:j+5, :]
                    patches.append(sub)
                    coords.append((i, j))
                    if len(patches) >= batch_size:
                        xb = torch.from_numpy(np.stack(patches)).float().to(device)
                        tokens, pos = model.tokenizer(xb)
                        H = model.context_encoder(tokens, pos) # [B, 100, 64]
                        # Extract center probe's 4 stages: [B, 4, 64]
                        H_center = H[:, 48:52, :].cpu().numpy()
                        for (ci, cj), h_c in zip(coords, H_center):
                            all_H[ci, cj] = h_c
                        patches, coords = [], []
            if patches:
                xb = torch.from_numpy(np.stack(patches)).float().to(device)
                tokens, pos = model.tokenizer(xb)
                H = model.context_encoder(tokens, pos)
                H_center = H[:, 48:52, :].cpu().numpy()
                for (ci, cj), h_c in zip(coords, H_center):
                    all_H[ci, cj] = h_c

        # Define 4 different readout representations:
        readouts = {
            "1. Current Baseline (Mean over 4 stages, 64-D)": all_H.mean(axis=2).reshape(-1, 64),
            "2. Concatenated Stages [tau0, tau1, tau2, tau3] (256-D)": all_H.reshape(-1, 256),
            "3. Deepest Stage Only [tau=3] (64-D)": all_H[:, :, 3, :].reshape(-1, 64),
            "4. Differential Diffusion Perturbation [tau1-0, tau2-0, tau3-0] (192-D)": (all_H[:, :, 1:, :] - all_H[:, :, :1, :]).reshape(-1, 192),
        }

        flat_y_bin = mask.reshape(-1)
        flat_y_depth = depth.reshape(-1)

        # Subsample for fast benchmark
        pos_idx = np.where(flat_y_bin == 1)[0]
        neg_idx = np.where(flat_y_bin == 0)[0]
        rng = np.random.RandomState(42)
        sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx)*5), replace=False)
        eval_idx = np.sort(np.concatenate([pos_idx, sub_neg]))

        y_bin = flat_y_bin[eval_idx]
        y_depth = flat_y_depth[eval_idx]

        # Also isolate defect-only samples for depth regression
        def_eval_idx = pos_idx
        y_def_depth = flat_y_depth[def_eval_idx]

        for name, feats in readouts.items():
            X = feats[eval_idx]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            # Task 1: Linear Probe AUC & AP
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs, aps = [], []
            for tr, te in skf.split(X_s, y_bin):
                lr = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
                lr.fit(X_s[tr], y_bin[tr])
                p = lr.predict_proba(X_s[te])[:, 1]
                aucs.append(roc_auc_score(y_bin[te], p))
                aps.append(average_precision_score(y_bin[te], p))

            # Task 2: Depth Regression R^2 on plate
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2s_plate = []
            for tr, te in kf.split(X_s):
                reg = Ridge(alpha=1.0)
                reg.fit(X_s[tr], y_depth[tr])
                pred = reg.predict(X_s[te])
                r2s_plate.append(r2_score(y_depth[te], pred))

            # Task 2b: Defect-Only Depth Regression R^2 (y > 0)
            X_def = feats[def_eval_idx]
            X_def_s = StandardScaler().fit_transform(X_def)
            r2s_def = []
            for tr, te in kf.split(X_def_s):
                reg = Ridge(alpha=1.0)
                reg.fit(X_def_s[tr], y_def_depth[tr])
                pred = reg.predict(X_def_s[te])
                r2s_def.append(r2_score(y_def_depth[te], pred))

            print(f"--- {name} ---")
            print(f"  Plate AUC: {np.mean(aucs)*100:.2f}% | AP: {np.mean(aps)*100:.2f}%")
            print(f"  Plate Depth R^2: {np.mean(r2s_plate):.4f}")
            print(f"  Defect-Only Depth R^2 (y > 0): {np.mean(r2s_def):.4f}")

if __name__ == "__main__":
    test_feature_readouts()
