#!/usr/bin/env python3
"""
Forensic Evaluation of Fastener Confounding on Specimen Mixed (Rivet_v2) for EXP-41.
Compares EXP-41 (Autonomous Dual-Domain) against Raw Energy and Random Untrained Encoder.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath("."))
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms, extract_full_cscan_map
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.calibration import compute_m1_mahalanobis_score, extract_scan_sound_reference


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"

    config = Spatiotemporal5x5Config.from_json(cfg_path)

    # 1. Trained EXP-41 Model
    trained_model = PECT_JEPA_5x5(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(ckpt["model_state_dict"])
    trained_model.to(device).eval()

    # 2. Random Untrained Model (seed 999)
    torch.manual_seed(999)
    random_model = PECT_JEPA_5x5(config)
    random_model.to(device).eval()

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    gt_mgr = get_ground_truth_manager("data")

    # Filter all 19 Mixed held-out test files
    mixed_test_files = [
        f for f in splits["test_files"]
        if "mixed" in f.lower() or "rivet_v2" in f.lower() or "rivet2" in f.lower()
    ]
    print(f"Found {len(mixed_test_files)} Mixed (Rivet_v2) held-out test files.")

    multi_mask = gt_mgr.generate_multiclass_mask("mixed", buffer_px=2)
    H, W = multi_mask.shape

    cad_mixed = gt_mgr.cad_specs.get("mixed", {}).get("features", [])
    clean_features = [f for f in cad_mixed if f.get("kind") == "rivet only"]
    corroded_features = [f for f in cad_mixed if f.get("kind") != "rivet only"]
    print(f"CAD specs: {len(clean_features)} clean rivets, {len(corroded_features)} corroded rivets.")

    records = []
    object_records = []

    ct = gt_mgr.default_crop["crop_top"]
    cl = gt_mgr.default_crop["crop_left"]
    Y_grid, X_grid = np.ogrid[:H, :W]

    for fpath in tqdm(mixed_test_files, desc="Evaluating EXP-41 on Mixed files"):
        bname = os.path.basename(fpath)
        std_cscan = load_cscan_from_tdms(fpath)

        # 1. Raw RMS Energy
        score_raw = np.sqrt(np.mean(std_cscan ** 2, axis=-1))

        # 2. Random Encoder M1 Mahalanobis
        fmap_rnd = extract_full_cscan_map(random_model, std_cscan, batch_size=2048, device=device)
        score_rnd = compute_m1_mahalanobis_score(fmap_rnd)

        # 3. Trained EXP-41 M1 Mahalanobis
        fmap_tr = extract_full_cscan_map(trained_model, std_cscan, batch_size=2048, device=device)
        score_tr = compute_m1_mahalanobis_score(fmap_tr)

        methods = {
            "Raw Energy": score_raw,
            "Random Enc": score_rnd,
            "EXP-41 JEPA": score_tr,
        }

        mask_flat = multi_mask.ravel()

        for mname, s_map in methods.items():
            s_flat = s_map.ravel()

            # A) Defect vs Sound
            valid_ds = (mask_flat == 0) | (mask_flat == 3)
            y_ds = (mask_flat[valid_ds] == 3).astype(int)
            auc_ds = roc_auc_score(y_ds, s_flat[valid_ds]) if len(np.unique(y_ds)) > 1 else np.nan

            # B) Clean Rivet vs Sound
            valid_cs = (mask_flat == 0) | (mask_flat == 2)
            y_cs = (mask_flat[valid_cs] == 2).astype(int)
            auc_cs = roc_auc_score(y_cs, s_flat[valid_cs]) if len(np.unique(y_cs)) > 1 else np.nan

            # C) Defect vs Clean Rivet
            valid_dc = (mask_flat == 2) | (mask_flat == 3)
            y_dc = (mask_flat[valid_dc] == 3).astype(int)
            auc_dc = roc_auc_score(y_dc, s_flat[valid_dc]) if len(np.unique(y_dc)) > 1 else np.nan

            records.append({
                "file": bname,
                "method": mname,
                "auc_defect_vs_sound": auc_ds,
                "auc_clean_vs_sound": auc_cs,
                "auc_defect_vs_clean": auc_dc,
            })

            # --- Object-Level Fastener Discrimination ---
            obj_labels = []
            obj_scores = []

            for feat in cad_mixed:
                rx = feat.get("x")
                ry = feat.get("y")
                rd = feat.get("rivetDiameter", 13.0)
                is_defective = (feat.get("kind") != "rivet only")
                if rx is None or ry is None:
                    continue
                cx = rx - cl
                cy = ry - ct
                r_disk = (rd / 2.0)
                disk_mask = ((X_grid - cx) ** 2 + (Y_grid - cy) ** 2) <= (r_disk ** 2)
                if np.sum(disk_mask) > 0:
                    mean_val = np.mean(s_map[disk_mask])
                    obj_labels.append(1 if is_defective else 0)
                    obj_scores.append(mean_val)

            obj_labels = np.array(obj_labels)
            obj_scores = np.array(obj_scores)
            if len(np.unique(obj_labels)) > 1:
                obj_auc = roc_auc_score(obj_labels, obj_scores)
            else:
                obj_auc = np.nan

            object_records.append({
                "file": bname,
                "method": mname,
                "object_auc_defect_vs_clean": obj_auc,
            })

    df_pix = pd.DataFrame(records)
    df_obj = pd.DataFrame(object_records)

    print("\n" + "=" * 110)
    print("EXP-41 FASTENER EVALUATION ON ALL 19 MIXED (RIVET_V2) TEST FILES (MEAN +/- STD)")
    print("=" * 110)
    summary_pix = df_pix.groupby("method").agg({
        "auc_defect_vs_sound": ["mean", "std"],
        "auc_clean_vs_sound": ["mean", "std"],
        "auc_defect_vs_clean": ["mean", "std"],
    })
    print(summary_pix.to_string())

    print("\n" + "=" * 110)
    print("OBJECT-LEVEL FASTENER DISCRIMINATION (25 CORRODED RIVETS vs 6 CLEAN RIVETS)")
    print("=" * 110)
    summary_obj = df_obj.groupby("method").agg({
        "object_auc_defect_vs_clean": ["mean", "std"],
    })
    print(summary_obj.to_string())


if __name__ == "__main__":
    main()
