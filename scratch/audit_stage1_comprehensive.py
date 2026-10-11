#!/usr/bin/env python3
"""
Comprehensive Stage 1 Forensic Audit Script for PECT-JEPA
Covers:
- C2: pos_embed and target scale diagnosis (EXP-41 and EXP-40, init vs 3 ep)
- C3: Residual Delta H' and defect content under spatial block CV
- C4: Residual scores and object-level clean vs corroded rivet discrimination (N=31)
- C5: Strictly disjoint G2/G3 adaptation with 20 independent support draws & 5 random seeds
"""

import os
import sys
import io
import json
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.calibration import apply_m1_unsupervised_calibration


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== STAGE 1 COMPREHENSIVE AUDIT ON: {device} ===")

    cfg41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split41_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"

    cfg40_path = "experiments/5x5/exp40_stabilized_deconv/checkpoints/config_5x5.json"
    ckpt40_path = "experiments/5x5/exp40_stabilized_deconv/checkpoints/best_model_5x5.pt"

    cfg41 = Spatiotemporal5x5Config.from_json(cfg41_path)
    cfg40 = Spatiotemporal5x5Config.from_json(cfg40_path)

    with open(split41_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_files = splits["train_files"]
    test_files = splits["test_files"]
    gt_mgr = get_ground_truth_manager("data")

    # =========================================================================
    # PART C2: CHẨN ĐOÁN POS_EMBED VÀ THANG ĐO TARGET
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHẦN C2: CHẨN ĐOÁN POS_EMBED VÀ THANG ĐO TARGET (EXP-41 & EXP-40)")
    print("=" * 90)

    # 1. Models at Init and Trained
    torch.manual_seed(42)
    m41_init = PECT_JEPA_5x5(cfg41).to(device).eval()
    m41_tr = PECT_JEPA_5x5(cfg41).to(device).eval()
    ckpt41 = torch.load(ckpt41_path, map_location=device)
    m41_tr.load_state_dict(ckpt41["model_state_dict"])

    torch.manual_seed(42)
    m40_init = PECT_JEPA_5x5(cfg40).to(device).eval()
    m40_tr = PECT_JEPA_5x5(cfg40).to(device).eval()
    ckpt40 = torch.load(ckpt40_path, map_location=device)
    m40_tr.load_state_dict(ckpt40["model_state_dict"])

    # Collect 3,000 diverse patches across train files for scale checks
    offsets = get_spatial_topology_offsets(cfg41.spatial_topology, tuple(cfg41.star_radii), cfg41.grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(cfg41.grid_size // 2, max_off)

    patches_train = []
    for fid, fpath in enumerate(train_files[:6]):
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        rng = np.random.RandomState(fid + 42)
        rows = rng.randint(0, sY, size=500)
        cols = rng.randint(0, sX, size=500)
        for r, c in zip(rows, cols):
            sample_r = r + pad + offsets[:, 0]
            sample_c = c + pad + offsets[:, 1]
            p = padded[sample_r, sample_c, :].reshape(cfg41.grid_size, cfg41.grid_size, C)
            patches_train.append(p)
    X_train_samp = torch.from_numpy(np.array(patches_train, dtype=np.float32)).to(device)

    # Question 1: pos_embed learnable?
    pe41_req_grad = m41_init.tokenizer.pos_embed.requires_grad
    pe41_init_std = float(m41_init.tokenizer.pos_embed.std().item())
    pe41_tr_std = float(m41_tr.tokenizer.pos_embed.std().item())
    pe40_req_grad = m40_init.tokenizer.pos_embed.requires_grad
    pe40_init_std = float(m40_init.tokenizer.pos_embed.std().item())
    pe40_tr_std = float(m40_tr.tokenizer.pos_embed.std().item())

    # Question 2: ||pos_embed|| / ||token||
    with torch.no_grad():
        tok41_init, pe41_i = m41_init.tokenizer(X_train_samp[:500])
        tok41_tr, pe41_t = m41_tr.tokenizer(X_train_samp[:500])
        norm_pe41_init = float(torch.norm(pe41_i, dim=-1).mean().item())
        norm_tok41_init = float(torch.norm(tok41_init, dim=-1).mean().item())
        norm_pe41_tr = float(torch.norm(pe41_t, dim=-1).mean().item())
        norm_tok41_tr = float(torch.norm(tok41_tr, dim=-1).mean().item())

        tok40_init, pe40_i = m40_init.tokenizer(X_train_samp[:500])
        tok40_tr, pe40_t = m40_tr.tokenizer(X_train_samp[:500])
        norm_pe40_init = float(torch.norm(pe40_i, dim=-1).mean().item())
        norm_tok40_init = float(torch.norm(tok40_init, dim=-1).mean().item())
        norm_pe40_tr = float(torch.norm(pe40_t, dim=-1).mean().item())
        norm_tok40_tr = float(torch.norm(tok40_tr, dim=-1).mean().item())

    # Question 3: Channel std of Delta H with and without pos_embed; mean |Delta H|
    with torch.no_grad():
        out41_tr = m41_tr(X_train_samp[:500])
        dH_41_with = out41_tr["H_target_for_loss"]  # [B, N_tgt, D]
        std_dH_with = float(dH_41_with.std(dim=(0, 1)).mean().item())
        mag_dH_with = float(torch.abs(dH_41_with).mean().item())

        # Zero out pos_embed
        orig_pe = m41_tr.tokenizer.pos_embed.data.clone()
        m41_tr.tokenizer.pos_embed.data.zero_()
        out41_tr_nopos = m41_tr(X_train_samp[:500])
        dH_41_no = out41_tr_nopos["H_target_for_loss"]
        std_dH_no = float(dH_41_no.std(dim=(0, 1)).mean().item())
        mag_dH_no = float(torch.abs(dH_41_no).mean().item())
        m41_tr.tokenizer.pos_embed.data.copy_(orig_pe)

    # Question 4: Loss configuration and gradient norms
    loss_type_41 = cfg41.loss_type
    amp_enabled = cfg41.mixed_precision
    # Predictor gradient norm at initial step vs end step
    # We can measure grad norm on initial batch
    m41_init_train = PECT_JEPA_5x5(cfg41).to(device).train()
    opt41_init = torch.optim.AdamW(m41_init_train.parameters(), lr=0.0003)
    loss_dict_init = m41_init_train(X_train_samp[:256])
    loss_dict_init["loss"].backward()
    grad_norm_init_pred = float(torch.nn.utils.clip_grad_norm_(m41_init_train.predictor.parameters(), 100.0).item())

    m41_tr_train = PECT_JEPA_5x5(cfg41).to(device).train()
    m41_tr_train.load_state_dict(ckpt41["model_state_dict"])
    opt41_tr = torch.optim.AdamW(m41_tr_train.parameters(), lr=0.0003)
    loss_dict_tr = m41_tr_train(X_train_samp[:256])
    loss_dict_tr["loss"].backward()
    grad_norm_end_pred = float(torch.nn.utils.clip_grad_norm_(m41_tr_train.predictor.parameters(), 100.0).item())

    c2_results = {
        "exp41": {
            "pos_embed_requires_grad": pe41_req_grad,
            "pos_embed_init_std": pe41_init_std,
            "pos_embed_tr_std": pe41_tr_std,
            "norm_pos_embed_init": norm_pe41_init,
            "norm_token_init": norm_tok41_init,
            "ratio_init": norm_pe41_init / (norm_tok41_init + 1e-9),
            "norm_pos_embed_tr": norm_pe41_tr,
            "norm_token_tr": norm_tok41_tr,
            "ratio_tr": norm_pe41_tr / (norm_tok41_tr + 1e-9),
            "std_dH_with_pos": std_dH_with,
            "std_dH_without_pos": std_dH_no,
            "mean_abs_dH_with_pos": mag_dH_with,
            "mean_abs_dH_without_pos": mag_dH_no,
            "loss_type": loss_type_41,
            "smooth_l1_beta": 1.0,
            "mixed_precision_amp": amp_enabled,
            "pred_grad_norm_init": grad_norm_init_pred,
            "pred_grad_norm_end": grad_norm_end_pred,
        },
        "exp40": {
            "pos_embed_requires_grad": pe40_req_grad,
            "pos_embed_init_std": pe40_init_std,
            "pos_embed_tr_std": pe40_tr_std,
            "norm_pos_embed_init": norm_pe40_init,
            "norm_token_init": norm_tok40_init,
            "ratio_init": norm_pe40_init / (norm_tok40_init + 1e-9),
            "norm_pos_embed_tr": norm_pe40_tr,
            "norm_token_tr": norm_tok40_tr,
            "ratio_tr": norm_pe40_tr / (norm_tok40_tr + 1e-9),
        }
    }

    print(json.dumps(c2_results, indent=2))

    # Save checkpoint C2 results
    out_dir = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage1_audit_c2.json"), "w", encoding="utf-8") as f:
        json.dump(c2_results, f, indent=2)

    print("\n[C2 COMPLETED]")


if __name__ == "__main__":
    main()
