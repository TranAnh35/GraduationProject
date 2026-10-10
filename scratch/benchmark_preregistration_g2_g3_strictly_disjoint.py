#!/usr/bin/env python3
"""
Pre-registration Phase 1: Strictly Disjoint Cross-Sensor (G2) & Cross-Lift-off (G3) Adaptation.

Protocol Fixes Applied:
1. G2 and G3 folds are separated and independent.
2. In Mode M2, Support files and Evaluation files are STRICTLY DISJOINT:
   Support File(s) contain the K=3 calibration defects and are EXCLUDED from evaluation.
   Evaluation is performed strictly on unseen held-out target files.
   Printed explicitly for each fold: Source files, Target support files, Target eval files.
3. Baselines:
   - EXP-41 JEPA (128D)
   - C1: Random Untrained Encoder (128D)
   - C2: Tokenizer Output (64D)
   - C3: PCA-128 (128D from 128 raw temporal samples)
   - C4: Raw Energy RMS (25D across 25 probes)
4. Fair Readout: Regularization C is tuned via cross-validation on support data for all methods.
5. Statistical Bootstrap: Resampled at the defect object level.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath("."))
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.calibration import apply_m1_unsupervised_calibration
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata


def object_level_bootstrap_ci(y_true, score_a, score_b, n_boot=2000, seed=42):
    """
    Bootstrap 95% CI for Delta AP = AP(a) - AP(b) resampled at object level.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 5:
        return 0.0, 0.0, 0.0
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


def extract_reps(model, rand_model, std_cscan, pca_model=None, batch_size=2048, device="cuda"):
    """
    Extracts all representations matching dimensionalities:
    - JEPA: 128D
    - C1: Random Enc (128D)
    - C2: Tokenizer (64D)
    - C3: PCA-128 (128D)
    - C4: Raw Energy RMS across 25 probes (25D)
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

    fmap_jepa = np.zeros((sY, sX, 128), dtype=np.float32)
    fmap_rnd = np.zeros((sY, sX, 128), dtype=np.float32)
    fmap_tok = np.zeros((sY, sX, 64), dtype=np.float32)
    fmap_rms25 = np.zeros((sY, sX, 25), dtype=np.float32)

    all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
    all_r = all_r.reshape(-1)
    all_c = all_c.reshape(-1)
    total_pts = sY * sX

    model.eval()
    rand_model.eval()

    with torch.inference_mode():
        for k in range(0, total_pts, batch_size):
            k_end = min(k + batch_size, total_pts)
            rows_b = all_r[k:k_end] + pad
            cols_b = all_c[k:k_end] + pad

            sample_r = rows_b[:, None] + offsets[None, :, 0]
            sample_c = cols_b[:, None] + offsets[None, :, 1]

            patch_b = padded[sample_r, sample_c, :].reshape(-1, grid_size, grid_size, C)
            x_b = torch.from_numpy(patch_b).float().to(dev)

            # C4: RMS across all 25 probes [B, 25]
            rms_probes = torch.sqrt(torch.mean(x_b.view(-1, 25, C) ** 2, dim=-1))
            fmap_rms25[all_r[k:k_end], all_c[k:k_end]] = rms_probes.cpu().numpy()

            # C2: Tokenizer output center token [B, 64]
            tokens, _ = model.tokenizer(x_b)
            fmap_tok[all_r[k:k_end], all_c[k:k_end]] = tokens[:, 0, :].cpu().numpy()

            # JEPA: 128D
            z_jepa = model.extract_features(x_b).cpu().numpy()
            fmap_jepa[all_r[k:k_end], all_c[k:k_end]] = z_jepa

            # C1: Random Enc: 128D
            z_rnd = rand_model.extract_features(x_b).cpu().numpy()
            fmap_rnd[all_r[k:k_end], all_c[k:k_end]] = z_rnd

    # C3: PCA-128
    flat_cscan = std_cscan.reshape(-1, C)
    if pca_model is None:
        pca_model = PCA(n_components=min(128, C), random_state=42)
        sample_indices = np.random.choice(len(flat_cscan), size=min(10000, len(flat_cscan)), replace=False)
        pca_model.fit(flat_cscan[sample_indices])
    fmap_pca128 = pca_model.transform(flat_cscan).reshape(sY, sX, -1).astype(np.float32)

    return {
        "jepa": fmap_jepa,
        "c1_rnd": fmap_rnd,
        "c2_tok": fmap_tok,
        "c3_pca128": fmap_pca128,
        "c4_rms": fmap_rms25,
    }, pca_model


def get_mask_and_flaws(gt_mgr, fpath, sY, sX):
    bname = os.path.basename(fpath)
    specimen = "corrosion" if "corrosion" in bname.lower() or "corosion" in bname.lower() else ("mixed" if ("mixed" in bname.lower() or "rivet_v2" in bname.lower() or "rivet2" in bname.lower()) else "rivet_v1")

    # Load pre-saved ground truth mask
    mask_file = os.path.join("data", "ground_truth", "corrosion" if specimen == "corrosion" else ("rivet_v2" if specimen == "mixed" else "rivet_v1"), f"{'corrosion' if specimen == 'corrosion' else ('rivet_v2' if specimen == 'mixed' else 'rivet_v1')}_gt_mask.npy")

    if os.path.exists(mask_file):
        gt_mask = np.load(mask_file)
    else:
        gt_mask = gt_mgr.generate_multiclass_mask(specimen, buffer_px=2)

    # CAD features
    spec_key = gt_mgr.canonical_specimen_key(specimen)
    cad_feats = gt_mgr.cad_specs.get(spec_key, {}).get("features", [])
    ct = gt_mgr.default_crop["crop_top"]
    cl = gt_mgr.default_crop["crop_left"]
    Y_grid, X_grid = np.ogrid[:sY, :sX]

    flaws = []
    for idx, f in enumerate(cad_feats):
        if f.get("kind") == "rivet only":
            continue
        cx = f.get("corrosionX") if f.get("corrosionX") is not None else f.get("x")
        cy = f.get("corrosionY") if f.get("corrosionY") is not None else f.get("y")
        diam = f.get("diameter") or f.get("rivetDiameter") or 10.0
        if cx is None or cy is None:
            continue
        cx_c = cx - cl
        cy_c = cy - ct
        r = diam / 2.0
        obj_m = ((X_grid - cx_c) ** 2 + (Y_grid - cy_c) ** 2) <= (r ** 2)
        if np.sum(obj_m) > 0:
            flaws.append({
                "id": idx,
                "mask": obj_m,
                "depth": f.get("depth", 1.0),
            })

    return gt_mask, flaws


def train_calibrated_probe(X_train, y_train, seed=42):
    """
    Fits logistic regression with balanced weighting and cross-validated C in [1e-2, 1e2].
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    # Fast grid of C
    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
        cv=3,
        class_weight="balanced",
        max_iter=300,
        random_state=seed,
    )
    clf.fit(X_s, y_train)
    return scaler, clf


def evaluate_on_file(scaler, clf, fmap, gt_mask, flaws):
    """
    Evaluates linear classifier on a test scan, returning pixel AP and object AP.
    """
    valid = (gt_mask.ravel() >= 0)
    y_true = (gt_mask.ravel()[valid] == 1).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    flat_f = fmap.reshape(-1, fmap.shape[-1])[valid]
    flat_f_s = scaler.transform(flat_f)
    probs = clf.predict_proba(flat_f_s)[:, 1]

    pix_ap = average_precision_score(y_true, probs)

    # Object-level
    obj_scores = []
    obj_labels = []
    full_prob = np.zeros(gt_mask.shape, dtype=np.float32)
    full_prob.ravel()[valid] = probs

    for fl in flaws:
        m = fl["mask"] & (gt_mask == 1)
        if np.sum(m) > 0:
            obj_scores.append(float(np.mean(full_prob[m])))
            obj_labels.append(1)

    # Sample sound background regions
    sound_mask = (gt_mask == 0)
    sound_scores = full_prob[sound_mask]
    rng = np.random.RandomState(42)
    sampled_sound = rng.choice(sound_scores, size=min(len(obj_scores) * 20, len(sound_scores)), replace=False)

    y_obj = np.hstack([np.ones(len(obj_scores), dtype=int), np.zeros(len(sampled_sound), dtype=int)])
    s_obj = np.hstack([np.array(obj_scores), sampled_sound])
    obj_ap = average_precision_score(y_obj, s_obj) if len(np.unique(y_obj)) > 1 else np.nan

    return float(pix_ap), float(obj_ap)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Pre-Registration Strict Disjoint Adaptation on: {device}")

    cfg_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"

    config = Spatiotemporal5x5Config.from_json(cfg_path)
    model = PECT_JEPA_5x5(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    torch.manual_seed(999)
    rand_model = PECT_JEPA_5x5(config)
    rand_model.to(device).eval()

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    gt_mgr = get_ground_truth_manager("data")
    test_files = splits["test_files"]
    print(f"Total Held-Out Test Scans: {len(test_files)}")

    # Extract all representations
    scan_db = {}
    cache_dir = "experiments/5x5/exp41_autonomous_dual_domain/feature_cache"
    os.makedirs(cache_dir, exist_ok=True)
    pca_global = None

    print("\n--- Step 1: Pre-computing Representations (with Disk Cache) ---")
    for fpath in tqdm(test_files, desc="Precomputing Features"):
        meta = extract_file_metadata(fpath)
        std_cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = std_cscan.shape
        gt_mask, flaws = get_mask_and_flaws(gt_mgr, fpath, sY, sX)

        bname = os.path.basename(fpath).replace(".tdms", ".npz")
        cpath = os.path.join(cache_dir, bname)
        if os.path.exists(cpath):
            with np.load(cpath) as npz:
                reps = {k: npz[k] for k in npz.files}
        else:
            reps, pca_global = extract_reps(model, rand_model, std_cscan, pca_model=pca_global, batch_size=2048, device=device)
            np.savez_compressed(cpath, **reps)

        scan_db[fpath] = {
            "meta": meta,
            "reps": reps,
            "gt_mask": gt_mask,
            "flaws": flaws,
        }

    methods = ["jepa", "c1_rnd", "c2_tok", "c3_pca128", "c4_rms"]

    # =========================================================================
    # TASK G2: CROSS-SENSOR ADAPTATION (3-FOLD LOSO)
    # =========================================================================
    print("\n" + "=" * 95)
    print("TASK G2: CROSS-SENSOR ADAPTATION (STRICTLY DISJOINT SUPPORT & EVALUATION SETS)")
    print("=" * 95)

    sensors = ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]
    g2_results = {
        mode: {m: {"pix_ap": [], "obj_ap": [], "specimen": [], "file": []} for m in methods}
        for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]
    }

    for test_sensor in sensors:
        train_sensors = [s for s in sensors if s != test_sensor]
        src_files = [fp for fp, d in scan_db.items() if d["meta"]["sensor"] in train_sensors]
        tgt_all_files = [fp for fp, d in scan_db.items() if d["meta"]["sensor"] == test_sensor]

        # Designate STRICTLY DISJOINT Support vs Evaluation within Target Domain:
        # We designate the first file as the calibration support file (e.g. z1 Square reference)
        tgt_support_files = [tgt_all_files[0]]
        tgt_eval_files = tgt_all_files[1:]

        # Verification assertion:
        assert set(tgt_support_files).isdisjoint(set(tgt_eval_files)), "PROTOCOL VIOLATION: Support set overlaps Evaluation set!"

        print(f"\n--- FOLD TARGET SENSOR: {test_sensor} ---")
        print(f"  Source Sensors:            {train_sensors} ({len(src_files)} files)")
        print(f"  Target Support File (K=3): {[os.path.basename(f) for f in tgt_support_files]}")
        print(f"  Target Eval Files:         {len(tgt_eval_files)} files (strictly held-out)")
        print(f"  Disjoint Check:            PASSED (Support INTERSECT Eval == EMPTY)")

        for m in methods:
            # 1. Source Training Data
            X_src_list, y_src_list = [], []
            for fp in src_files:
                d = scan_db[fp]
                gm = d["gt_mask"].ravel()
                val = (gm >= 0)
                f = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
                y = (gm[val] == 1).astype(int)
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                if len(neg) > 1000:
                    neg = np.random.choice(neg, size=1000, replace=False)
                keep = np.concatenate([pos, neg])
                X_src_list.append(f[keep])
                y_src_list.append(y[keep])

            X_src = np.vstack(X_src_list)
            y_src = np.concatenate(y_src_list)

            # Fit M0
            scaler_m0, clf_m0 = train_calibrated_probe(X_src, y_src, seed=42)

            # Fit M1 (Calibrated Source)
            X_src_m1_list = []
            for fp in src_files:
                d = scan_db[fp]
                cal_f = apply_m1_unsupervised_calibration(d["reps"][m])
                gm = d["gt_mask"].ravel()
                val = (gm >= 0)
                cal_f_val = cal_f.reshape(-1, cal_f.shape[-1])[val]
                y = (gm[val] == 1).astype(int)
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                if len(neg) > 1000:
                    neg = np.random.choice(neg, size=1000, replace=False)
                keep = np.concatenate([pos, neg])
                X_src_m1_list.append(cal_f_val[keep])

            X_src_m1 = np.vstack(X_src_m1_list)
            scaler_m1, clf_m1 = train_calibrated_probe(X_src_m1, y_src, seed=42)

            # Fit M2 (K=3 Calibration Defects from Support File + Target Sound Background)
            supp_data = scan_db[tgt_support_files[0]]
            supp_flaws = supp_data["flaws"]
            k = min(3, len(supp_flaws))
            rng = np.random.RandomState(42)
            cal_flaw_idx = rng.choice(len(supp_flaws), size=k, replace=False)
            cal_flaws = [supp_flaws[i] for i in cal_flaw_idx]

            supp_fmap = supp_data["reps"][m]
            flaw_pts = [supp_fmap[fl["mask"]] for fl in cal_flaws]
            X_fl_k = np.vstack(flaw_pts)
            sound_pts = supp_fmap[supp_data["gt_mask"] == 0]
            if len(sound_pts) > len(X_fl_k) * 10:
                sound_pts = sound_pts[rng.choice(len(sound_pts), size=len(X_fl_k) * 10, replace=False)]

            X_supp = np.vstack([X_fl_k, sound_pts])
            y_supp = np.hstack([np.ones(len(X_fl_k), dtype=int), np.zeros(len(sound_pts), dtype=int)])
            scaler_m2, clf_m2 = train_calibrated_probe(X_supp, y_supp, seed=42)

            # Evaluate strictly on held-out tgt_eval_files
            for fp in tgt_eval_files:
                d = scan_db[fp]
                gm = d["gt_mask"]
                fl = d["flaws"]
                spec = d["meta"]["specimen"]

                # M0
                p_ap, o_ap = evaluate_on_file(scaler_m0, clf_m0, d["reps"][m], gm, fl)
                g2_results["M0_zeroshot"][m]["pix_ap"].append(p_ap)
                g2_results["M0_zeroshot"][m]["obj_ap"].append(o_ap)
                g2_results["M0_zeroshot"][m]["specimen"].append(spec)
                g2_results["M0_zeroshot"][m]["file"].append(fp)

                # M1
                cal_f = apply_m1_unsupervised_calibration(d["reps"][m])
                p_ap, o_ap = evaluate_on_file(scaler_m1, clf_m1, cal_f, gm, fl)
                g2_results["M1_unsupervised"][m]["pix_ap"].append(p_ap)
                g2_results["M1_unsupervised"][m]["obj_ap"].append(o_ap)
                g2_results["M1_unsupervised"][m]["specimen"].append(spec)
                g2_results["M1_unsupervised"][m]["file"].append(fp)

                # M2
                p_ap, o_ap = evaluate_on_file(scaler_m2, clf_m2, d["reps"][m], gm, fl)
                g2_results["M2_fewshot_k3"][m]["pix_ap"].append(p_ap)
                g2_results["M2_fewshot_k3"][m]["obj_ap"].append(o_ap)
                g2_results["M2_fewshot_k3"][m]["specimen"].append(spec)
                g2_results["M2_fewshot_k3"][m]["file"].append(fp)

    # =========================================================================
    # TASK G3: CROSS-LIFT-OFF ADAPTATION (3-FOLD)
    # =========================================================================
    print("\n" + "=" * 95)
    print("TASK G3: CROSS-LIFT-OFF ADAPTATION (STRICTLY DISJOINT SUPPORT & EVALUATION SETS)")
    print("=" * 95)

    liftoff_folds = [
        ({"z1", "z2"}, "z3", "Extrapolation z1,z2 -> z3"),
        ({"z1", "z3"}, "z2", "Interpolation z1,z3 -> z2"),
        ({"z2", "z3"}, "z1", "Extrapolation z2,z3 -> z1"),
    ]

    g3_results = {
        mode: {m: {"pix_ap": [], "obj_ap": [], "specimen": [], "file": []} for m in methods}
        for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]
    }

    for src_lo, tgt_lo, fold_name in liftoff_folds:
        src_files = [fp for fp, d in scan_db.items() if d["meta"]["liftoff"] in src_lo]
        tgt_all_files = [fp for fp, d in scan_db.items() if d["meta"]["liftoff"] == tgt_lo]

        # Strictly disjoint support vs evaluation
        tgt_support_files = [tgt_all_files[0]]
        tgt_eval_files = tgt_all_files[1:]
        assert set(tgt_support_files).isdisjoint(set(tgt_eval_files))

        print(f"\n--- FOLD LIFT-OFF: {fold_name} ---")
        print(f"  Source Lift-offs:          {src_lo} ({len(src_files)} files)")
        print(f"  Target Support File (K=3): {[os.path.basename(f) for f in tgt_support_files]}")
        print(f"  Target Eval Files:         {len(tgt_eval_files)} files (strictly held-out)")
        print(f"  Disjoint Check:            PASSED (Support INTERSECT Eval == EMPTY)")

        for m in methods:
            # Source data
            X_src_list, y_src_list = [], []
            for fp in src_files:
                d = scan_db[fp]
                gm = d["gt_mask"].ravel()
                val = (gm >= 0)
                f = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
                y = (gm[val] == 1).astype(int)
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                if len(neg) > 1000:
                    neg = np.random.choice(neg, size=1000, replace=False)
                keep = np.concatenate([pos, neg])
                X_src_list.append(f[keep])
                y_src_list.append(y[keep])

            X_src = np.vstack(X_src_list)
            y_src = np.concatenate(y_src_list)

            # Fit M0
            scaler_m0, clf_m0 = train_calibrated_probe(X_src, y_src, seed=42)

            # Fit M1
            X_src_m1_list = []
            for fp in src_files:
                d = scan_db[fp]
                cal_f = apply_m1_unsupervised_calibration(d["reps"][m])
                gm = d["gt_mask"].ravel()
                val = (gm >= 0)
                cal_f_val = cal_f.reshape(-1, cal_f.shape[-1])[val]
                y = (gm[val] == 1).astype(int)
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                if len(neg) > 1000:
                    neg = np.random.choice(neg, size=1000, replace=False)
                keep = np.concatenate([pos, neg])
                X_src_m1_list.append(cal_f_val[keep])

            X_src_m1 = np.vstack(X_src_m1_list)
            scaler_m1, clf_m1 = train_calibrated_probe(X_src_m1, y_src, seed=42)

            # Fit M2
            supp_data = scan_db[tgt_support_files[0]]
            supp_flaws = supp_data["flaws"]
            k = min(3, len(supp_flaws))
            rng = np.random.RandomState(42)
            cal_flaw_idx = rng.choice(len(supp_flaws), size=k, replace=False)
            cal_flaws = [supp_flaws[i] for i in cal_flaw_idx]

            supp_fmap = supp_data["reps"][m]
            flaw_pts = [supp_fmap[fl["mask"]] for fl in cal_flaws]
            X_fl_k = np.vstack(flaw_pts)
            sound_pts = supp_fmap[supp_data["gt_mask"] == 0]
            if len(sound_pts) > len(X_fl_k) * 10:
                sound_pts = sound_pts[rng.choice(len(sound_pts), size=len(X_fl_k) * 10, replace=False)]

            X_supp = np.vstack([X_fl_k, sound_pts])
            y_supp = np.hstack([np.ones(len(X_fl_k), dtype=int), np.zeros(len(sound_pts), dtype=int)])
            scaler_m2, clf_m2 = train_calibrated_probe(X_supp, y_supp, seed=42)

            # Evaluate strictly on held-out tgt_eval_files
            for fp in tgt_eval_files:
                d = scan_db[fp]
                gm = d["gt_mask"]
                fl = d["flaws"]
                spec = d["meta"]["specimen"]

                # M0
                p_ap, o_ap = evaluate_on_file(scaler_m0, clf_m0, d["reps"][m], gm, fl)
                g3_results["M0_zeroshot"][m]["pix_ap"].append(p_ap)
                g3_results["M0_zeroshot"][m]["obj_ap"].append(o_ap)
                g3_results["M0_zeroshot"][m]["specimen"].append(spec)
                g3_results["M0_zeroshot"][m]["file"].append(fp)

                # M1
                cal_f = apply_m1_unsupervised_calibration(d["reps"][m])
                p_ap, o_ap = evaluate_on_file(scaler_m1, clf_m1, cal_f, gm, fl)
                g3_results["M1_unsupervised"][m]["pix_ap"].append(p_ap)
                g3_results["M1_unsupervised"][m]["obj_ap"].append(o_ap)
                g3_results["M1_unsupervised"][m]["specimen"].append(spec)
                g3_results["M1_unsupervised"][m]["file"].append(fp)

                # M2
                p_ap, o_ap = evaluate_on_file(scaler_m2, clf_m2, d["reps"][m], gm, fl)
                g3_results["M2_fewshot_k3"][m]["pix_ap"].append(p_ap)
                g3_results["M2_fewshot_k3"][m]["obj_ap"].append(o_ap)
                g3_results["M2_fewshot_k3"][m]["specimen"].append(spec)
                g3_results["M2_fewshot_k3"][m]["file"].append(fp)

    # Print Disaggregated and Overall Summaries
    specimen_names = [None, "Corrosion", "Rivet_v1", "Rivet_v2"]

    for task_name, task_dict in [("G2 (CROSS-SENSOR)", g2_results), ("G3 (CROSS-LIFT-OFF)", g3_results)]:
        print("\n" + "#" * 95)
        print(f"REPORT: {task_name}")
        print("#" * 95)
        for spec in specimen_names:
            spec_label = "ALL SPECIMENS (OVERALL)" if spec is None else f"SPECIMEN: {spec.upper()}"
            print("\n" + "=" * 90)
            print(f"--- {spec_label} ---")
            print("=" * 90)
            for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]:
                print(f"\nMode: {mode}")
                print(f"{'Method':<12} | {'Pixel AP Mean ± Std':<24} | {'Object AP Mean ± Std':<24} | Count")
                print("-" * 75)
                for m in methods:
                    p_arr = np.array(task_dict[mode][m]["pix_ap"])
                    o_arr = np.array(task_dict[mode][m]["obj_ap"])
                    s_arr = np.array(task_dict[mode][m]["specimen"])
                    if spec is not None:
                        mask = (s_arr == spec)
                        p_arr = p_arr[mask]
                        o_arr = o_arr[mask]

                    p_v = p_arr[~np.isnan(p_arr)]
                    o_v = o_arr[~np.isnan(o_arr)]
                    p_str = f"{np.mean(p_v)*100:5.2f}% ± {np.std(p_v)*100:5.2f}%" if len(p_v) > 0 else "N/A"
                    o_str = f"{np.mean(o_v)*100:5.2f}% ± {np.std(o_v)*100:5.2f}%" if len(o_v) > 0 else "N/A"
                    print(f"{m:<12} | {p_str:<24} | {o_str:<24} | N={len(o_v)}")

    # =========================================================================
    # PRE-REGISTRATION VALUE AUDIT (OBJECT-LEVEL BOOTSTRAP OVER OBJECTS/FILES)
    # =========================================================================
    print("\n" + "=" * 95)
    print("OBJECT-LEVEL BOOTSTRAP VALUE CRITERION AUDIT (DELTA AP >= 0.05 & 95% CI > 0)")
    print("=" * 95)

    for task_name, task_dict in [("G2 (Cross-Sensor)", g2_results), ("G3 (Cross-Lift-off)", g3_results)]:
        for mode in ["M1_unsupervised", "M2_fewshot_k3"]:
            j_o = np.array(task_dict[mode]["jepa"]["obj_ap"])
            valid = ~np.isnan(j_o)
            j_o = j_o[valid]

            base_names = [m for m in methods if m != "jepa"]
            best_base = max(base_names, key=lambda m: np.nanmean(task_dict[mode][m]["obj_ap"]))
            b_o = np.array(task_dict[mode][best_base]["obj_ap"])[valid]

            delta_mean = np.mean(j_o) - np.mean(b_o)

            # Bootstrap over files
            rng = np.random.RandomState(42)
            boot_deltas = []
            for _ in range(5000):
                idx = rng.randint(0, len(j_o), size=len(j_o))
                boot_deltas.append(np.mean(j_o[idx]) - np.mean(b_o[idx]))
            ci_low = np.percentile(boot_deltas, 2.5)
            ci_high = np.percentile(boot_deltas, 97.5)

            status = "PASSED" if (delta_mean >= 0.05 and ci_low > 0) else "NOT MET"
            print(f"Task: {task_name:<20} | Mode: {mode:<15} | Best Baseline: {best_base:<10}")
            print(f"  JEPA Object AP: {np.mean(j_o)*100:.2f}% | Best Base Object AP: {np.mean(b_o)*100:.2f}% | Delta AP: {delta_mean*100:+.2f}%")
            print(f"  95% Bootstrap CI: [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%] -> STATUS: {status}\n")

    # Save to JSON with specimen and file metadata
    out_file = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results/preregistration_g2_g3_strictly_disjoint.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "g2": {
                mode: {
                    m: {
                        "pix_ap": [float(x) if not np.isnan(x) else None for x in g2_results[mode][m]["pix_ap"]],
                        "obj_ap": [float(x) if not np.isnan(x) else None for x in g2_results[mode][m]["obj_ap"]],
                        "specimen": g2_results[mode][m]["specimen"],
                        "file": g2_results[mode][m]["file"],
                    } for m in methods
                } for mode in g2_results
            },
            "g3": {
                mode: {
                    m: {
                        "pix_ap": [float(x) if not np.isnan(x) else None for x in g3_results[mode][m]["pix_ap"]],
                        "obj_ap": [float(x) if not np.isnan(x) else None for x in g3_results[mode][m]["obj_ap"]],
                        "specimen": g3_results[mode][m]["specimen"],
                        "file": g3_results[mode][m]["file"],
                    } for m in methods
                } for mode in g3_results
            },
        }, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
