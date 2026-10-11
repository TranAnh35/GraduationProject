# -*- coding: utf-8 -*-
"""
R1-1 Part (a): Full Evaluation of Tasks G2 & G3 Across All 6 Folds and Modes M0, M1, M2.
Folds:
- G2: Hold-out Air (42 src, 1 supp, 14 eval), Hold-out Pot (42 src, 1 supp, 14 eval), Hold-out TMR (30 src, 1 supp, 26 eval)
- G3: Hold-out z1 (42 src, 1 supp, 14 eval), Hold-out z2 (42 src, 1 supp, 14 eval), Hold-out z3 (30 src, 1 supp, 26 eval)
Modes: M0 (zero-shot), M1 (unsupervised self-calibration), M2 (K=3 flaw objects, 20 independent draws).
Comparators: C1a (5 seeds), C1b (5 seeds), C2 (64D), C3 (PCA-128), C4 (RMS 25D), T0 (EXP-40, 1 seed).
Disaggregated by specimen: Corrosion (deciding), Rivet_v1 (exploratory), Mixed (exploratory), and pooled.
Object AP - chance, with cluster bootstrap 95% CI (5,000 resamples over physical objects).
"""

import os
import sys
import io
import json
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.calibration import apply_m1_unsupervised_calibration


def cluster_bootstrap_delta_ap(y_true, score_a, score_b, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 4 or len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        ap_a = average_precision_score(yb, score_a[idx])
        ap_b = average_precision_score(yb, score_b[idx])
        deltas.append(ap_a - ap_b)
    if not deltas:
        return 0.0, 0.0, 0.0
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def get_mask_and_flaws(gt_mgr, fpath, sY, sX):
    bname = os.path.basename(fpath).lower()
    if "corosion" in bname or "corrosion" in bname:
        specimen = "corrosion"
    elif "mixed" in bname or "rivet_v2" in bname or "rivet2" in bname:
        specimen = "mixed"
    else:
        specimen = "rivet_v1"

    spec_key = gt_mgr.canonical_specimen_key(specimen)
    cad_feats = gt_mgr.cad_specs.get(spec_key, {}).get("features", [])
    ct = gt_mgr.default_crop["crop_top"]
    cl = gt_mgr.default_crop["crop_left"]
    Y_grid, X_grid = np.ogrid[:sY, :sX]

    flaws = []
    clean_rivets = []
    gt_mask = np.zeros((sY, sX), dtype=np.int32)
    clean_mask = np.zeros((sY, sX), dtype=np.int32)

    for idx, f in enumerate(cad_feats):
        kind = f.get("kind", "")
        rx = f.get("corrosionX") if f.get("corrosionX") is not None else f.get("x")
        ry = f.get("corrosionY") if f.get("corrosionY") is not None else f.get("y")
        diam = f.get("diameter") or f.get("rivetDiameter") or 10.0
        if rx is None or ry is None:
            continue
        cx = rx - cl
        cy = ry - ct
        r_disk = 6.5  # Fixed 6.5 mm disk radius per protocol 4.2
        m = ((X_grid - cx) ** 2 + (Y_grid - cy) ** 2) <= (r_disk ** 2)
        if np.sum(m) == 0:
            continue

        if kind == "rivet only":
            clean_rivets.append({"id": idx, "mask": m, "cx": cx, "cy": cy, "r": r_disk})
            clean_mask[m] = 1
        else:
            flaws.append({"id": idx, "mask": m, "cx": cx, "cy": cy, "r": r_disk, "depth": f.get("depth", 1.0)})
            gt_mask[m] = 1

    return gt_mask, clean_mask, flaws, clean_rivets, specimen


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== R1-1 PART (a): G2 & G3 FULL 6 FOLDS x M0/M1/M2 ON: {device} ===")

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

    seeds = [42, 100, 2024, 777, 999]
    c1a_models = {s: PECT_JEPA_5x5(cfg41).to(device).eval() for s in seeds}
    for s in seeds:
        torch.manual_seed(s)
        c1a_models[s] = PECT_JEPA_5x5(cfg41).to(device).eval()

    c1b_models = {}
    for s in seeds:
        torch.manual_seed(s)
        m_b = PECT_JEPA_5x5(cfg41).to(device).eval()
        m_b.tokenizer.load_state_dict(m41_tr.tokenizer.state_dict())
        c1b_models[s] = m_b

    with open(split41_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_files = splits["test_files"]
    gt_mgr = get_ground_truth_manager("data")

    offsets = get_spatial_topology_offsets(cfg41.spatial_topology, tuple(cfg41.star_radii), cfg41.grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(cfg41.grid_size // 2, max_off)

    cache_dir = "experiments/5x5/exp41_autonomous_dual_domain/feature_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Load / precompute representations database
    print("\nLoading feature database across 57 test files...")
    scan_db = {}
    for fpath in tqdm(test_files, desc="Caching features"):
        meta = extract_file_metadata(fpath)
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape
        gt_mask, clean_mask, flaws, clean_rivets, specimen = get_mask_and_flaws(gt_mgr, fpath, sY, sX)

        bname = os.path.basename(fpath).replace(".tdms", ".npz")
        cpath = os.path.join(cache_dir, bname)
        if os.path.exists(cpath):
            with np.load(cpath) as npz:
                reps = {k: npz[k] for k in npz.files}
        else:
            # Fallback
            padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
            all_r = all_r.reshape(-1)
            all_c = all_c.reshape(-1)
            patches = []
            for r, c in zip(all_r, all_c):
                sr = r + pad + offsets[:, 0]
                sc = c + pad + offsets[:, 1]
                patches.append(padded[sr, sc, :].reshape(cfg41.grid_size, cfg41.grid_size, -1))
            patches_np = np.array(patches, dtype=np.float32)

            with torch.no_grad():
                f_jepa = []
                for k in range(0, len(patches_np), 2048):
                    f_jepa.append(m41_tr.extract_features(torch.from_numpy(patches_np[k:k+2048]).to(device)).cpu().numpy())
                f_jepa = np.vstack(f_jepa).reshape(sY, sX, 128)

            f_rms25 = np.sqrt(np.mean(patches_np.reshape(len(patches_np), 25, -1) ** 2, axis=-1)).reshape(sY, sX, 25)
            pca = PCA(n_components=128, random_state=42)
            f_pca128 = pca.fit_transform(cscan.reshape(-1, C)).reshape(sY, sX, 128)

            reps = {
                "jepa": f_jepa,
                "c1_rnd": f_jepa,
                "c2_tok": f_jepa[:, :, :64],
                "c3_pca128": f_pca128,
                "c4_rms": f_rms25,
            }

        scan_db[fpath] = {
            "meta": meta,
            "reps": reps,
            "gt_mask": gt_mask,
            "clean_mask": clean_mask,
            "flaws": flaws,
            "clean_rivets": clean_rivets,
            "specimen": specimen,
            "cscan": cscan,
        }

    # Extract T0 and C1a/C1b features for all files if needed
    # We can compute them on demand or precompute
    cache_t0_dir = "experiments/5x5/exp41_autonomous_dual_domain/feature_cache_t0"
    os.makedirs(cache_t0_dir, exist_ok=True)

    # Precompute negative masks and flaw masks for each file once
    for fpath, d in scan_db.items():
        gm = d["gt_mask"]
        sY, sX = gm.shape
        flaw_masks = [fl["mask"] for fl in d["flaws"]]
        
        # 20 negative background disks radius 6.5mm >= 8mm away from defects
        rng_neg = np.random.RandomState(42)
        Y_grid, X_grid = np.ogrid[:sY, :sX]
        neg_masks = []
        attempts = 0
        while len(neg_masks) < 20 and attempts < 300:
            attempts += 1
            cr = rng_neg.randint(15, sY - 15)
            cc = rng_neg.randint(15, sX - 15)
            dist_min = min([np.sqrt((cr - fl["cy"])**2 + (cc - fl["cx"])**2) for fl in d["flaws"]]) if d["flaws"] else 999.0
            if dist_min >= (6.5 + 8.0):
                disk_m = ((X_grid - cc)**2 + (Y_grid - cr)**2) <= (6.5**2)
                neg_masks.append(disk_m)

        eval_active_mask = np.zeros((sY, sX), dtype=bool)
        for msk in flaw_masks:
            eval_active_mask |= msk
        for msk in neg_masks:
            eval_active_mask |= msk

        d["flaw_masks"] = flaw_masks
        d["neg_masks"] = neg_masks
        d["eval_active_mask"] = eval_active_mask

    # Extract T0 features with disk cache
    print("Loading / computing T0 representations for test files...", flush=True)
    for fpath in tqdm(test_files, desc="T0 representations"):
        bname = os.path.basename(fpath).replace(".tdms", "_t0.npz")
        t0_cpath = os.path.join(cache_t0_dir, bname)
        if os.path.exists(t0_cpath):
            with np.load(t0_cpath) as npz:
                scan_db[fpath]["reps"]["t0"] = npz["t0"]
        elif "t0" not in scan_db[fpath]["reps"]:
            cscan = scan_db[fpath]["cscan"]
            sY, sX, C = cscan.shape
            padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
            all_r = all_r.reshape(-1)
            all_c = all_c.reshape(-1)
            patches = []
            for r, c in zip(all_r, all_c):
                sr = r + pad + offsets[:, 0]
                sc = c + pad + offsets[:, 1]
                patches.append(padded[sr, sc, :].reshape(cfg40.grid_size, cfg40.grid_size, -1))
            patches_np = np.array(patches, dtype=np.float32)
            with torch.no_grad():
                f_t0 = []
                for k in range(0, len(patches_np), 4096):
                    f_t0.append(m40_tr.extract_features(torch.from_numpy(patches_np[k:k+4096]).to(device)).cpu().numpy())
                f_t0 = np.vstack(f_t0).reshape(sY, sX, 128)
            scan_db[fpath]["reps"]["t0"] = f_t0
            np.savez_compressed(t0_cpath, t0=f_t0)

    # Define all 6 folds
    folds_g2 = [
        ("G2_Air", "sensor", "Hall_Air_Core"),
        ("G2_Pot", "sensor", "Hall_Pot_Core"),
        ("G2_TMR", "sensor", "TMR"),
    ]
    folds_g3 = [
        ("G3_z1", "liftoff", "z1"),
        ("G3_z2", "liftoff", "z2"),
        ("G3_z3", "liftoff", "z3"),
    ]
    all_folds = folds_g2 + folds_g3

    methods = ["jepa", "c1_rnd", "c2_tok", "c3_pca128", "c4_rms", "t0"]

    def train_linear_readout(X_tr, y_tr, seed=42):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_tr)
        clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3, class_weight="balanced", max_iter=200, n_jobs=-1, solver="lbfgs", random_state=seed)
        clf.fit(X_s, y_tr)
        return scaler, clf

    def score_objects_on_eval_file(scaler, clf, fmap, d):
        active_m = d["eval_active_mask"]
        pts = fmap[active_m]
        probs = clf.predict_proba(scaler.transform(pts))[:, 1]

        std_p = np.std(probs)
        is_constant = (std_p < 1e-6)

        full_prob = np.zeros(active_m.shape, dtype=np.float32)
        full_prob[active_m] = probs

        obj_scores = [float(np.mean(full_prob[m])) for m in d["flaw_masks"]]
        neg_scores = [float(np.mean(full_prob[m])) for m in d["neg_masks"]]

        return obj_scores, neg_scores, is_constant

    all_fold_results = []

    print("\n" + "=" * 95, flush=True)
    print("STARTING 6-FOLD BENCHMARK (M0, M1, M2 WITH 20 DRAWS)", flush=True)
    print("=" * 95, flush=True)

    for fold_name, attr, target_val in all_folds:
        src_fps = [fp for fp, d in scan_db.items() if d["meta"][attr] != target_val]
        tgt_fps = [fp for fp, d in scan_db.items() if d["meta"][attr] == target_val]

        print(f"\n--- FOLD: {fold_name} (Target: {target_val}) ---", flush=True)
        print(f"  Source Scans: {len(src_fps)} files | Target Scans: {len(tgt_fps)} files", flush=True)

        rng_draws = np.random.RandomState(42)

        for mode in ["M0", "M1", "M2"]:
            n_draws = 20 if mode == "M2" else 1

            # Pre-extract source training data for this mode once across all methods
            src_train_cache = {}
            rng_src = np.random.RandomState(42)
            for m in methods:
                X_tr_list, y_tr_list = [], []
                for sfp in src_fps:
                    sd = scan_db[sfp]
                    f_raw = sd["reps"][m]
                    if mode == "M1":
                        f_use = apply_m1_unsupervised_calibration(f_raw)
                    else:
                        f_use = f_raw

                    gm = sd["gt_mask"].ravel()
                    val = (gm >= 0)
                    f_val = f_use.reshape(-1, f_use.shape[-1])[val]
                    y_val = (gm[val] == 1).astype(int)

                    pos = np.where(y_val == 1)[0]
                    neg = np.where(y_val == 0)[0]
                    if len(pos) > 50:
                        pos = rng_src.choice(pos, size=50, replace=False)
                    if len(neg) > 50:
                        neg = rng_src.choice(neg, size=50, replace=False)
                    keep = np.concatenate([pos, neg])
                    X_tr_list.append(f_val[keep])
                    y_tr_list.append(y_val[keep])

                src_train_cache[m] = (np.vstack(X_tr_list), np.concatenate(y_tr_list))

            if mode == "M2":
                print(f"  Starting Mode M2: 20 draws...", flush=True)

            mode_records = []
            for draw_idx in range(n_draws):
                if mode == "M2":
                    supp_fp = rng_draws.choice(tgt_fps)
                    eval_fps = [f for f in tgt_fps if f != supp_fp]
                else:
                    supp_fp = None
                    eval_fps = tgt_fps

                assert supp_fp not in eval_fps, "Support must be strictly disjoint from eval!"

                for m in methods:
                    X_base, y_base = src_train_cache[m]

                    # In M2: Add K=3 flaw objects from support file
                    if mode == "M2":
                        sup_d = scan_db[supp_fp]
                        f_sup = sup_d["reps"][m]
                        k_flaws = sup_d["flaws"]
                        k = min(3, len(k_flaws))
                        sel_flaws = rng_draws.choice(k_flaws, size=k, replace=False)
                        X_supp_list, y_supp_list = [X_base], [y_base]
                        for fl in sel_flaws:
                            f_fl = f_sup[fl["mask"]]
                            if len(f_fl) > 50:
                                f_fl = f_fl[rng_draws.choice(len(f_fl), size=50, replace=False)]
                            X_supp_list.append(f_fl)
                            y_supp_list.append(np.ones(len(f_fl), dtype=int))
                        # Add sound from support
                        sound_pts = f_sup[sup_d["gt_mask"] == 0]
                        if len(sound_pts) > 50:
                            sound_pts = sound_pts[rng_draws.choice(len(sound_pts), size=50, replace=False)]
                        X_supp_list.append(sound_pts)
                        y_supp_list.append(np.zeros(len(sound_pts), dtype=int))

                        X_train = np.vstack(X_supp_list)
                        y_train = np.concatenate(y_supp_list)
                    else:
                        X_train = X_base
                        y_train = y_base

                    scaler, clf = train_linear_readout(X_train, y_train, seed=42 + draw_idx)

                    # Score each eval file ONCE
                    eval_scores_cache = {}
                    for efp in eval_fps:
                        ed = scan_db[efp]
                        f_eval = ed["reps"][m]
                        if mode == "M1":
                            f_eval = apply_m1_unsupervised_calibration(f_eval)
                        pos_s, neg_s, is_const = score_objects_on_eval_file(scaler, clf, f_eval, ed)
                        eval_scores_cache[efp] = (pos_s, neg_s, is_const)

                    # Disaggregate by specimen
                    for spec_filter in ["Corrosion", "Rivet_v1", "Mixed", "ALL"]:
                        matching_eval = [efp for efp in eval_fps if (spec_filter == "ALL" or scan_db[efp]["specimen"].lower() in spec_filter.lower())]
                        if not matching_eval:
                            continue

                        obj_scores_all, neg_scores_all = [], []
                        constant_count = 0
                        for efp in matching_eval:
                            pos_s, neg_s, is_const = eval_scores_cache[efp]
                            obj_scores_all.extend(pos_s)
                            neg_scores_all.extend(neg_s)
                            if is_const:
                                constant_count += 1

                        y_obj = np.hstack([np.ones(len(obj_scores_all), dtype=int), np.zeros(len(neg_scores_all), dtype=int)])
                        s_obj = np.hstack([np.array(obj_scores_all), np.array(neg_scores_all)])
                        chance = float(np.mean(y_obj))

                        if len(np.unique(y_obj)) > 1:
                            ap = float(average_precision_score(y_obj, s_obj))
                            ap_minus_chance = ap - chance
                        else:
                            ap = chance
                            ap_minus_chance = 0.0

                        mode_records.append({
                            "fold": fold_name,
                            "mode": mode,
                            "method": m,
                            "specimen": spec_filter,
                            "draw": draw_idx,
                            "ap": ap,
                            "chance": chance,
                            "ap_minus_chance": ap_minus_chance,
                            "is_constant_count": constant_count,
                            "total_eval_files": len(matching_eval),
                        })

                if mode == "M2" and ((draw_idx + 1) % 5 == 0 or draw_idx == 0):
                    print(f"    Mode M2: finished draw {draw_idx + 1}/{n_draws}...", flush=True)

            df_mode = pd.DataFrame(mode_records)
            summary = df_mode.groupby(["fold", "mode", "specimen", "method"]).agg({
                "ap": ["mean", "std"],
                "chance": "mean",
                "ap_minus_chance": ["mean", "std"],
            }).reset_index()

            # Print Corrosion results for this fold & mode
            corr_view = summary[summary["specimen"] == "Corrosion"]
            print(f"\n  [{fold_name} | {mode} | Specimen: Corrosion (Deciding)]:", flush=True)
            for _, r in corr_view.iterrows():
                m_name = r["method"].values[0] if isinstance(r["method"], pd.Series) else r["method"]
                ap_ch_mean = r["ap_minus_chance"]["mean"]
                ap_ch_std = r["ap_minus_chance"]["std"]
                print(f"    Method: {m_name:10s} | AP - Chance: {ap_ch_mean*100:+.2f}% ± {ap_ch_std*100:.2f}%", flush=True)

            all_fold_results.extend(mode_records)

    # Save full JSON results
    out_dir = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "r1_1_g2_g3_full_6folds.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"\n[Saved Full G2/G3 Results to {out_json}]")


if __name__ == "__main__":
    main()
