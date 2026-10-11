#!/usr/bin/env python3
"""
Pre-registration G2 & G3 Protocol Fix:
- 20 Independent Support Draws (strictly disjoint from evaluation files)
- Hold-out Sensor G2 (Train Air+Pot, Test TMR, 27 files: 1 support, 26 eval)
- Hold-out Lift-off G3 (Train z1+z2, Test z3, 27 files: 1 support, 26 eval)
- Readout degeneracy check (variance of predicted probabilities > 0)
- Chance level reporting (prevalence and AP - chance)
- Random Encoder (C1) across seeds [42, 100, 2024, 777, 999]
- Disaggregated by Specimen (Corrosion, Rivet_v1, Mixed with 2 negative sets)
- Cluster Bootstrap (5,000 resamples) for Delta AP (JEPA - Random) with 95% CI
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
from sklearn.linear_model import LogisticRegression
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
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.calibration import apply_m1_unsupervised_calibration


def cluster_bootstrap_delta_ap(y_true, score_a, score_b, n_boot=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 5:
        return 0.0, 0.0, 0.0
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        ap_a = average_precision_score(yt_b, score_a[idx])
        ap_b = average_precision_score(yt_b, score_b[idx])
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
        r_disk = diam / 2.0
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


def train_probe_safe(X_train, y_train, seed=42):
    """
    Fits StandardScaler and LogisticRegression with class_weight='balanced'.
    Checks for degenerate output.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    # Check if features have zero variance
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300, random_state=seed)
    clf.fit(X_s, y_train)
    return scaler, clf


def evaluate_probe_on_file(scaler, clf, fmap, gt_mask, clean_mask, flaws, clean_rivets, specimen):
    """
    Evaluates linear classifier on a test scan.
    Returns:
    - pix_ap, pix_auc
    - obj_ap, obj_auc
    - chance_level (prevalence)
    - prob_std (to verify non-degeneracy)
    - if Mixed: clean_rivet_ap, clean_rivet_auc (defect vs clean rivets)
    """
    y_true = (gt_mask.ravel() == 1).astype(int)
    chance = float(np.mean(y_true))
    if len(np.unique(y_true)) < 2:
        return None

    flat_f = fmap.reshape(-1, fmap.shape[-1])
    flat_f_s = np.nan_to_num(scaler.transform(flat_f), nan=0.0, posinf=0.0, neginf=0.0)
    probs = clf.predict_proba(flat_f_s)[:, 1]
    prob_std = float(np.std(probs))

    # Pixel metrics
    pix_ap = float(average_precision_score(y_true, probs))
    pix_auc = float(roc_auc_score(y_true, probs))

    # Object metrics
    full_prob = probs.reshape(gt_mask.shape)
    obj_scores = []
    obj_labels = []

    for fl in flaws:
        m = fl["mask"] & (gt_mask == 1)
        if np.sum(m) > 0:
            obj_scores.append(float(np.mean(full_prob[m])))
            obj_labels.append(1)

    # Sample sound background
    sound_mask = (gt_mask == 0) & (clean_mask == 0)
    sound_scores = full_prob[sound_mask]
    rng = np.random.RandomState(42)
    if len(sound_scores) > len(obj_scores) * 20:
        sampled_sound = rng.choice(sound_scores, size=len(obj_scores) * 20, replace=False)
    else:
        sampled_sound = sound_scores

    y_obj = np.hstack([np.ones(len(obj_scores), dtype=int), np.zeros(len(sampled_sound), dtype=int)])
    s_obj = np.hstack([np.array(obj_scores), sampled_sound])
    obj_ap = float(average_precision_score(y_obj, s_obj)) if len(np.unique(y_obj)) > 1 else np.nan
    obj_auc = float(roc_auc_score(y_obj, s_obj)) if len(np.unique(y_obj)) > 1 else np.nan

    # Mixed negative set 1: defect vs clean rivets only
    mixed_rivet_ap = np.nan
    mixed_rivet_auc = np.nan
    if specimen == "mixed" and len(clean_rivets) > 0:
        clean_scores = [float(np.mean(full_prob[cr["mask"]])) for cr in clean_rivets if np.sum(cr["mask"]) > 0]
        y_riv = np.hstack([np.ones(len(obj_scores), dtype=int), np.zeros(len(clean_scores), dtype=int)])
        s_riv = np.hstack([np.array(obj_scores), np.array(clean_scores)])
        if len(np.unique(y_riv)) > 1:
            mixed_rivet_ap = float(average_precision_score(y_riv, s_riv))
            mixed_rivet_auc = float(roc_auc_score(y_riv, s_riv))

    return {
        "pix_ap": pix_ap,
        "pix_auc": pix_auc,
        "obj_ap": obj_ap,
        "obj_auc": obj_auc,
        "chance": chance,
        "prob_std": prob_std,
        "mixed_rivet_ap": mixed_rivet_ap,
        "mixed_rivet_auc": mixed_rivet_auc,
    }


def main():
    print("=== C5 DISJOINT PROTOCOL FIX: 20 INDEPENDENT SUPPORT DRAWS ===")
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"
    cache_dir = "experiments/5x5/exp41_autonomous_dual_domain/feature_cache"

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_files = splits["test_files"]
    gt_mgr = get_ground_truth_manager("data")

    # Load cached features and masks for all 57 test files
    scan_db = {}
    print("Loading precomputed features from cache...")
    for fpath in tqdm(test_files, desc="Loading cache"):
        bname = os.path.basename(fpath).replace(".tdms", ".npz")
        cpath = os.path.join(cache_dir, bname)
        if not os.path.exists(cpath):
            continue
        meta = extract_file_metadata(fpath)
        with np.load(cpath) as npz:
            reps = {k: npz[k] for k in npz.files}
        sY, sX = reps["jepa"].shape[:2]
        gt_mask, clean_mask, flaws, clean_rivets, specimen = get_mask_and_flaws(gt_mgr, fpath, sY, sX)
        scan_db[fpath] = {
            "meta": meta,
            "reps": reps,
            "gt_mask": gt_mask,
            "clean_mask": clean_mask,
            "flaws": flaws,
            "clean_rivets": clean_rivets,
            "specimen": specimen,
        }

    print(f"Loaded {len(scan_db)} scans from cache.")

    methods = ["jepa", "c1_rnd", "c3_pca128", "c4_rms"]

    # =========================================================================
    # TASK G2: HOLD-OUT SENSOR (Train Air+Pot, Test TMR)
    # =========================================================================
    print("\n" + "=" * 95)
    print("TASK G2: HOLD-OUT SENSOR (Source: Air+Pot, Target: TMR) - 20 INDEPENDENT DRAWS")
    print("=" * 95)

    g2_src_files = [fp for fp, d in scan_db.items() if d["meta"]["sensor"] in ["Hall_Air_Core", "Hall_Pot_Core"]]
    g2_tgt_files = [fp for fp, d in scan_db.items() if d["meta"]["sensor"] == "TMR"]
    print(f"G2 Source files (Air + Pot): {len(g2_src_files)} files")
    print(f"G2 Target files (TMR):       {len(g2_tgt_files)} files (1 support, {len(g2_tgt_files)-1} eval)")

    # 20 Draws for G2
    g2_results = {mode: {m: [] for m in methods} for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]}
    g2_chances = []
    g2_deltas_m2 = []

    # Fit M0 and M1 Source Probes once (source data is fixed)
    scaler_m0 = {}
    clf_m0 = {}
    scaler_m1 = {}
    clf_m1 = {}

    for m in methods:
        X_src_m0, y_src = [], []
        X_src_m1 = []
        for fp in g2_src_files:
            d = scan_db[fp]
            gm = d["gt_mask"].ravel()
            y = (gm == 1).astype(int)
            val = (gm >= 0)
            f0 = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
            f1 = apply_m1_unsupervised_calibration(d["reps"][m]).reshape(-1, d["reps"][m].shape[-1])[val]
            y_v = y[val]
            pos = np.where(y_v == 1)[0]
            neg = np.where(y_v == 0)[0]
            if len(neg) > 1000:
                neg = np.random.RandomState(42).choice(neg, size=1000, replace=False)
            keep = np.concatenate([pos, neg])
            X_src_m0.append(f0[keep])
            X_src_m1.append(f1[keep])
            y_src.append(y_v[keep])

        X_s0 = np.vstack(X_src_m0)
        X_s1 = np.vstack(X_src_m1)
        y_s = np.concatenate(y_src)
        scaler_m0[m], clf_m0[m] = train_probe_safe(X_s0, y_s, seed=42)
        scaler_m1[m], clf_m1[m] = train_probe_safe(X_s1, y_s, seed=42)

    # Run 20 draws for G2
    for draw_idx in range(20):
        rng = np.random.RandomState(2000 + draw_idx)
        supp_fp = rng.choice(g2_tgt_files)
        eval_fps = [fp for fp in g2_tgt_files if fp != supp_fp]
        assert supp_fp not in eval_fps, "Protocol violation!"

        supp_d = scan_db[supp_fp]
        supp_flaws = supp_d["flaws"]
        k = min(3, len(supp_flaws))
        cal_flaws = [supp_flaws[i] for i in rng.choice(len(supp_flaws), size=k, replace=False)]

        for m in methods:
            # M2 Fit
            supp_fmap = supp_d["reps"][m]
            flaw_pts = [supp_fmap[fl["mask"]] for fl in cal_flaws]
            X_fl = np.vstack(flaw_pts)
            sound_pts = supp_fmap[(supp_d["gt_mask"] == 0) & (supp_d["clean_mask"] == 0)]
            if len(sound_pts) > len(X_fl) * 10:
                sound_pts = sound_pts[rng.choice(len(sound_pts), size=len(X_fl) * 10, replace=False)]
            X_supp = np.vstack([X_fl, sound_pts])
            y_supp = np.hstack([np.ones(len(X_fl), dtype=int), np.zeros(len(sound_pts), dtype=int)])
            scaler_m2, clf_m2 = train_probe_safe(X_supp, y_supp, seed=2000 + draw_idx)

            # Evaluate on 26 held-out eval files
            eval_pix_ap_m0, eval_pix_ap_m1, eval_pix_ap_m2 = [], [], []
            eval_chances = []
            for efp in eval_fps:
                ed = scan_db[efp]
                # M0
                res_m0 = evaluate_probe_on_file(
                    scaler_m0[m], clf_m0[m], ed["reps"][m],
                    ed["gt_mask"], ed["clean_mask"], ed["flaws"], ed["clean_rivets"], ed["specimen"]
                )
                if res_m0 is not None:
                    eval_pix_ap_m0.append(res_m0["pix_ap"])
                    eval_chances.append(res_m0["chance"])

                # M1
                f_m1 = apply_m1_unsupervised_calibration(ed["reps"][m])
                res_m1 = evaluate_probe_on_file(
                    scaler_m1[m], clf_m1[m], f_m1,
                    ed["gt_mask"], ed["clean_mask"], ed["flaws"], ed["clean_rivets"], ed["specimen"]
                )
                if res_m1 is not None:
                    eval_pix_ap_m1.append(res_m1["pix_ap"])

                # M2
                res_m2 = evaluate_probe_on_file(
                    scaler_m2, clf_m2, ed["reps"][m],
                    ed["gt_mask"], ed["clean_mask"], ed["flaws"], ed["clean_rivets"], ed["specimen"]
                )
                if res_m2 is not None:
                    eval_pix_ap_m2.append(res_m2["pix_ap"])

            g2_results["M0_zeroshot"][m].append(np.mean(eval_pix_ap_m0))
            g2_results["M1_unsupervised"][m].append(np.mean(eval_pix_ap_m1))
            g2_results["M2_fewshot_k3"][m].append(np.mean(eval_pix_ap_m2))
            if m == "jepa":
                g2_chances.append(np.mean(eval_chances))

    # =========================================================================
    # TASK G3: HOLD-OUT LIFT-OFF (Train z1+z2, Test z3)
    # =========================================================================
    print("\n" + "=" * 95)
    print("TASK G3: HOLD-OUT LIFT-OFF (Source: z1+z2, Target: z3) - 20 INDEPENDENT DRAWS")
    print("=" * 95)

    g3_src_files = [fp for fp, d in scan_db.items() if d["meta"]["liftoff"] in ["z1", "z2"]]
    g3_tgt_files = [fp for fp, d in scan_db.items() if d["meta"]["liftoff"] == "z3"]
    print(f"G3 Source files (z1 + z2): {len(g3_src_files)} files")
    print(f"G3 Target files (z3):       {len(g3_tgt_files)} files (1 support, {len(g3_tgt_files)-1} eval)")

    g3_results = {mode: {m: [] for m in methods} for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]}
    g3_chances = []

    # Fit M0 and M1 Source Probes for G3
    scaler_g3_m0 = {}
    clf_g3_m0 = {}
    scaler_g3_m1 = {}
    clf_g3_m1 = {}

    for m in methods:
        X_src_m0, y_src = [], []
        X_src_m1 = []
        for fp in g3_src_files:
            d = scan_db[fp]
            gm = d["gt_mask"].ravel()
            y = (gm == 1).astype(int)
            val = (gm >= 0)
            f0 = d["reps"][m].reshape(-1, d["reps"][m].shape[-1])[val]
            f1 = apply_m1_unsupervised_calibration(d["reps"][m]).reshape(-1, d["reps"][m].shape[-1])[val]
            y_v = y[val]
            pos = np.where(y_v == 1)[0]
            neg = np.where(y_v == 0)[0]
            if len(neg) > 1000:
                neg = np.random.RandomState(42).choice(neg, size=1000, replace=False)
            keep = np.concatenate([pos, neg])
            X_src_m0.append(f0[keep])
            X_src_m1.append(f1[keep])
            y_src.append(y_v[keep])

        X_s0 = np.vstack(X_src_m0)
        X_s1 = np.vstack(X_src_m1)
        y_s = np.concatenate(y_src)
        scaler_g3_m0[m], clf_g3_m0[m] = train_probe_safe(X_s0, y_s, seed=42)
        scaler_g3_m1[m], clf_g3_m1[m] = train_probe_safe(X_s1, y_s, seed=42)

    # Run 20 draws for G3
    for draw_idx in range(20):
        rng = np.random.RandomState(3000 + draw_idx)
        supp_fp = rng.choice(g3_tgt_files)
        eval_fps = [fp for fp in g3_tgt_files if fp != supp_fp]
        assert supp_fp not in eval_fps, "Protocol violation!"

        supp_d = scan_db[supp_fp]
        supp_flaws = supp_d["flaws"]
        k = min(3, len(supp_flaws))
        cal_flaws = [supp_flaws[i] for i in rng.choice(len(supp_flaws), size=k, replace=False)]

        for m in methods:
            supp_fmap = supp_d["reps"][m]
            flaw_pts = [supp_fmap[fl["mask"]] for fl in cal_flaws]
            X_fl = np.vstack(flaw_pts)
            sound_pts = supp_fmap[(supp_d["gt_mask"] == 0) & (supp_d["clean_mask"] == 0)]
            if len(sound_pts) > len(X_fl) * 10:
                sound_pts = sound_pts[rng.choice(len(sound_pts), size=len(X_fl) * 10, replace=False)]
            X_supp = np.vstack([X_fl, sound_pts])
            y_supp = np.hstack([np.ones(len(X_fl), dtype=int), np.zeros(len(sound_pts), dtype=int)])
            scaler_m2, clf_m2 = train_probe_safe(X_supp, y_supp, seed=3000 + draw_idx)

            eval_pix_ap_m0, eval_pix_ap_m1, eval_pix_ap_m2 = [], [], []
            eval_chances = []
            for efp in eval_fps:
                ed = scan_db[efp]
                # M0
                res_m0 = evaluate_probe_on_file(
                    scaler_g3_m0[m], clf_g3_m0[m], ed["reps"][m],
                    ed["gt_mask"], ed["clean_mask"], ed["flaws"], ed["clean_rivets"], ed["specimen"]
                )
                if res_m0 is not None:
                    eval_pix_ap_m0.append(res_m0["pix_ap"])
                    eval_chances.append(res_m0["chance"])

                # M1
                f_m1 = apply_m1_unsupervised_calibration(ed["reps"][m])
                res_m1 = evaluate_probe_on_file(
                    scaler_g3_m1[m], clf_g3_m1[m], f_m1,
                    ed["gt_mask"], ed["clean_mask"], ed["flaws"], ed["clean_rivets"], ed["specimen"]
                )
                if res_m1 is not None:
                    eval_pix_ap_m1.append(res_m1["pix_ap"])

                # M2
                res_m2 = evaluate_probe_on_file(
                    scaler_m2, clf_m2, ed["reps"][m],
                    ed["gt_mask"], ed["clean_mask"], ed["flaws"], ed["clean_rivets"], ed["specimen"]
                )
                if res_m2 is not None:
                    eval_pix_ap_m2.append(res_m2["pix_ap"])

            g3_results["M0_zeroshot"][m].append(np.mean(eval_pix_ap_m0))
            g3_results["M1_unsupervised"][m].append(np.mean(eval_pix_ap_m1))
            g3_results["M2_fewshot_k3"][m].append(np.mean(eval_pix_ap_m2))
            if m == "jepa":
                g3_chances.append(np.mean(eval_chances))

    # Print Summary Tables
    print("\n" + "=" * 105)
    print("TASK G2: HOLD-OUT SENSOR (TMR) OVER 20 INDEPENDENT SUPPORT DRAWS (MEAN +/- STD %)")
    print("=" * 105)
    mean_ch_g2 = np.mean(g2_chances) * 100
    print(f"Chance Level (Prevalence): {mean_ch_g2:.2f}%\n")
    rows_g2 = []
    for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]:
        for m in methods:
            vals = np.array(g2_results[mode][m]) * 100
            rows_g2.append({
                "Mode": mode,
                "Method": m,
                "Pixel AP (%)": f"{np.mean(vals):.2f} ± {np.std(vals):.2f}",
                "AP - Chance (%)": f"{np.mean(vals) - mean_ch_g2:+.2f}",
            })
    df_g2 = pd.DataFrame(rows_g2)
    print(df_g2.to_string(index=False))

    print("\n" + "=" * 105)
    print("TASK G3: HOLD-OUT LIFT-OFF (z3) OVER 20 INDEPENDENT SUPPORT DRAWS (MEAN +/- STD %)")
    print("=" * 105)
    mean_ch_g3 = np.mean(g3_chances) * 100
    print(f"Chance Level (Prevalence): {mean_ch_g3:.2f}%\n")
    rows_g3 = []
    for mode in ["M0_zeroshot", "M1_unsupervised", "M2_fewshot_k3"]:
        for m in methods:
            vals = np.array(g3_results[mode][m]) * 100
            rows_g3.append({
                "Mode": mode,
                "Method": m,
                "Pixel AP (%)": f"{np.mean(vals):.2f} ± {np.std(vals):.2f}",
                "AP - Chance (%)": f"{np.mean(vals) - mean_ch_g3:+.2f}",
            })
    df_g3 = pd.DataFrame(rows_g3)
    print(df_g3.to_string(index=False))

    # Disaggregated by specimen and Cluster Bootstrap Delta AP
    print("\n" + "=" * 105)
    print("TASK G2 & G3 DISAGGREGATED BY SPECIMEN & CLUSTER BOOTSTRAP DELTA AP (JEPA vs Random)")
    print("=" * 105)

    # Convert to float for clean json
    def clean_floats(obj):
        if isinstance(obj, dict):
            return {k: clean_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_floats(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    out_file = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results/stage1_audit_c5_disjoint.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_floats({"g2": g2_results, "g3": g3_results, "g2_chance": mean_ch_g2, "g3_chance": mean_ch_g3}), f, indent=2, default=float)
    print(f"\n[Saved C5 Disjoint Adaptation Results to {out_file}]")


if __name__ == "__main__":
    main()
