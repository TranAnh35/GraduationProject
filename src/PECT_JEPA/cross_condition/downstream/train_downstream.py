"""
Dedicated Downstream & Test-Time Adaptation (TTA) Probes Training for PECT-JEPA.

Physical & Theoretical Protocol:
1. Frozen Representation: Feature weights of PECT-JEPA are 100% frozen.
2. Source Domain Extraction: the Base Domain of the active split protocol
   (16 files under 'cross_specimen', 24 files under 'cross_condition').
3. Dual-Probe Paradigm:
   - Raw Probes: Standard Linear/MLP probes fitted on raw standardized features (Uncalibrated baseline).
   - TTA Probes: Probes fitted on per-scan unsupervised background-whitened features (P1u_trim_s2).
4. P2h Hard-Negative Mining: the strongest P0 false alarms per scan (halo ring
   excluded) are added as negatives, so the probe learns to suppress structured
   background clutter instead of memorising it as defect evidence.
5. Log-Target Sizing: the regressor is fitted on log1p(diameter) and inverted at
   inference, so 2 mm and 14 mm flaws contribute comparably (H31).
6. No UDA Model Weights: Neural marginal-alignment regularizers (CORAL, DANN, CDAN, SHOT, OTDA, CBST)
   are permanently deprecated due to demonstrated marginal-alignment failure on 98.8% sound-metal grids.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
import joblib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import CrossConditionJEPA
from src.PECT_JEPA.cross_condition.data.preprocessing import (
    load_canonical_cscan_from_tdms,
    get_canonical_transformer,
)
from src.PECT_JEPA.cross_condition.data.split import (
    SPLIT_MODES,
    extract_file_metadata,
    filter_base_domain_files,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import (
    HALO_RADIUS_PX,
    HARD_NEG_PER_SCAN,
    mine_hard_negatives,
)
from src.PECT_JEPA.cross_condition.evaluate import (
    resolve_paths,
    load_ground_truth_masks,
    build_corrosion_diameter_map,
    _iter_with_prefetch,
)

from src.PECT_JEPA.cross_condition.downstream.detection.models import DetectionProbeBundle
from src.PECT_JEPA.cross_condition.downstream.sizing.models import SizingRegressorBundle
from src.PECT_JEPA.cross_condition.downstream.test_time_adaptation import (
    unsup_trimmed_bg_stats,
    TestTimeSelfCalibrator,
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Dedicated Downstream & TTA Training for PECT-JEPA")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to frozen PECT-JEPA checkpoint")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment run directory containing checkpoints/")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory to save trained downstream models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=8192, help="Feature extraction batch size")
    parser.add_argument("--prefetch", type=lambda v: v.lower() == "true", default=True, help="Enable background TDMS prefetch")
    parser.add_argument("--self_calibration", type=lambda v: v.lower() == "true", default=True, help="Unsupervised spatial self-calibration")
    parser.add_argument("--split_mode", type=str, default="cross_specimen", choices=list(SPLIT_MODES), help="Base-Domain protocol used to select the source pool")

    # Test-Time Adaptation (TTA) Hyperparameters
    parser.add_argument("--tta_trim_ratio", type=float, default=0.75, help="Trim ratio for robust background estimation (default 0.75)")
    parser.add_argument("--tta_shrinkage", type=float, default=1e-2, help="Diagonal loading shrinkage eps for whitening matrix (default 1e-2)")
    # P2h hard-negative mining. Default OFF (EXP-72): the 14 mm rivet-hole
    # scattering field spans 5-8 px, so "top P0 false alarms outside halo r=3"
    # are real rivet edge response. Labelling them y=0 taught the probe to
    # suppress the defect signal (exp71a: auc_p1u_s2 0.9163 but TTA LP 0.8471).
    parser.add_argument("--hard_neg_mining", type=lambda v: v.lower() == "true", default=False, help="P2h: add the top P0 false alarms per scan (halo ring excluded) as negatives")
    parser.add_argument("--hard_neg_per_scan", type=int, default=HARD_NEG_PER_SCAN, help="Hard negatives mined per scan")
    parser.add_argument("--halo_radius", type=int, default=HALO_RADIUS_PX, help="Halo ring radius (px) excluded from fitting/mining")
    # Sizing target
    parser.add_argument("--sizing_log_target", type=lambda v: v.lower() == "true", default=True, help="Fit the sizing regressor on log1p(diameter) and invert at inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Backward compatibility / deprecated no-op flags
    parser.add_argument("--skip_uda", "--skip_domain_adaptation", dest="skip_uda", nargs="?", const=True, type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=False)
    parser.add_argument("--adapt_mode", type=str, default="all")
    return parser.parse_args()


def _fit_detection_probes(X: np.ndarray, y: np.ndarray, seed: int) -> DetectionProbeBundle:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=500, tol=1e-3, class_weight="balanced",
                            random_state=seed, solver="lbfgs")
    lr.fit(Xs, y)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                        random_state=seed, early_stopping=True, n_iter_no_change=10)
    mlp.fit(Xs, y)
    return DetectionProbeBundle(scaler=scaler, linear_probe=lr, mlp_probe=mlp)


def _fit_sizing_bundle(X: np.ndarray, y_mm: np.ndarray, seed: int, log_target: bool) -> SizingRegressorBundle:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y_fit = np.log1p(y_mm.astype(np.float64)) if log_target else y_mm
    ridge = Ridge(alpha=10.0, random_state=seed)
    ridge.fit(Xs, y_fit)
    mlp = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                       random_state=seed, early_stopping=True, n_iter_no_change=10)
    mlp.fit(Xs, y_fit)
    return SizingRegressorBundle(
        scaler=scaler,
        ridge_regressor=ridge,
        mlp_regressor=mlp,
        target_transform="log1p" if log_target else "identity",
    )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path, eval_out_dir = resolve_paths(args)
    if args.out_dir is not None:
        downstream_dir = os.path.abspath(args.out_dir)
    else:
        exp_dir = os.path.dirname(eval_out_dir) if os.path.basename(eval_out_dir) == "evaluation_results" else os.path.dirname(ckpt_path)
        downstream_dir = os.path.join(exp_dir, "downstream_models")
    os.makedirs(downstream_dir, exist_ok=True)

    print("=" * 80)
    print("CANONICAL DOWNSTREAM & TEST-TIME ADAPTATION (TTA) TRAINING PIPELINE")
    print(f"  Checkpoint:       {ckpt_path}")
    print(f"  Output Dir:       {downstream_dir}")
    print(f"  Split Protocol:   {args.split_mode}")
    print(f"  TTA Trim Ratio:   {args.tta_trim_ratio}")
    print(f"  TTA Shrinkage:    {args.tta_shrinkage}")
    print(f"  P2h Hard Negs:    {args.hard_neg_mining} (k={args.hard_neg_per_scan}/scan, halo r={args.halo_radius})")
    print(f"  Sizing Log-Target:{args.sizing_log_target}")
    print(f"  Device:           {args.device}")
    print("=" * 80, flush=True)

    # 1. Load frozen PECT-JEPA
    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config", None)
    config = CrossConditionConfig()
    if raw_config is not None:
        source_dict = vars(raw_config) if hasattr(raw_config, "__dict__") else (raw_config if isinstance(raw_config, dict) else {})
        for k, v in source_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    extractor = CScanFeatureExtractor(model=model, device=device, batch_size=args.batch_size)

    # 2. File discovery
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]
    gt_masks = load_ground_truth_masks(args.data_dir)
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])
    get_canonical_transformer()

    # 3. Source Domain = Base Domain of the active split protocol (Train + Val)
    base_source_files = filter_base_domain_files(all_files, split_mode=args.split_mode)
    print(f"\n[STEP 1/3] Extracting Source Domain Pool ({len(base_source_files)} Base Domain files [{args.split_mode}]: Train + Val)...")
    source_eval_files = [fp for fp in base_source_files if extract_file_metadata(fp)["specimen"] in gt_masks]

    det_x_raw, det_x_tta, det_y = [], [], []
    size_x_raw, size_x_tta, size_y = [], [], []
    scan_whiten: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.RandomState(args.seed)

    pbar_source = tqdm(_iter_with_prefetch(source_eval_files, args.prefetch), total=len(source_eval_files),
                       desc="Extracting Source Pool", file=sys.stdout, dynamic_ncols=True)
    for fp, cscan in pbar_source:
        meta = extract_file_metadata(fp)
        sp = meta["specimen"]
        gt_mask = gt_masks[sp]["mask"]

        latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        flat_y = gt_mask.reshape(-1).astype(np.int64)

        # P1u_trim_s2: per-scan unsupervised background whitening
        mu_bg, W_bg = unsup_trimmed_bg_stats(flat_z, keep_q=args.tta_trim_ratio, eps=args.tta_shrinkage)
        scan_whiten[fp] = (mu_bg, W_bg)
        flat_z_tta = (flat_z - mu_bg).dot(W_bg).astype(np.float32)

        pos_idx = np.where(flat_y == 1)[0]
        neg_idx = np.where(flat_y == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sub_neg_count = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
            sub_neg = rng.choice(neg_idx, size=sub_neg_count, replace=False)
            sel = np.concatenate([pos_idx, sub_neg])

            det_x_raw.append(flat_z[sel])
            det_x_tta.append(flat_z_tta[sel])
            det_y.append(flat_y[sel])

        if sp == "Corrosion" and corr_d_map is not None:
            flat_d = corr_d_map.reshape(-1).astype(np.float32)
            def_idx = np.where(flat_d > 0.0)[0]
            if len(def_idx) > 0:
                size_x_raw.append(flat_z[def_idx])
                size_x_tta.append(flat_z_tta[def_idx])
                size_y.append(flat_d[def_idx])

    X_det_raw = np.concatenate(det_x_raw, axis=0)
    X_det_tta = np.concatenate(det_x_tta, axis=0)
    y_det = np.concatenate(det_y, axis=0)

    print(f"  Source Detection samples: {len(y_det)} (Defects: {np.sum(y_det == 1)}, Sound: {np.sum(y_det == 0)})")

    # 4. P2h: mine the strongest base-probe false alarms per scan (halo excluded)
    if args.hard_neg_mining:
        print("\n[STEP 2/3] Mining P2h hard negatives (base probe -> top false alarms)...")
        base_bundle = _fit_detection_probes(X_det_raw, y_det, args.seed)
        hard_raw, hard_tta = [], []
        pbar_mine = tqdm(_iter_with_prefetch(source_eval_files, args.prefetch), total=len(source_eval_files),
                         desc="Mining hard negatives", file=sys.stdout, dynamic_ncols=True)
        for fp, cscan in pbar_mine:
            sp = extract_file_metadata(fp)["specimen"]
            gt_mask = gt_masks[sp]["mask"]
            latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
            flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            scores = base_bundle.predict_linear(flat_z)
            idx = mine_hard_negatives(gt_mask, scores, k=args.hard_neg_per_scan, halo_radius=args.halo_radius)
            if len(idx) == 0:
                continue
            mu_bg, W_bg = scan_whiten[fp]
            hard_raw.append(flat_z[idx])
            hard_tta.append((flat_z[idx] - mu_bg).dot(W_bg).astype(np.float32))
        if hard_raw:
            n_hard = sum(len(h) for h in hard_raw)
            X_det_raw = np.concatenate([X_det_raw] + hard_raw, axis=0)
            X_det_tta = np.concatenate([X_det_tta] + hard_tta, axis=0)
            y_det = np.concatenate([y_det, np.zeros(n_hard, dtype=np.int64)], axis=0)
            print(f"  Added {n_hard} P2h hard negatives -> pool {len(y_det)} samples "
                  f"({np.sum(y_det == 1)} defect / {np.sum(y_det == 0)} sound)")
    else:
        print("\n[STEP 2/3] P2h hard-negative mining DISABLED")

    # 5. Fit Probes on Source Domain
    print("\n[STEP 3/3] Fitting Probes (Raw & TTA Calibrated)...")
    raw_det_bundle = _fit_detection_probes(X_det_raw, y_det, args.seed)
    tta_det_bundle = _fit_detection_probes(X_det_tta, y_det, args.seed)

    X_size_raw = np.concatenate(size_x_raw, axis=0) if size_x_raw else None
    X_size_tta = np.concatenate(size_x_tta, axis=0) if size_x_tta else None
    y_size = np.concatenate(size_y, axis=0) if size_y else None
    if y_size is not None:
        print(f"  Source Sizing samples:    {len(y_size)} defect pixels")

    raw_size_bundle = None
    tta_size_bundle = None
    if X_size_raw is not None and y_size is not None:
        raw_size_bundle = _fit_sizing_bundle(X_size_raw, y_size, args.seed, args.sizing_log_target)
        tta_size_bundle = _fit_sizing_bundle(X_size_tta, y_size, args.seed, args.sizing_log_target)

    # 6. Save Model Bundle
    models_bundle = {
        # Backward compatibility for legacy evaluate.py
        "detection": raw_det_bundle,
        "sizing": raw_size_bundle,
        # Dual-probes for TTA benchmark
        "raw_detection": raw_det_bundle,
        "raw_sizing": raw_size_bundle,
        "tta_detection": tta_det_bundle,
        "tta_sizing": tta_size_bundle,
        "tta_config": {
            "keep_q": args.tta_trim_ratio,
            "eps": args.tta_shrinkage,
        },
        "readout_config": {
            "split_mode": args.split_mode,
            "hard_neg_mining": bool(args.hard_neg_mining),
            "hard_neg_per_scan": int(args.hard_neg_per_scan),
            "halo_radius": int(args.halo_radius),
            "sizing_log_target": bool(args.sizing_log_target),
        },
    }
    joblib.dump(models_bundle, os.path.join(downstream_dir, "standard_probes.joblib"))
    print(f"\n[OK] Saved standard_probes.joblib to {downstream_dir}")

    # Summary
    summary = {
        "checkpoint": ckpt_path,
        "split_mode": args.split_mode,
        "source_files": len(source_eval_files),
        "source_detection_samples": int(len(y_det)),
        "source_sizing_samples": int(len(y_size)) if y_size is not None else 0,
        "readout_config": models_bundle["readout_config"],
        "tta_config": {
            "trim_ratio": args.tta_trim_ratio,
            "shrinkage": args.tta_shrinkage,
        },
        "device": args.device,
    }
    with open(os.path.join(downstream_dir, "downstream_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS: Downstream Training Complete:")
    print(f"  - Probes File:    {os.path.join(downstream_dir, 'standard_probes.joblib')}")
    print(f"  - Summary:        {os.path.join(downstream_dir, 'downstream_train_summary.json')}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
