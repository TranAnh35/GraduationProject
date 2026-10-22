"""
Dedicated Downstream & Test-Time Adaptation (TTA) Probes Training for PECT-JEPA.

Physical & Theoretical Protocol:
1. Frozen Representation: Feature weights of PECT-JEPA are 100% frozen.
2. Source Domain Extraction: 16 Base Domain files (Train + Val splits).
3. Dual-Probe Paradigm:
   - Raw Probes: Standard Linear/MLP probes fitted on raw standardized features (Uncalibrated baseline).
   - TTA Probes: Probes fitted on per-scan unsupervised background-whitened features (Test-Time Adaptation baseline).
4. No UDA Model Weights: Neural marginal-alignment regularizers (CORAL, DANN, CDAN, SHOT, OTDA, CBST)
   are permanently deprecated due to demonstrated marginal-alignment failure on 98.8% sound-metal grids.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import argparse
from typing import Dict, List, Any, Optional
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
    extract_file_metadata,
    get_canonical_train_val_test_splits,
    is_base_domain_file,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
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
    
    # Test-Time Adaptation (TTA) Hyperparameters
    parser.add_argument("--tta_trim_ratio", type=float, default=0.75, help="Trim ratio for robust background estimation (default 0.75)")
    parser.add_argument("--tta_shrinkage", type=float, default=1e-2, help="Diagonal loading shrinkage eps for whitening matrix (default 1e-2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Backward compatibility / deprecated no-op flags
    parser.add_argument("--skip_uda", "--skip_domain_adaptation", dest="skip_uda", nargs="?", const=True, type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=False)
    parser.add_argument("--adapt_mode", type=str, default="all")
    return parser.parse_args()


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
    print(f"  TTA Trim Ratio:   {args.tta_trim_ratio}")
    print(f"  TTA Shrinkage:    {args.tta_shrinkage}")
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
    gt_masks = load_ground_truth_masks()
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])
    get_canonical_transformer()

    # 3. Extract Source Domain Data (16 Base Domain files: train + val)
    base_source_files = [fp for fp in all_files if is_base_domain_file(extract_file_metadata(fp))]
    print(f"\n[STEP 1/2] Extracting Source Domain Pool (16 Base Domain files: Train + Val)...")
    source_eval_files = [fp for fp in base_source_files if extract_file_metadata(fp)["specimen"] in gt_masks]
    
    train_det_x_raw, train_det_x_tta, train_det_y = [], [], []
    train_size_x_raw, train_size_x_tta, train_size_y = [], [], []
    rng = np.random.RandomState(args.seed)

    pbar_source = tqdm(_iter_with_prefetch(source_eval_files, args.prefetch), total=len(source_eval_files),
                       desc="Extracting Source Pool", file=sys.stdout, dynamic_ncols=True)
    for fp, cscan in pbar_source:
        meta = extract_file_metadata(fp)
        sp = meta["specimen"]
        gt_mask = gt_masks[sp]["mask"]
        
        # Raw features
        latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        flat_y = gt_mask.reshape(-1).astype(np.int64)

        # TTA-whitened features
        mu_bg, W_bg = unsup_trimmed_bg_stats(flat_z, keep_q=args.tta_trim_ratio, eps=args.tta_shrinkage)
        flat_z_tta = (flat_z - mu_bg).dot(W_bg).astype(np.float32)

        pos_idx = np.where(flat_y == 1)[0]
        neg_idx = np.where(flat_y == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sub_neg_count = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
            sub_neg = rng.choice(neg_idx, size=sub_neg_count, replace=False)
            sel = np.concatenate([pos_idx, sub_neg])
            
            train_det_x_raw.append(flat_z[sel])
            train_det_x_tta.append(flat_z_tta[sel])
            train_det_y.append(flat_y[sel])

        if sp == "Corrosion" and corr_d_map is not None:
            flat_d = corr_d_map.reshape(-1).astype(np.float32)
            def_idx = np.where(flat_d > 0.0)[0]
            if len(def_idx) > 0:
                train_size_x_raw.append(flat_z[def_idx])
                train_size_x_tta.append(flat_z_tta[def_idx])
                train_size_y.append(flat_d[def_idx])

    X_train_det_raw = np.concatenate(train_det_x_raw, axis=0)
    X_train_det_tta = np.concatenate(train_det_x_tta, axis=0)
    y_train_det = np.concatenate(train_det_y, axis=0)

    X_train_size_raw = np.concatenate(train_size_x_raw, axis=0) if train_size_x_raw else None
    X_train_size_tta = np.concatenate(train_size_x_tta, axis=0) if train_size_x_tta else None
    y_train_size = np.concatenate(train_size_y, axis=0) if train_size_y else None

    print(f"  Source Detection samples: {len(y_train_det)} (Defects: {np.sum(y_train_det == 1)}, Sound: {np.sum(y_train_det == 0)})")
    if y_train_size is not None:
        print(f"  Source Sizing samples:    {len(y_train_size)} defect pixels")

    # 4. Fit Probes on Source Domain
    print("\n[STEP 2/2] Fitting Probes (Raw & TTA Calibrated)...")
    
    # 4a. Raw Probes (Uncalibrated Zero-Shot Baseline)
    scaler_det_raw = StandardScaler()
    X_det_raw_s = scaler_det_raw.fit_transform(X_train_det_raw)
    lr_raw = LogisticRegression(C=1.0, max_iter=500, tol=1e-3, class_weight="balanced",
                                random_state=args.seed, solver="lbfgs")
    lr_raw.fit(X_det_raw_s, y_train_det)
    mlp_raw = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                            random_state=args.seed, early_stopping=True, n_iter_no_change=10)
    mlp_raw.fit(X_det_raw_s, y_train_det)
    raw_det_bundle = DetectionProbeBundle(scaler=scaler_det_raw, linear_probe=lr_raw, mlp_probe=mlp_raw)

    raw_size_bundle = None
    if X_train_size_raw is not None and y_train_size is not None:
        scaler_size_raw = StandardScaler()
        X_size_raw_s = scaler_size_raw.fit_transform(X_train_size_raw)
        ridge_raw = Ridge(alpha=10.0, random_state=args.seed)
        ridge_raw.fit(X_size_raw_s, y_train_size)
        mlp_reg_raw = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                                   random_state=args.seed, early_stopping=True, n_iter_no_change=10)
        mlp_reg_raw.fit(X_size_raw_s, y_train_size)
        raw_size_bundle = SizingRegressorBundle(scaler=scaler_size_raw, ridge_regressor=ridge_raw, mlp_regressor=mlp_reg_raw)

    # 4b. TTA Calibrated Probes (Trained on Background-Whitened Canonical Manifold)
    scaler_det_tta = StandardScaler()
    X_det_tta_s = scaler_det_tta.fit_transform(X_train_det_tta)
    lr_tta = LogisticRegression(C=1.0, max_iter=500, tol=1e-3, class_weight="balanced",
                                random_state=args.seed, solver="lbfgs")
    lr_tta.fit(X_det_tta_s, y_train_det)
    mlp_tta = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                            random_state=args.seed, early_stopping=True, n_iter_no_change=10)
    mlp_tta.fit(X_det_tta_s, y_train_det)
    tta_det_bundle = DetectionProbeBundle(scaler=scaler_det_tta, linear_probe=lr_tta, mlp_probe=mlp_tta)

    tta_size_bundle = None
    if X_train_size_tta is not None and y_train_size is not None:
        scaler_size_tta = StandardScaler()
        X_size_tta_s = scaler_size_tta.fit_transform(X_train_size_tta)
        ridge_tta = Ridge(alpha=10.0, random_state=args.seed)
        ridge_tta.fit(X_size_tta_s, y_train_size)
        mlp_reg_tta = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                                   random_state=args.seed, early_stopping=True, n_iter_no_change=10)
        mlp_reg_tta.fit(X_size_tta_s, y_train_size)
        tta_size_bundle = SizingRegressorBundle(scaler=scaler_size_tta, ridge_regressor=ridge_tta, mlp_regressor=mlp_reg_tta)

    # 5. Save Model Bundle
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
    }
    joblib.dump(models_bundle, os.path.join(downstream_dir, "standard_probes.joblib"))
    print(f"\n[OK] Saved standard_probes.joblib to {downstream_dir}")

    # Summary
    summary = {
        "checkpoint": ckpt_path,
        "source_files": len(source_eval_files),
        "source_detection_samples": int(len(y_train_det)),
        "source_sizing_samples": int(len(y_train_size)) if y_train_size is not None else 0,
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
