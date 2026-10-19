"""
Dedicated Training Pipeline for Downstream Tasks & Unsupervised Domain Adaptation (UDA).

Trains downstream models on top of frozen PECT-JEPA representations:
1. Standard Probes:
   - Detection: Linear Probe (Logistic Regression) & 2-Layer MLP
   - Sizing: Ridge Regressor & MLP Regressor
2. Unsupervised Domain Adaptation (UDA) Models:
   - CORAL (Correlation Alignment): Matches covariance matrices of source and target domains
   - DANN (Domain-Adversarial Neural Network): Feature adapter with domain discriminator & GRL

Outputs saved to: <exp_dir>/downstream_models/
  ├── standard_probes.joblib
  ├── coral_model.pt
  ├── dann_model.pt
  └── downstream_train_summary.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import joblib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.coral import CORALAdapter, coral_loss
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.dann import DANNModel

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Dedicated Downstream & UDA Training for PECT-JEPA")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to frozen PECT-JEPA checkpoint")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment run directory containing checkpoints/")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory to save trained downstream models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=8192, help="Feature extraction batch size")
    parser.add_argument("--prefetch", type=lambda v: v.lower() == "true", default=True, help="Enable background TDMS prefetch")
    parser.add_argument("--self_calibration", type=lambda v: v.lower() == "true", default=True, help="Unsupervised spatial self-calibration")
    
    # UDA Training Parameters
    parser.add_argument("--uda_batch_size", type=int, default=256, help="Batch size for UDA neural training")
    parser.add_argument("--coral_epochs", type=int, default=25, help="Number of training epochs for CORAL")
    parser.add_argument("--coral_weight", type=float, default=1.0, help="Weight lambda for CORAL covariance loss")
    parser.add_argument("--dann_epochs", type=int, default=25, help="Number of training epochs for DANN")
    parser.add_argument("--dann_weight", type=float, default=0.1, help="Weight lambda for DANN domain discriminator loss")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for UDA optimizers")
    parser.add_argument("--target_sample_size", type=int, default=50000, help="Number of unlabeled target pixels to subsample for UDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def train_coral(
    X_s: np.ndarray,
    y_s: np.ndarray,
    X_t: np.ndarray,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    coral_weight: float = 1.0,
    device: str = "cpu",
) -> CORALAdapter:
    """Trains a CORALAdapter on source labeled data and target unlabeled data."""
    in_dim = X_s.shape[1]
    model = CORALAdapter(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Calculate class weighting for BCE
    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Tensor datasets
    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())

    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [CORAL UDA] Training for {epochs} epochs (lambda_coral={coral_weight})...")
    for epoch in range(epochs):
        total_loss, total_task, total_coral = 0.0, 0.0, 0.0
        iter_t = iter(loader_t)

        for batch_s, labels_s in loader_s:
            try:
                batch_t = next(iter_t)[0]
            except StopIteration:
                iter_t = iter(loader_t)
                batch_t = next(iter_t)[0]

            batch_s, labels_s = batch_s.to(device), labels_s.to(device)
            batch_t = batch_t.to(device)

            optimizer.zero_grad()
            z_s_adapted = model.adapt(batch_s)
            z_t_adapted = model.adapt(batch_t)

            logits_s = model.classifier(z_s_adapted).squeeze(-1)
            loss_task = criterion_task(logits_s, labels_s)
            loss_coral = coral_loss(z_s_adapted, z_t_adapted)

            loss = loss_task + coral_weight * loss_coral
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_coral += loss_coral.item()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, CORAL={total_coral/n_b:.4f})")

    model.eval()
    return model


def train_dann(
    X_s: np.ndarray,
    y_s: np.ndarray,
    X_t: np.ndarray,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    dann_weight: float = 0.1,
    device: str = "cpu",
) -> DANNModel:
    """Trains a DANNModel using Gradient Reversal Layer (GRL)."""
    in_dim = X_s.shape[1]
    model = DANNModel(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_domain = nn.BCEWithLogitsLoss()

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())

    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [DANN UDA] Training for {epochs} epochs (lambda_dann={dann_weight})...")
    for epoch in range(epochs):
        total_loss, total_task, total_domain = 0.0, 0.0, 0.0
        iter_t = iter(loader_t)
        p = float(epoch) / max(1, epochs)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Dynamic GRL schedule

        for batch_s, labels_s in loader_s:
            try:
                batch_t = next(iter_t)[0]
            except StopIteration:
                iter_t = iter(loader_t)
                batch_t = next(iter_t)[0]

            batch_s, labels_s = batch_s.to(device), labels_s.to(device)
            batch_t = batch_t.to(device)

            optimizer.zero_grad()

            # Source forward
            class_logits_s, domain_logits_s = model(batch_s, alpha=alpha)
            loss_task = criterion_task(class_logits_s, labels_s)
            domain_label_s = torch.zeros_like(domain_logits_s)
            loss_domain_s = criterion_domain(domain_logits_s, domain_label_s)

            # Target forward
            _, domain_logits_t = model(batch_t, alpha=alpha)
            domain_label_t = torch.ones_like(domain_logits_t)
            loss_domain_t = criterion_domain(domain_logits_t, domain_label_t)

            loss_domain = 0.5 * (loss_domain_s + loss_domain_t)
            loss = loss_task + dann_weight * loss_domain

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_domain += loss_domain.item()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, Domain={total_domain/n_b:.4f}, alpha={alpha:.2f})")

    model.eval()
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path, eval_out_dir = resolve_paths(args)
    if args.out_dir is not None:
        downstream_dir = os.path.abspath(args.out_dir)
    else:
        # Default: <exp_dir>/downstream_models/
        exp_dir = os.path.dirname(eval_out_dir) if os.path.basename(eval_out_dir) == "evaluation_results" else os.path.dirname(ckpt_path)
        downstream_dir = os.path.join(exp_dir, "downstream_models")
    os.makedirs(downstream_dir, exist_ok=True)

    print("=" * 80)
    print(f"DOWNSTREAM & UDA TRAINING PIPELINE")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output Dir: {downstream_dir}")
    print(f"  Device:     {args.device}")
    print("=" * 80, flush=True)

    # 1. Load frozen PECT-JEPA model
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

    # 2. Partitions & Data Loading
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]
    splits = get_canonical_train_val_test_splits(all_files, val_files_count=4, seed=42)
    gt_masks = load_ground_truth_masks()
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])
    get_canonical_transformer()

    print(f"Partition: {len(splits['train'])} Train, {len(splits['val'])} Val, {len(splits['test'])} Test files.")

    # 3. Extract Source Domain (Train Split) Data
    print("\n[STEP 1/3] Extracting Source Domain (Train Pool) Representations...")
    train_eval_files = [fp for fp in splits["train"] if extract_file_metadata(fp)["specimen"] in gt_masks]
    train_det_x, train_det_y = [], []
    train_size_x, train_size_y = [], []
    rng = np.random.RandomState(args.seed)

    pbar_train = tqdm(_iter_with_prefetch(train_eval_files, args.prefetch), total=len(train_eval_files),
                      desc="Extracting Source Pool", file=sys.stdout, dynamic_ncols=True)
    for fp, cscan in pbar_train:
        meta = extract_file_metadata(fp)
        sp = meta["specimen"]
        gt_mask = gt_masks[sp]["mask"]
        latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        flat_y = gt_mask.reshape(-1).astype(np.int64)

        pos_idx = np.where(flat_y == 1)[0]
        neg_idx = np.where(flat_y == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sub_neg_count = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
            sub_neg = rng.choice(neg_idx, size=sub_neg_count, replace=False)
            sel = np.concatenate([pos_idx, sub_neg])
            train_det_x.append(flat_z[sel])
            train_det_y.append(flat_y[sel])

        if sp == "Corrosion" and corr_d_map is not None:
            flat_d = corr_d_map.reshape(-1).astype(np.float32)
            def_idx = np.where(flat_d > 0.0)[0]
            if len(def_idx) > 0:
                train_size_x.append(flat_z[def_idx])
                train_size_y.append(flat_d[def_idx])

    X_train_det = np.concatenate(train_det_x, axis=0)
    y_train_det = np.concatenate(train_det_y, axis=0)
    X_train_size = np.concatenate(train_size_x, axis=0) if train_size_x else None
    y_train_size = np.concatenate(train_size_y, axis=0) if train_size_y else None
    print(f"  Source Detection samples: {len(y_train_det)} (Defects: {np.sum(y_train_det == 1)}, Sound: {np.sum(y_train_det == 0)})")
    if y_train_size is not None:
        print(f"  Source Sizing samples:    {len(y_train_size)} defect pixels")

    # 4. Extract Target Domain (Unlabeled Test Pool) Data for UDA
    print("\n[STEP 2/3] Extracting Unlabeled Target Pool Representations for UDA...")
    target_files = splits["test"]
    # Sample subset of test files across diverse sensors/waveforms if many
    sample_files = target_files[::max(1, len(target_files) // 16)]
    target_z_list = []
    pbar_target = tqdm(_iter_with_prefetch(sample_files, args.prefetch), total=len(sample_files),
                       desc="Extracting Target Pool", file=sys.stdout, dynamic_ncols=True)
    for fp, cscan in pbar_target:
        latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        # Random sample pixels from this scan (strictly zero GT labels used)
        sample_n = min(len(flat_z), max(2000, args.target_sample_size // len(sample_files)))
        idx = rng.choice(len(flat_z), size=sample_n, replace=False)
        target_z_list.append(flat_z[idx])

    X_target = np.concatenate(target_z_list, axis=0)
    if len(X_target) > args.target_sample_size:
        X_target = X_target[rng.choice(len(X_target), size=args.target_sample_size, replace=False)]
    print(f"  Unlabeled Target samples: {len(X_target)} pixels (from {len(sample_files)} OOD scans)")

    # 5. Train Downstream Probes & UDA Models
    print("\n[STEP 3/3] Training Downstream Probes & Domain Adaptation Models...")

    # A. Standard Detection Probes
    print("  Fitting Standard Linear Probe & MLP Classifier...")
    scaler_det = StandardScaler()
    X_train_det_s = scaler_det.fit_transform(X_train_det)
    lr_probe = LogisticRegression(C=1.0, max_iter=500, tol=1e-3, class_weight="balanced",
                                  random_state=args.seed, solver="lbfgs")
    lr_probe.fit(X_train_det_s, y_train_det)

    mlp_probe = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                              random_state=args.seed, early_stopping=True, n_iter_no_change=10)
    mlp_probe.fit(X_train_det_s, y_train_det)

    det_bundle = DetectionProbeBundle(scaler=scaler_det, linear_probe=lr_probe, mlp_probe=mlp_probe)

    # B. Flaw Sizing Regressors
    size_bundle = None
    if X_train_size is not None and y_train_size is not None:
        print("  Fitting Flaw Sizing Regressors...")
        scaler_size = StandardScaler()
        X_train_size_s = scaler_size.fit_transform(X_train_size)
        ridge_reg = Ridge(alpha=10.0, random_state=args.seed)
        ridge_reg.fit(X_train_size_s, y_train_size)
        mlp_reg = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                               random_state=args.seed, early_stopping=True, n_iter_no_change=10)
        mlp_reg.fit(X_train_size_s, y_train_size)
        size_bundle = SizingRegressorBundle(scaler=scaler_size, ridge_regressor=ridge_reg, mlp_regressor=mlp_reg)

    # C. CORAL UDA Model
    coral_model = train_coral(
        X_s=X_train_det,
        y_s=y_train_det,
        X_t=X_target,
        epochs=args.coral_epochs,
        batch_size=args.uda_batch_size,
        lr=args.learning_rate,
        coral_weight=args.coral_weight,
        device=args.device,
    )

    # D. DANN UDA Model
    dann_model = train_dann(
        X_s=X_train_det,
        y_s=y_train_det,
        X_t=X_target,
        epochs=args.dann_epochs,
        batch_size=args.uda_batch_size,
        lr=args.learning_rate,
        dann_weight=args.dann_weight,
        device=args.device,
    )

    # 6. Save Artifacts
    print("\nSaving Downstream Models...")
    # Save standard bundles via joblib
    joblib.dump({"detection": det_bundle, "sizing": size_bundle}, os.path.join(downstream_dir, "standard_probes.joblib"))

    # Save PyTorch UDA models
    torch.save(coral_model.state_dict(), os.path.join(downstream_dir, "coral_model.pt"))
    torch.save(dann_model.state_dict(), os.path.join(downstream_dir, "dann_model.pt"))

    summary = {
        "checkpoint": ckpt_path,
        "train_samples_det": int(len(y_train_det)),
        "train_samples_size": int(len(y_train_size)) if y_train_size is not None else 0,
        "target_samples_uda": int(len(X_target)),
        "coral_epochs": args.coral_epochs,
        "coral_weight": args.coral_weight,
        "dann_epochs": args.dann_epochs,
        "dann_weight": args.dann_weight,
        "device": args.device,
    }
    with open(os.path.join(downstream_dir, "downstream_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"SUCCESS: All Downstream Models Trained & Saved to:")
    print(f"  - {os.path.join(downstream_dir, 'standard_probes.joblib')}")
    print(f"  - {os.path.join(downstream_dir, 'coral_model.pt')}")
    print(f"  - {os.path.join(downstream_dir, 'dann_model.pt')}")
    print(f"  - {os.path.join(downstream_dir, 'downstream_train_summary.json')}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
