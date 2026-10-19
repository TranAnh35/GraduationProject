"""
Dedicated Training Pipeline for Downstream Tasks & Unsupervised Domain Adaptation (UDA).

Trains downstream models on top of frozen PECT-JEPA representations:
1. Standard Probes:
   - Detection: Linear Probe (Logistic Regression) & 2-Layer MLP
   - Sizing: Ridge Regressor & MLP Regressor
2. Unsupervised Domain Adaptation (UDA) Models:
   - CORAL: Second-order covariance matrix alignment
   - DANN: Domain-Adversarial Neural Network with Gradient Reversal Layer (GRL)
   - DAN: Deep Adaptation Network with Multi-Kernel Maximum Mean Discrepancy (MK-MMD)
   - CDAN: Conditional Domain-Adversarial Network with multilinear prediction conditioning
   - SHOT: Source Hypothesis Transfer with Information Maximization (Entropy + Diversity)

Outputs saved to: <exp_dir>/downstream_models/
  ├── standard_probes.joblib
  ├── coral_model.pt
  ├── dann_model.pt
  ├── dan_model.pt
  ├── cdan_model.pt
  ├── shot_model.pt
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
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.dan_mmd import DANModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.cdan import CDANModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.shot import SHOTModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.otda import OTDAModel, wasserstein_distance
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.cbst import CBSTModel
from src.PECT_JEPA.cross_condition.data.split import get_canonical_shift_files, is_base_domain_file

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
    
    # Domain Definition & Shift Mode
    parser.add_argument(
        "--adaptation_mode",
        "--condition_shift",
        dest="adaptation_mode",
        type=str,
        default="all",
        choices=["all", "cross_sensor", "cross_waveform", "cross_specimen", "cross_liftoff", "all_test"],
        help="Target domain selection: 'all' (all 4 canonical single-shifts), 'cross_sensor' (TMR), 'cross_waveform' (Chirp), 'cross_specimen' (Rivet_v2), 'cross_liftoff' (z2)",
    )
    
    # UDA Training Hyperparameters
    parser.add_argument("--uda_batch_size", type=int, default=256, help="Batch size for UDA neural training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for UDA models")
    parser.add_argument("--coral_weight", type=float, default=1.0, help="Weight lambda for CORAL loss")
    parser.add_argument("--dann_weight", type=float, default=0.1, help="Weight lambda for DANN discriminator loss")
    parser.add_argument("--dan_weight", type=float, default=0.5, help="Weight lambda for DAN MK-MMD loss")
    parser.add_argument("--cdan_weight", type=float, default=0.1, help="Weight lambda for CDAN discriminator loss")
    parser.add_argument("--shot_div_weight", type=float, default=2.0, help="Weight for SHOT diversity regularizer")
    parser.add_argument("--ot_weight", type=float, default=0.5, help="Weight lambda for Optimal Transport Wasserstein loss")
    parser.add_argument("--cbst_weight", type=float, default=0.5, help="Weight lambda for Class-Balanced Self-Training loss")
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=1e-3, help="Learning rate for UDA optimizers")
    parser.add_argument("--target_sample_size", type=int, default=50000, help="Number of unlabeled target pixels to subsample for UDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


# =========================================================================
# 1. CORAL Training
# =========================================================================
def train_coral(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    coral_weight: float = 1.0, device: str = "cpu",
) -> CORALAdapter:
    in_dim = X_s.shape[1]
    model = CORALAdapter(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())
    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [1/5 CORAL UDA] Training for {epochs} epochs...")
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
            z_s = model.adapt(batch_s)
            z_t = model.adapt(batch_t)
            logits_s = model.classifier(z_s).squeeze(-1)

            loss_task = criterion_task(logits_s, labels_s)
            loss_c = coral_loss(z_s, z_t)
            loss = loss_task + coral_weight * loss_c

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_coral += loss_c.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, CORAL={total_coral/n_b:.4f})")
    model.eval()
    return model


# =========================================================================
# 2. DANN Training
# =========================================================================
def train_dann(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    dann_weight: float = 0.1, device: str = "cpu",
) -> DANNModel:
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
    print(f"  [2/5 DANN UDA] Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss, total_task, total_domain = 0.0, 0.0, 0.0
        iter_t = iter(loader_t)
        p = float(epoch) / max(1, epochs)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        for batch_s, labels_s in loader_s:
            try:
                batch_t = next(iter_t)[0]
            except StopIteration:
                iter_t = iter(loader_t)
                batch_t = next(iter_t)[0]

            batch_s, labels_s = batch_s.to(device), labels_s.to(device)
            batch_t = batch_t.to(device)

            optimizer.zero_grad()
            class_logits_s, domain_logits_s = model(batch_s, alpha=alpha)
            loss_task = criterion_task(class_logits_s, labels_s)
            loss_domain_s = criterion_domain(domain_logits_s, torch.zeros_like(domain_logits_s))

            _, domain_logits_t = model(batch_t, alpha=alpha)
            loss_domain_t = criterion_domain(domain_logits_t, torch.ones_like(domain_logits_t))

            loss_domain = 0.5 * (loss_domain_s + loss_domain_t)
            loss = loss_task + dann_weight * loss_domain

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_domain += loss_domain.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, Domain={total_domain/n_b:.4f}, alpha={alpha:.2f})")
    model.eval()
    return model


# =========================================================================
# 3. DAN (MK-MMD) Training
# =========================================================================
def train_dan(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    dan_weight: float = 0.5, device: str = "cpu",
) -> DANModel:
    in_dim = X_s.shape[1]
    model = DANModel(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())
    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [3/5 DAN MK-MMD UDA] Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss, total_task, total_mmd = 0.0, 0.0, 0.0
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
            z_s = model.adapt(batch_s)
            z_t = model.adapt(batch_t)
            logits_s = model.classifier(z_s).squeeze(-1)

            loss_task = criterion_task(logits_s, labels_s)
            loss_mmd = model.compute_mmd_loss(z_s, z_t)
            loss = loss_task + dan_weight * loss_mmd

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_mmd += loss_mmd.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, MMD={total_mmd/n_b:.4f})")
    model.eval()
    return model


# =========================================================================
# 4. CDAN Training
# =========================================================================
def train_cdan(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    cdan_weight: float = 0.1, device: str = "cpu",
) -> CDANModel:
    in_dim = X_s.shape[1]
    model = CDANModel(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_domain = nn.BCEWithLogitsLoss(reduction="none")

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())
    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [4/5 CDAN UDA] Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss, total_task, total_domain = 0.0, 0.0, 0.0
        iter_t = iter(loader_t)
        p = float(epoch) / max(1, epochs)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        for batch_s, labels_s in loader_s:
            try:
                batch_t = next(iter_t)[0]
            except StopIteration:
                iter_t = iter(loader_t)
                batch_t = next(iter_t)[0]

            batch_s, labels_s = batch_s.to(device), labels_s.to(device)
            batch_t = batch_t.to(device)

            optimizer.zero_grad()
            class_logits_s, domain_logits_s, weights_s = model(batch_s, alpha=alpha)
            loss_task = criterion_task(class_logits_s, labels_s)
            loss_domain_s = torch.mean(weights_s * criterion_domain(domain_logits_s, torch.zeros_like(domain_logits_s)))

            _, domain_logits_t, weights_t = model(batch_t, alpha=alpha)
            loss_domain_t = torch.mean(weights_t * criterion_domain(domain_logits_t, torch.ones_like(domain_logits_t)))

            loss_domain = 0.5 * (loss_domain_s + loss_domain_t)
            loss = loss_task + cdan_weight * loss_domain

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_domain += loss_domain.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, Domain={total_domain/n_b:.4f})")
    model.eval()
    return model


# =========================================================================
# 5. SHOT (Information Maximization) Training
# =========================================================================
def train_shot(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    div_weight: float = 2.0, device: str = "cpu",
) -> SHOTModel:
    in_dim = X_s.shape[1]
    model = SHOTModel(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())
    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [5/5 SHOT UDA] Training for {epochs} epochs (Source Task + Target InfoMax)...")
    for epoch in range(epochs):
        total_loss, total_task, total_im = 0.0, 0.0, 0.0
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
            logits_s = model(batch_s)
            loss_task = criterion_task(logits_s, labels_s)

            # Target Information Maximization (Entropy min + Class-Prior diversity)
            z_t = model.adapt(batch_t)
            loss_im = model.compute_im_loss(z_t, defect_prior=0.02)
            loss = loss_task + 0.5 * loss_im

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_im += loss_im.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, InfoMax={total_im/n_b:.4f})")
    model.eval()
    return model


# =========================================================================
# 6. OTDA Training (Optimal Transport / Wasserstein Sinkhorn)
# =========================================================================
def train_otda(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    ot_weight: float = 0.5, device: str = "cpu",
) -> OTDAModel:
    in_dim = X_s.shape[1]
    model = OTDAModel(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())
    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    print(f"  [6/7 OTDA UDA] Training for {epochs} epochs (Task + Wasserstein Sinkhorn Transport)...")
    for epoch in range(epochs):
        total_loss, total_task, total_ot = 0.0, 0.0, 0.0
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
            logits_s = model(batch_s)
            loss_task = criterion_task(logits_s, labels_s)

            feat_s = model.extract_features(batch_s)
            feat_t = model.extract_features(batch_t)
            loss_ot = wasserstein_distance(feat_s, feat_t, epsilon=0.1, max_iter=30)

            loss = loss_task + ot_weight * loss_ot
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_ot += loss_ot.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, OT={total_ot/n_b:.4f})")
    model.eval()
    return model


# =========================================================================
# 7. CBST Training (Class-Balanced Prototype Self-Training)
# =========================================================================
def train_cbst(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
    epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
    st_weight: float = 0.5, device: str = "cpu",
) -> CBSTModel:
    in_dim = X_s.shape[1]
    model = CBSTModel(in_dim=in_dim, hidden_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pos_count = np.sum(y_s == 1)
    neg_count = np.sum(y_s == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=device)
    criterion_task = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_s = TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float())
    ds_t = TensorDataset(torch.from_numpy(X_t).float())
    loader_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_t = DataLoader(ds_t, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initial prototype computation from source data
    with torch.no_grad():
        all_xs = torch.from_numpy(X_s[:min(10000, len(X_s))]).float().to(device)
        all_ys = torch.from_numpy(y_s[:min(10000, len(y_s))]).float().to(device)
        init_feats = model.extract_features(all_xs)
        model.update_prototypes(init_feats, all_ys, momentum=0.0)

    model.train()
    print(f"  [7/7 CBST UDA] Training for {epochs} epochs (Source Task + Balanced Pseudo-Labels)...")
    for epoch in range(epochs):
        total_loss, total_task, total_st = 0.0, 0.0, 0.0
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
            logits_s = model(batch_s)
            loss_task = criterion_task(logits_s, labels_s)

            # Update running prototypes from current batch
            feat_s = model.extract_features(batch_s)
            model.update_prototypes(feat_s.detach(), labels_s, momentum=0.9)

            # Generate class-balanced pseudo-labels for target batch
            feat_t = model.extract_features(batch_t)
            sel_feat_t, pseudo_yt = model.generate_balanced_pseudo_labels(
                feat_t,
                top_k_defect=max(4, batch_size // 16),
                top_k_sound=max(16, batch_size // 4),
            )

            if len(pseudo_yt) > 0:
                logits_pseudo = model.classifier(sel_feat_t).squeeze(-1)
                loss_st = F.binary_cross_entropy_with_logits(logits_pseudo, pseudo_yt)
            else:
                loss_st = torch.tensor(0.0, device=device)

            loss = loss_task + st_weight * loss_st
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task += loss_task.item()
            total_st += loss_st.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            n_b = len(loader_s)
            print(f"    Epoch {epoch+1:2d}/{epochs:2d}: Loss={total_loss/n_b:.4f} (Task={total_task/n_b:.4f}, PseudoST={total_st/n_b:.4f})")
    model.eval()
    return model


# =========================================================================
# Main Pipeline
# =========================================================================
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

    canonical_shifts = ["cross_sensor", "cross_waveform", "cross_specimen", "cross_liftoff"]
    if args.adaptation_mode in ["all", "all_test"]:
        active_shifts = canonical_shifts
    else:
        active_shifts = [args.adaptation_mode]

    print("=" * 80)
    print(f"CANONICAL DOWNSTREAM & UDA TRAINING PIPELINE")
    print(f"  Checkpoint:       {ckpt_path}")
    print(f"  Output Dir:       {downstream_dir}")
    print(f"  Adaptation Mode:  {args.adaptation_mode} -> {active_shifts}")
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

    # 3. Extract Source Domain Data ONCE (16 Base Domain files: train + val)
    base_source_files = [fp for fp in all_files if is_base_domain_file(extract_file_metadata(fp))]
    print(f"\n[STEP 1/3] Extracting Source Domain Pool (16 Base Domain files: Train + Val)...")
    source_eval_files = [fp for fp in base_source_files if extract_file_metadata(fp)["specimen"] in gt_masks]
    train_det_x, train_det_y = [], []
    train_size_x, train_size_y = [], []
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

    # Fit Standard Probes ONCE on Source
    print("\n[STEP 2/3] Fitting & Saving Standard Probes (Linear/MLP Detection & Sizing)...")
    scaler_det = StandardScaler()
    X_train_det_s = scaler_det.fit_transform(X_train_det)
    lr_probe = LogisticRegression(C=1.0, max_iter=500, tol=1e-3, class_weight="balanced",
                                  random_state=args.seed, solver="lbfgs")
    lr_probe.fit(X_train_det_s, y_train_det)
    mlp_probe = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                              random_state=args.seed, early_stopping=True, n_iter_no_change=10)
    mlp_probe.fit(X_train_det_s, y_train_det)
    det_bundle = DetectionProbeBundle(scaler=scaler_det, linear_probe=lr_probe, mlp_probe=mlp_probe)

    size_bundle = None
    if X_train_size is not None and y_train_size is not None:
        scaler_size = StandardScaler()
        X_train_size_s = scaler_size.fit_transform(X_train_size)
        ridge_reg = Ridge(alpha=10.0, random_state=args.seed)
        ridge_reg.fit(X_train_size_s, y_train_size)
        mlp_reg = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", max_iter=150,
                               random_state=args.seed, early_stopping=True, n_iter_no_change=10)
        mlp_reg.fit(X_train_size_s, y_train_size)
        size_bundle = SizingRegressorBundle(scaler=scaler_size, ridge_regressor=ridge_reg, mlp_regressor=mlp_reg)

    joblib.dump({"detection": det_bundle, "sizing": size_bundle}, os.path.join(downstream_dir, "standard_probes.joblib"))
    print(f"  Saved standard_probes.joblib to {downstream_dir}")

    # 4. Train 7 UDA Models for Each Canonical Shift
    print(f"\n[STEP 3/3] Training 7 UDA Models across {len(active_shifts)} Canonical Shifts...")

    for shift in active_shifts:
        src_files, tgt_files = get_canonical_shift_files(all_files, shift)
        shift_dir = os.path.join(downstream_dir, shift)
        os.makedirs(shift_dir, exist_ok=True)

        print("\n" + "=" * 75)
        print(f"  >>> TRAINING UDA FOR CANONICAL SHIFT: [{shift.upper()}] <<<")
        print(f"  Source: {len(src_files)} Base files | Target: {len(tgt_files)} Single-Shift files")
        print(f"  Target Destination: {shift_dir}")
        print("=" * 75, flush=True)

        # Extract target features for this specific shift
        target_z_list = []
        pbar_target = tqdm(_iter_with_prefetch(tgt_files, args.prefetch), total=len(tgt_files),
                           desc=f"Extracting Target [{shift}]", file=sys.stdout, dynamic_ncols=True)
        for fp, cscan in pbar_target:
            latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
            flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            sample_n = min(len(flat_z), max(2000, args.target_sample_size // len(tgt_files)))
            idx = rng.choice(len(flat_z), size=sample_n, replace=False)
            target_z_list.append(flat_z[idx])

        X_target = np.concatenate(target_z_list, axis=0)
        if len(X_target) > args.target_sample_size:
            X_target = X_target[rng.choice(len(X_target), size=args.target_sample_size, replace=False)]
        print(f"  Target Samples for [{shift}]: {len(X_target)} pixels")

        # Train Model 1: CORAL
        coral_model = train_coral(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            coral_weight=args.coral_weight, device=args.device,
        )

        # Train Model 2: DANN
        dann_model = train_dann(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            dann_weight=args.dann_weight, device=args.device,
        )

        # Train Model 3: DAN (MK-MMD)
        dan_model = train_dan(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            dan_weight=args.dan_weight, device=args.device,
        )

        # Train Model 4: CDAN
        cdan_model = train_cdan(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            cdan_weight=args.cdan_weight, device=args.device,
        )

        # Train Model 5: SHOT
        shot_model = train_shot(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            div_weight=args.shot_div_weight, device=args.device,
        )

        # Train Model 6: OTDA (Wasserstein Sinkhorn)
        otda_model = train_otda(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            ot_weight=args.ot_weight, device=args.device,
        )

        # Train Model 7: CBST (Class-Balanced Self-Training)
        cbst_model = train_cbst(
            X_s=X_train_det, y_s=y_train_det, X_t=X_target,
            epochs=args.epochs, batch_size=args.uda_batch_size, lr=args.learning_rate,
            st_weight=args.cbst_weight, device=args.device,
        )

        # Save all 7 models into shift directory
        torch.save(coral_model.state_dict(), os.path.join(shift_dir, "coral_model.pt"))
        torch.save(dann_model.state_dict(), os.path.join(shift_dir, "dann_model.pt"))
        torch.save(dan_model.state_dict(), os.path.join(shift_dir, "dan_model.pt"))
        torch.save(cdan_model.state_dict(), os.path.join(shift_dir, "cdan_model.pt"))
        torch.save(shot_model.state_dict(), os.path.join(shift_dir, "shot_model.pt"))
        torch.save(otda_model.state_dict(), os.path.join(shift_dir, "otda_model.pt"))
        torch.save(cbst_model.state_dict(), os.path.join(shift_dir, "cbst_model.pt"))

        summary = {
            "checkpoint": ckpt_path,
            "shift": shift,
            "source_files": len(src_files),
            "target_files": len(tgt_files),
            "source_samples": int(len(y_train_det)),
            "target_samples": int(len(X_target)),
            "epochs": args.epochs,
            "models": ["CORAL", "DANN", "DAN", "CDAN", "SHOT", "OTDA", "CBST"],
            "device": args.device,
        }
        with open(os.path.join(shift_dir, "downstream_train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"  [OK] Saved all 7 UDA models for [{shift}] to: {shift_dir}")

    print("\n" + "=" * 80)
    print(f"SUCCESS: All Downstream Models Trained:")
    print(f"  - Probes: {os.path.join(downstream_dir, 'standard_probes.joblib')}")
    for shift in active_shifts:
        print(f"  - UDA Suite ({shift}): {os.path.join(downstream_dir, shift)} (7 models: CORAL, DANN, DAN, CDAN, SHOT, OTDA, CBST)")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
