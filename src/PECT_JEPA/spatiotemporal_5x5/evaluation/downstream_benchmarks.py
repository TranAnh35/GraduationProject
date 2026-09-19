"""
Comprehensive Downstream Benchmark Suite for 5x5 PECT-JEPA.

Implements and benchmarks:
1. Linear Probe (LogisticRegression / Ridge)
2. MLP 2-Layer Probe (Non-linear 2-layer neural network: Linear -> LayerNorm/ReLU -> Linear)

Across 4 Key NDT Tasks:
- Task 1: Binary Defect Detection (AUC-ROC, Average Precision, F1, Accuracy)
- Task 2: 4-Class Semantic Segmentation (Sound, Free Corrosion, Sound Rivet, Rivet+Corrosion)
- Task 3: Defect Severity / Depth Binning (Sound, Shallow <=0.2mm, Medium 0.3-0.6mm, Deep >=0.7mm)
- Task 4: Quantitative Defect Depth Regression (Predict physical depth d in mm -> R^2 score, MAE, RMSE)

Computes the Representation Gap (MLP - Linear) to rigorously audit:
- Is the latent space already linearly separable?
- Does a non-linear readout unlock higher fidelity for subtle/sub-surface rivet defects?
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DownstreamBenchmarkSuite:
    """
    Evaluates both Linear Probe and MLP 2-Layer Probe on frozen PECT-JEPA representations.
    Supports balanced class weighting to eliminate prior bias and ensure the 0.5 decision threshold
    strictly aligns with the geometric separating hyperplane w^T z = 0.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        mlp_hidden_dim: int = 64,
        max_iter: int = 150,
        class_weight: str = "balanced",
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.mlp_hidden_dim = mlp_hidden_dim
        self.max_iter = max_iter
        self.class_weight = class_weight

    # =========================================================================
    # Task 1: Binary Defect Detection
    # =========================================================================
    def benchmark_binary_detection(
        self,
        features: np.ndarray,  # [N, D] or [sY, sX, D]
        labels: np.ndarray,    # [N] or [sY, sX] (1 = defect, 0 = sound, -1 = buffer ignore)
    ) -> Dict[str, Any]:
        """
        Benchmarks Linear vs MLP 2-Layer on binary defect detection.
        """
        flat_feats = features.reshape(-1, features.shape[-1])
        flat_labels = labels.reshape(-1)
        valid_idx = np.where(flat_labels >= 0)[0]
        X = flat_feats[valid_idx].astype(np.float32)
        y = flat_labels[valid_idx].astype(np.int64)

        n_def = int(np.sum(y == 1))
        if n_def < self.n_splits:
            return {"error": f"Too few defects ({n_def}) for {self.n_splits}-fold CV"}

        # Smart background subsampling: preserve 100% of defect samples, cap sound metal background at 8000
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(neg_idx) > 8000:
            rng = np.random.RandomState(self.random_state)
            sub_neg = rng.choice(neg_idx, size=min(len(neg_idx), max(len(pos_idx) * 10, 8000)), replace=False)
            keep_idx = np.sort(np.concatenate([pos_idx, sub_neg]))
            X = X[keep_idx]
            y = y[keep_idx]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        linear_auc, linear_ap, linear_f1, linear_acc = [], [], [], []
        mlp_auc, mlp_ap, mlp_f1, mlp_acc = [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Feature Standardization (strictly fit on train fold)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. Linear Probe (Logistic Regression) with Balanced Weighting
            lr = LogisticRegression(
                C=1.0,
                max_iter=max(500, self.max_iter),
                tol=1e-3,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            )
            lr.fit(X_tr_s, y_tr)
            p_lr = lr.predict_proba(X_te_s)[:, 1]
            y_pred_lr = (p_lr >= 0.5).astype(int)

            linear_auc.append(float(roc_auc_score(y_te, p_lr)))
            linear_ap.append(float(average_precision_score(y_te, p_lr)))
            linear_f1.append(float(f1_score(y_te, y_pred_lr, zero_division=0)))
            linear_acc.append(float(accuracy_score(y_te, y_pred_lr)))

            # 2. MLP 2-Layer Probe with Balanced Sample Weights
            mlp = MLPClassifier(
                hidden_layer_sizes=(self.mlp_hidden_dim,),
                activation="relu",
                max_iter=self.max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=self.random_state,
            )
            sample_weights = compute_sample_weight("balanced", y_tr) if self.class_weight == "balanced" else None
            mlp.fit(X_tr_s, y_tr, sample_weight=sample_weights)
            p_mlp = mlp.predict_proba(X_te_s)[:, 1]
            y_pred_mlp = (p_mlp >= 0.5).astype(int)

            mlp_auc.append(float(roc_auc_score(y_te, p_mlp)))
            mlp_ap.append(float(average_precision_score(y_te, p_mlp)))
            mlp_f1.append(float(f1_score(y_te, y_pred_mlp, zero_division=0)))
            mlp_acc.append(float(accuracy_score(y_te, y_pred_mlp)))

        l_auc_m, l_ap_m, l_f1_m, l_acc_m = np.mean(linear_auc), np.mean(linear_ap), np.mean(linear_f1), np.mean(linear_acc)
        m_auc_m, m_ap_m, m_f1_m, m_acc_m = np.mean(mlp_auc), np.mean(mlp_ap), np.mean(mlp_f1), np.mean(mlp_acc)

        return {
            "linear_probe": {
                "auc_roc": round(float(l_auc_m), 4),
                "average_precision": round(float(l_ap_m), 4),
                "f1_score": round(float(l_f1_m), 4),
                "accuracy": round(float(l_acc_m), 4),
            },
            "mlp_2layer": {
                "auc_roc": round(float(m_auc_m), 4),
                "average_precision": round(float(m_ap_m), 4),
                "f1_score": round(float(m_f1_m), 4),
                "accuracy": round(float(m_acc_m), 4),
            },
            "representation_gap": {
                "delta_auc_roc": round(float(m_auc_m - l_auc_m), 4),
                "delta_average_precision": round(float(m_ap_m - l_ap_m), 4),
                "delta_f1": round(float(m_f1_m - l_f1_m), 4),
            },
        }

    # =========================================================================
    # Task 2: 4-Class Semantic Segmentation (Sound, Corrosion, Rivet, Rivet+Corrosion)
    # =========================================================================
    def benchmark_multiclass_semantic(
        self,
        features: np.ndarray,   # [N, D] or [sY, sX, D]
        labels: np.ndarray,     # [N] or [sY, sX] (0: sound, 1: free corrosion, 2: sound rivet, 3: rivet+corrosion, -1: buffer)
    ) -> Dict[str, Any]:
        flat_feats = features.reshape(-1, features.shape[-1])
        flat_labels = labels.reshape(-1)
        valid_idx = np.where(flat_labels >= 0)[0]
        X = flat_feats[valid_idx].astype(np.float32)
        y = flat_labels[valid_idx].astype(np.int64)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return {"error": f"Need at least 2 distinct classes, found {len(classes)}"}

        # Subsample dominant sound metal (class 0) to avoid excessive training latency on uniform background
        c0_idx = np.where(y == 0)[0]
        cother_idx = np.where(y > 0)[0]
        if len(c0_idx) > 8000:
            rng = np.random.RandomState(self.random_state)
            sub_c0 = rng.choice(c0_idx, size=8000, replace=False)
            keep_idx = np.sort(np.concatenate([cother_idx, sub_c0]))
            X = X[keep_idx]
            y = y[keep_idx]
            classes, counts = np.unique(y, return_counts=True)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        linear_acc, linear_mf1, mlp_acc, mlp_mf1 = [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Linear Probe (Multinomial Logistic Regression) with Balanced Weighting
            lr = LogisticRegression(
                C=1.0,
                max_iter=max(500, self.max_iter),
                tol=1e-3,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
            )
            lr.fit(X_tr_s, y_tr)
            y_pred_lr = lr.predict(X_te_s)
            linear_acc.append(float(accuracy_score(y_te, y_pred_lr)))
            linear_mf1.append(float(f1_score(y_te, y_pred_lr, average="macro", zero_division=0)))

            # MLP 2-Layer with Balanced Sample Weights
            mlp = MLPClassifier(
                hidden_layer_sizes=(self.mlp_hidden_dim,),
                activation="relu",
                max_iter=self.max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=self.random_state,
            )
            sample_weights = compute_sample_weight("balanced", y_tr) if self.class_weight == "balanced" else None
            mlp.fit(X_tr_s, y_tr, sample_weight=sample_weights)
            y_pred_mlp = mlp.predict(X_te_s)
            mlp_acc.append(float(accuracy_score(y_te, y_pred_mlp)))
            mlp_mf1.append(float(f1_score(y_te, y_pred_mlp, average="macro", zero_division=0)))

        l_acc_m, l_mf1_m = np.mean(linear_acc), np.mean(linear_mf1)
        m_acc_m, m_mf1_m = np.mean(mlp_acc), np.mean(mlp_mf1)

        return {
            "classes": [int(c) for c in classes],
            "class_counts": {int(c): int(cnt) for c, cnt in zip(classes, counts)},
            "linear_probe": {
                "accuracy": round(float(l_acc_m), 4),
                "macro_f1": round(float(l_mf1_m), 4),
            },
            "mlp_2layer": {
                "accuracy": round(float(m_acc_m), 4),
                "macro_f1": round(float(m_mf1_m), 4),
            },
            "representation_gap": {
                "delta_accuracy": round(float(m_acc_m - l_acc_m), 4),
                "delta_macro_f1": round(float(m_mf1_m - l_mf1_m), 4),
            },
        }

    # =========================================================================
    # Task 3: Defect Severity Classification (Sound, Shallow, Medium, Deep)
    # =========================================================================
    def benchmark_severity_classification(
        self,
        features: np.ndarray,
        severity_mask: np.ndarray,  # 0: sound, 1: shallow, 2: medium, 3: deep, -1: buffer
    ) -> Dict[str, Any]:
        return self.benchmark_multiclass_semantic(features, severity_mask)

    # =========================================================================
    # Task 4: Quantitative Defect Depth Sizing Regression (Predict mm)
    # =========================================================================
    def benchmark_depth_regression(
        self,
        features: np.ndarray,     # [N, D]
        depth_map: np.ndarray,    # [N] continuous float depth in mm
        focus_defects_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates quantitative depth sizing regression.
        If focus_defects_only is True, evaluates exclusively on defective pixels (depth > 0).
        """
        flat_feats = features.reshape(-1, features.shape[-1])
        flat_depth = depth_map.reshape(-1)

        if focus_defects_only:
            valid_idx = np.where(flat_depth > 0.0)[0]
        else:
            valid_idx = np.arange(len(flat_depth))

        X = flat_feats[valid_idx].astype(np.float32)
        y = flat_depth[valid_idx].astype(np.float32)

        if len(y) < self.n_splits:
            return {"error": "Too few samples for regression"}

        if not focus_defects_only:
            def_idx = np.where(y > 0.0)[0]
            snd_idx = np.where(y == 0.0)[0]
            if len(snd_idx) > 8000:
                rng = np.random.RandomState(self.random_state)
                sub_snd = rng.choice(snd_idx, size=8000, replace=False)
                keep_idx = np.sort(np.concatenate([def_idx, sub_snd]))
                X = X[keep_idx]
                y = y[keep_idx]

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        linear_r2, linear_mae, linear_rmse = [], [], []
        mlp_r2, mlp_mae, mlp_rmse = [], [], []

        for train_idx, test_idx in kf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. Linear Ridge Regressor
            ridge = Ridge(alpha=1.0, random_state=self.random_state)
            ridge.fit(X_tr_s, y_tr)
            y_pred_lr = ridge.predict(X_te_s)

            linear_r2.append(float(r2_score(y_te, y_pred_lr)))
            linear_mae.append(float(mean_absolute_error(y_te, y_pred_lr)))
            linear_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred_lr))))

            # 2. MLP 2-Layer Regressor
            mlp = MLPRegressor(
                hidden_layer_sizes=(self.mlp_hidden_dim,),
                activation="relu",
                max_iter=self.max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=self.random_state,
            )
            mlp.fit(X_tr_s, y_tr)
            y_pred_mlp = mlp.predict(X_te_s)

            mlp_r2.append(float(r2_score(y_te, y_pred_mlp)))
            mlp_mae.append(float(mean_absolute_error(y_te, y_pred_mlp)))
            mlp_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred_mlp))))

        l_r2, l_mae, l_rmse = np.mean(linear_r2), np.mean(linear_mae), np.mean(linear_rmse)
        m_r2, m_mae, m_rmse = np.mean(mlp_r2), np.mean(mlp_mae), np.mean(mlp_rmse)

        return {
            "mode": "defects_only" if focus_defects_only else "full_plate",
            "samples": len(y),
            "linear_probe": {
                "r2_score": round(float(l_r2), 4),
                "mae_mm": round(float(l_mae), 4),
                "rmse_mm": round(float(l_rmse), 4),
            },
            "mlp_2layer": {
                "r2_score": round(float(m_r2), 4),
                "mae_mm": round(float(m_mae), 4),
                "rmse_mm": round(float(m_rmse), 4),
            },
            "representation_gap": {
                "delta_r2": round(float(m_r2 - l_r2), 4),
                "delta_mae_mm": round(float(m_mae - l_mae), 4),
            },
        }
