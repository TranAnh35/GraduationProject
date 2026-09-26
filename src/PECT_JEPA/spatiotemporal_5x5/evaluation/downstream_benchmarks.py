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

    # =========================================================================
    # Task 4b: Two-Stage Hurdle Protocol for Zero-Inflated Depth Regression
    # =========================================================================
    def benchmark_hurdle_depth_regression(
        self,
        features: np.ndarray,      # [N, D]
        depth_map: np.ndarray,     # [N] continuous float depth in mm
    ) -> Dict[str, Any]:
        """
        Two-Stage Hurdle Protocol (Rule 3 in GEMINI.md):
        Stage 1: Flaw Gate Classifier (P(y > 0 | z)) identifies defective metal vs sound metal.
        Stage 2: Conditional Sizing Regressor (E[y | y > 0, z]) regresses true flaw depth.
        Compound: y_hurdle = I(p > tau) * d_pred.
        Eliminates the zero-inflation noise penalty across 80,000 sound pixels.
        """
        flat_feats = features.reshape(-1, features.shape[-1])
        flat_depth = depth_map.reshape(-1)

        valid_idx = np.where(flat_depth >= 0.0)[0]
        X = flat_feats[valid_idx].astype(np.float32)
        y = flat_depth[valid_idx].astype(np.float32)
        y_bin = (y > 0.0).astype(np.int64)

        if np.sum(y_bin) < self.n_splits * 2:
            return {"error": "Too few defect samples for hurdle regression"}

        # Subsample sound background to prevent quadratic latency in CV
        def_idx = np.where(y_bin == 1)[0]
        snd_idx = np.where(y_bin == 0)[0]
        if len(snd_idx) > 8000:
            rng = np.random.RandomState(self.random_state)
            sub_snd = rng.choice(snd_idx, size=8000, replace=False)
            keep_idx = np.sort(np.concatenate([def_idx, sub_snd]))
            X = X[keep_idx]
            y = y[keep_idx]
            y_bin = y_bin[keep_idx]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        gate_auc, gate_ap = [], []
        standard_r2, standard_mae = [], []
        hurdle_r2, hurdle_mae = [], []
        defect_r2, defect_mae = [], []

        for train_idx, test_idx in skf.split(X, y_bin):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            y_bin_tr, y_bin_te = y_bin[train_idx], y_bin[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Stage 1: Balanced Gate Classifier
            clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=self.random_state)
            clf.fit(X_tr_s, y_bin_tr)
            p_te = clf.predict_proba(X_te_s)[:, 1]
            gate_auc.append(float(roc_auc_score(y_bin_te, p_te)))
            gate_ap.append(float(average_precision_score(y_bin_te, p_te)))

            # Stage 2: Conditional Sizing Regressor (fitted on y > 0)
            def_mask_tr = (y_tr > 0.0)
            ridge_def = Ridge(alpha=1.0, random_state=self.random_state)
            ridge_def.fit(X_tr_s[def_mask_tr], y_tr[def_mask_tr])
            d_pred_te = np.maximum(0.0, ridge_def.predict(X_te_s))

            # Baseline Standard Regressor (fitted on all y)
            ridge_all = Ridge(alpha=1.0, random_state=self.random_state)
            ridge_all.fit(X_tr_s, y_tr)
            y_pred_all = ridge_all.predict(X_te_s)
            standard_r2.append(float(r2_score(y_te, y_pred_all)))
            standard_mae.append(float(mean_absolute_error(y_te, y_pred_all)))

            # Compound Hurdle Prediction (tau = 0.5)
            y_hurdle_te = np.where(p_te >= 0.5, d_pred_te, 0.0)
            hurdle_r2.append(float(r2_score(y_te, y_hurdle_te)))
            hurdle_mae.append(float(mean_absolute_error(y_te, y_hurdle_te)))

            # Defect-only sizing on true flaws
            def_mask_te = (y_te > 0.0)
            if np.sum(def_mask_te) >= 2:
                defect_r2.append(float(r2_score(y_te[def_mask_te], d_pred_te[def_mask_te])))
                defect_mae.append(float(mean_absolute_error(y_te[def_mask_te], d_pred_te[def_mask_te])))

        return {
            "samples": len(y),
            "defect_samples": int(np.sum(y_bin)),
            "gate": {
                "auc_roc": round(float(np.mean(gate_auc)), 4),
                "average_precision": round(float(np.mean(gate_ap)), 4),
            },
            "conditional_defect_sizing": {
                "r2_score": round(float(np.mean(defect_r2)), 4) if defect_r2 else 0.0,
                "mae_mm": round(float(np.mean(defect_mae)), 4) if defect_mae else 0.0,
            },
            "compound_hurdle": {
                "plate_r2_score": round(float(np.mean(hurdle_r2)), 4),
                "plate_mae_mm": round(float(np.mean(hurdle_mae)), 4),
            },
            "standard_unclamped": {
                "plate_r2_score": round(float(np.mean(standard_r2)), 4),
                "plate_mae_mm": round(float(np.mean(standard_mae)), 4),
            },
        }

    # =========================================================================
    # Task 1b: True Cross-File Out-of-Distribution (OOD) Transfer Benchmark
    # =========================================================================
    def benchmark_cross_file_ood(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        True Out-of-Distribution (OOD) Transfer Benchmark.
        Trains Linear Probe and MLP on frozen features from training files,
        and evaluates strictly zero-shot on held-out test files with zero spatial leakage.
        """
    def fit_binary_detector(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Any, Any]:
        """Fits StandardScaler, Linear Probe, and MLP 2-Layer on training pool."""
        X_tr = X_train.reshape(-1, X_train.shape[-1]).astype(np.float32)
        y_tr = y_train.reshape(-1).astype(np.int64)
        v_tr = np.where(y_tr >= 0)[0]
        X_tr, y_tr = X_tr[v_tr], y_tr[v_tr]

        if len(np.unique(y_tr)) < 2:
            raise ValueError("Training set needs at least 2 distinct classes")

        pos_tr = np.where(y_tr == 1)[0]
        neg_tr = np.where(y_tr == 0)[0]
        if len(neg_tr) > 10000:
            rng = np.random.RandomState(self.random_state)
            sub_neg = rng.choice(neg_tr, size=min(len(neg_tr), max(len(pos_tr) * 10, 10000)), replace=False)
            keep_idx = np.sort(np.concatenate([pos_tr, sub_neg]))
            X_tr, y_tr = X_tr[keep_idx], y_tr[keep_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        lr = LogisticRegression(
            C=1.0,
            max_iter=max(500, self.max_iter),
            tol=1e-3,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        lr.fit(X_tr_s, y_tr)

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
        return scaler, lr, mlp

    def eval_binary_detector(
        self,
        scaler: Any,
        lr: Any,
        mlp: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluates pre-fitted probes zero-shot on test data."""
        X_te = X_test.reshape(-1, X_test.shape[-1]).astype(np.float32)
        y_te = y_test.reshape(-1).astype(np.int64)
        v_te = np.where(y_te >= 0)[0]
        X_te, y_te = X_te[v_te], y_te[v_te]

        if len(np.unique(y_te)) < 2:
            return {"error": "Test set needs at least 2 distinct classes for AUC"}

        X_te_s = scaler.transform(X_te)
        p_lr = lr.predict_proba(X_te_s)[:, 1]
        y_pred_lr = (p_lr >= 0.5).astype(int)

        l_auc = float(roc_auc_score(y_te, p_lr))
        l_ap = float(average_precision_score(y_te, p_lr))
        l_f1 = float(f1_score(y_te, y_pred_lr, zero_division=0))
        l_acc = float(accuracy_score(y_te, y_pred_lr))

        p_mlp = mlp.predict_proba(X_te_s)[:, 1]
        y_pred_mlp = (p_mlp >= 0.5).astype(int)

        m_auc = float(roc_auc_score(y_te, p_mlp))
        m_ap = float(average_precision_score(y_te, p_mlp))
        m_f1 = float(f1_score(y_te, y_pred_mlp, zero_division=0))
        m_acc = float(accuracy_score(y_te, y_pred_mlp))

        return {
            "linear_probe": {
                "auc_roc": round(l_auc, 4),
                "average_precision": round(l_ap, 4),
                "f1_score": round(l_f1, 4),
                "accuracy": round(l_acc, 4),
            },
            "mlp_2layer": {
                "auc_roc": round(m_auc, 4),
                "average_precision": round(m_ap, 4),
                "f1_score": round(m_f1, 4),
                "accuracy": round(m_acc, 4),
            },
            "representation_gap": {
                "delta_auc_roc": round(m_auc - l_auc, 4),
                "delta_average_precision": round(m_ap - l_ap, 4),
                "delta_f1": round(m_f1 - l_f1, 4),
            },
        }

    def benchmark_cross_file_ood(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Fits probe on training file(s) and evaluates zero-shot on an unseen test file.
        """
        try:
            scaler, lr, mlp = self.fit_binary_detector(X_train, y_train)
        except ValueError as e:
            return {"error": str(e)}
        return self.eval_binary_detector(scaler, lr, mlp, X_test, y_test)

    # =========================================================================
    # Task 1c: Single C-Scan Spatial-Block Cross-Validation (Zero Patch Overlap)
    # =========================================================================
    def benchmark_binary_detection_spatial_block(
        self,
        features_2d: np.ndarray,  # [H, W, D]
        labels_2d: np.ndarray,    # [H, W]
        buffer_margin: int = 5,
    ) -> Dict[str, Any]:
        """
        Spatial-Block Evaluation for a single C-scan.
        Divides the scan spatially (top half train, bottom half test)
        with an excluded buffer margin >= 5 pixels between them.
        Eliminates the 5x5 patch overlap spatial leakage inherent in random pixel splitting.
        """
        H, W, D = features_2d.shape
        mid_y = H // 2

        train_y_max = max(0, mid_y - buffer_margin)
        test_y_min = min(H, mid_y + buffer_margin)

        X_tr = features_2d[:train_y_max, :, :].reshape(-1, D)
        y_tr = labels_2d[:train_y_max, :].reshape(-1)

        X_te = features_2d[test_y_min:, :, :].reshape(-1, D)
        y_te = labels_2d[test_y_min:, :].reshape(-1)

        pos_tr = np.sum(y_tr == 1)
        pos_te = np.sum(y_te == 1)

        if pos_tr < 5 or pos_te < 5:
            mid_x = W // 2
            train_x_max = max(0, mid_x - buffer_margin)
            test_x_min = min(W, mid_x + buffer_margin)

            X_tr = features_2d[:, :train_x_max, :].reshape(-1, D)
            y_tr = labels_2d[:, :train_x_max].reshape(-1)

            X_te = features_2d[:, test_x_min:, :].reshape(-1, D)
            y_te = labels_2d[:, test_x_min:].reshape(-1)

        return self.benchmark_cross_file_ood(X_tr, y_tr, X_te, y_te)
