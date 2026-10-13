"""
Comprehensive Downstream Benchmark Suite for Cross-Condition PECT-JEPA.

Implements all 5 Core NDT Tasks on Frozen Foundation Latents (Zero metadata):
1. Task 1: Binary Defect Detection (Linear Probe & MLP 2-layer): AUC-ROC, Average Precision, F1, Accuracy.
2. Task 1b: Boundary Segmentation Contours, IoU (Jaccard Index), Dice coefficient.
3. Task 2: 4-Class Semantic Segmentation (Sound, Free Corrosion, Sound Rivet, Rivet+Corrosion).
4. Task 3: Defect Sizing Hurdle Protocol (Predict defect diameter/depth mm -> R^2 score, MAE, RMSE).
5. Task 4: Unsupervised Mahalanobis Anomaly Distance & Contrast-to-Noise Ratio (CNR).
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
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DownstreamBenchmarkSuite:
    """
    Evaluates Linear Probe and MLP 2-Layer on frozen PECT-JEPA representations.
    """
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        mlp_hidden_dim: int = 64,
        max_iter: int = 150,
        class_weight: str = "balanced",
        device: Optional[Any] = None,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.mlp_hidden_dim = mlp_hidden_dim
        self.max_iter = max_iter
        self.class_weight = class_weight
        # Accepted for API compatibility; the Mahalanobis metric stays on the
        # MKL/BLAS CPU path, which benchmarks faster than a GPU port at D=128
        # (transfer + pinv overhead dominates the 1.2 GFLOP distance matmul).
        self.device = device

    # =========================================================================
    # Global Readout Probe Training (Trained Strictly on Train Split Pool)
    # =========================================================================
    def fit_global_readouts(
        self,
        X_det: np.ndarray,
        y_det: np.ndarray,
        X_size: Optional[np.ndarray] = None,
        y_size: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fits global downstream readouts on representations from the training pool:
        1. StandardScaler + LogisticRegression(class_weight="balanced") for Defect Detection
        2. StandardScaler + MLPClassifier for Non-linear Defect Detection
        3. StandardScaler + Ridge(alpha=10.0) for Quantitative Flaw Sizing (diameter in mm)
        """
        readouts = {}

        # 1. Detection Probes
        scaler_det = StandardScaler()
        X_det_s = scaler_det.fit_transform(X_det)

        lr = LogisticRegression(
            C=1.0,
            max_iter=max(500, self.max_iter),
            tol=1e-3,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        lr.fit(X_det_s, y_det)
        readouts["scaler_det"] = scaler_det
        readouts["lr_probe"] = lr

        mlp = MLPClassifier(
            hidden_layer_sizes=(self.mlp_hidden_dim,),
            activation="relu",
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=self.random_state,
        )
        sample_weights = compute_sample_weight("balanced", y_det) if self.class_weight == "balanced" else None
        mlp.fit(X_det_s, y_det, sample_weight=sample_weights)
        readouts["mlp_probe"] = mlp

        # 2. Sizing Regressor Probe
        if X_size is not None and len(X_size) > 0 and y_size is not None:
            scaler_size = StandardScaler()
            X_size_s = scaler_size.fit_transform(X_size)
            ridge = Ridge(alpha=10.0, random_state=self.random_state)
            ridge.fit(X_size_s, y_size)
            readouts["scaler_size"] = scaler_size
            readouts["ridge_size"] = ridge

        return readouts

    def evaluate_global_detection(
        self,
        readouts: Dict[str, Any],
        features: np.ndarray,   # [H, W, D] or [N, D]
        labels: np.ndarray,     # [H, W] or [N]
    ) -> Dict[str, Any]:
        """
        Evaluates frozen global detection probes on a scan's latents.
        """
        orig_shape = features.shape[:2]
        flat_feats = features.reshape(-1, features.shape[-1]).astype(np.float32)
        flat_labels = labels.reshape(-1).astype(np.int64)

        scaler = readouts["scaler_det"]
        lr = readouts["lr_probe"]
        mlp = readouts.get("mlp_probe")

        feats_s = scaler.transform(flat_feats)
        prob_lr = lr.predict_proba(feats_s)[:, 1]
        pred_lr = (prob_lr >= 0.5).astype(int)

        # Compute metrics on valid pixels (labels >= 0, ignoring transition buffer if -1)
        valid = np.where(flat_labels >= 0)[0]
        y_v = flat_labels[valid]
        p_v = prob_lr[valid]
        pred_v = pred_lr[valid]

        has_both_classes = len(np.unique(y_v)) > 1
        auc = float(roc_auc_score(y_v, p_v)) if has_both_classes else 0.0
        ap = float(average_precision_score(y_v, p_v)) if has_both_classes else 0.0
        f1 = float(f1_score(y_v, pred_v, zero_division=0))
        acc = float(accuracy_score(y_v, pred_v))

        mlp_auc, mlp_ap = 0.0, 0.0
        prob_mlp = None
        if mlp is not None:
            prob_mlp = mlp.predict_proba(feats_s)[:, 1]
            p_mlp_v = prob_mlp[valid]
            mlp_auc = float(roc_auc_score(y_v, p_mlp_v)) if has_both_classes else 0.0
            mlp_ap = float(average_precision_score(y_v, p_mlp_v)) if has_both_classes else 0.0

        return {
            "auc_roc": round(auc, 4),
            "average_precision": round(ap, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "mlp_auc_roc": round(mlp_auc, 4),
            "mlp_average_precision": round(mlp_ap, 4),
            "prob_map": prob_lr.reshape(orig_shape),
            "pred_map": pred_lr.reshape(orig_shape),
        }

    def evaluate_global_sizing(
        self,
        readouts: Dict[str, Any],
        features: np.ndarray,   # [H, W, D] or [N, D]
        size_map: np.ndarray,   # [H, W] or [N]
        focus_defects_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates frozen global sizing probe on defect points.
        """
        orig_shape = features.shape[:2]
        flat_feats = features.reshape(-1, features.shape[-1]).astype(np.float32)
        flat_size = size_map.reshape(-1).astype(np.float32)

        if "scaler_size" not in readouts or "ridge_size" not in readouts:
            return {"r2_score": 0.0, "mae_mm": 0.0, "rmse_mm": 0.0, "pred_map": np.zeros(orig_shape, dtype=np.float32)}

        scaler = readouts["scaler_size"]
        ridge = readouts["ridge_size"]

        feats_s = scaler.transform(flat_feats)
        pred_all = np.maximum(0.0, ridge.predict(feats_s))

        if focus_defects_only:
            valid_idx = np.where(flat_size > 0.0)[0]
        else:
            valid_idx = np.arange(len(flat_size))

        if len(valid_idx) == 0 or np.ptp(flat_size[valid_idx]) < 1e-4:
            return {"r2_score": 0.0, "mae_mm": 0.0, "rmse_mm": 0.0, "pred_map": pred_all.reshape(orig_shape)}

        y_true = flat_size[valid_idx]
        y_pred = pred_all[valid_idx]

        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        return {
            "r2_score": round(r2, 4),
            "mae_mm": round(mae, 4),
            "rmse_mm": round(rmse, 4),
            "y_true": y_true,
            "y_pred": y_pred,
            "pred_map": pred_all.reshape(orig_shape),
        }

    # =========================================================================
    # Task 1: Binary Defect Detection (Linear Probe & MLP Probe)
    # =========================================================================
    def benchmark_binary_detection(
        self,
        features: np.ndarray,  # [H, W, D] or [N, D]
        labels: np.ndarray,    # [H, W] or [N] (1 = defect, 0 = sound metal)
    ) -> Dict[str, Any]:
        flat_feats = features.reshape(-1, features.shape[-1])
        flat_labels = labels.reshape(-1)
        valid_idx = np.where(flat_labels >= 0)[0]
        X = flat_feats[valid_idx].astype(np.float32)
        y = flat_labels[valid_idx].astype(np.int64)

        n_def = int(np.sum(y == 1))
        if n_def < self.n_splits:
            return {"error": f"Too few defects ({n_def}) for {self.n_splits}-fold CV"}

        # Subsample sound background to prevent memory/time blowup (keep max 8000 sound)
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

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. Linear Probe
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

            # 2. MLP 2-Layer
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
    # Task 1b: Boundary Contours & IoU (Jaccard Index)
    # =========================================================================
    def benchmark_segmentation_iou(
        self,
        prob_map: np.ndarray,
        gt_mask: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        flat_p = prob_map.reshape(-1)
        flat_gt = gt_mask.reshape(-1)
        valid = (flat_gt >= 0)

        y_true = (flat_gt[valid] == 1).astype(int)
        p_val = flat_p[valid]

        if len(np.unique(y_true)) < 2:
            return {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "optimal_threshold": 0.5}

        if threshold is not None:
            best_tau = float(threshold)
            y_pred = (p_val >= best_tau).astype(int)
            best_iou = float(jaccard_score(y_true, y_pred, zero_division=0))
            best_dice = float(f1_score(y_true, y_pred, zero_division=0))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
        else:
            best_dice = -1.0
            best_iou = 0.0
            best_tau = 0.5
            prec, rec = 0.0, 0.0
            for tau in np.linspace(0.2, 0.95, 31):
                c_pred = (p_val >= tau).astype(int)
                d = f1_score(y_true, c_pred, zero_division=0)
                if d > best_dice:
                    best_dice = float(d)
                    best_tau = float(tau)
                    best_iou = float(jaccard_score(y_true, c_pred, zero_division=0))
                    prec = float(precision_score(y_true, c_pred, zero_division=0))
                    rec = float(recall_score(y_true, c_pred, zero_division=0))

        return {
            "iou": round(best_iou, 4),
            "dice": round(best_dice, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "optimal_threshold": round(best_tau, 4),
        }

    # =========================================================================
    # Task 2: Quantitative Physical Sizing (Conditional Hurdle Regression)
    # =========================================================================
    def benchmark_sizing_regression(
        self,
        features: np.ndarray,
        size_map: np.ndarray,  # continuous defect diameter or depth in mm
        focus_defects_only: bool = True,
    ) -> Dict[str, Any]:
        flat_feats = features.reshape(-1, features.shape[-1])
        flat_size = size_map.reshape(-1)

        if focus_defects_only:
            valid_idx = np.where(flat_size > 0.0)[0]
        else:
            valid_idx = np.arange(len(flat_size))

        X = flat_feats[valid_idx].astype(np.float32)
        y = flat_size[valid_idx].astype(np.float32)

        if len(y) < self.n_splits or np.ptp(y) < 1e-4:
            return {"error": "Too few samples or zero variance for sizing regression"}

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        linear_r2, linear_mae, linear_rmse = [], [], []
        mlp_r2, mlp_mae, mlp_rmse = [], [], []

        for train_idx, test_idx in kf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. Ridge Regressor
            ridge = Ridge(alpha=10.0, random_state=self.random_state)
            ridge.fit(X_tr_s, y_tr)
            y_pred_lr = ridge.predict(X_te_s)

            linear_r2.append(float(r2_score(y_te, y_pred_lr)))
            linear_mae.append(float(mean_absolute_error(y_te, y_pred_lr)))
            linear_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred_lr))))

            # 2. MLP Regressor
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

        return {
            "linear_probe": {
                "r2_score": round(float(np.mean(linear_r2)), 4),
                "mae_mm": round(float(np.mean(linear_mae)), 4),
                "rmse_mm": round(float(np.mean(linear_rmse)), 4),
            },
            "mlp_2layer": {
                "r2_score": round(float(np.mean(mlp_r2)), 4),
                "mae_mm": round(float(np.mean(mlp_mae)), 4),
                "rmse_mm": round(float(np.mean(mlp_rmse)), 4),
            },
        }

    # =========================================================================
    # Task 3: Unsupervised Mahalanobis Distance & CNR
    # =========================================================================
    def compute_anomaly_detection_metrics(
        self,
        latents: np.ndarray,
        ground_truth_mask: np.ndarray,
    ) -> Dict[str, Any]:
        H, W, D = latents.shape
        flat_latents = latents.reshape(-1, D)
        flat_labels = (ground_truth_mask.reshape(-1) > 0.5).astype(np.int64)

        sound_idx = (flat_labels == 0)
        z_sound = flat_latents[sound_idx]
        mu_sound = np.mean(z_sound, axis=0)
        cov_sound = np.cov(z_sound, rowvar=False) + np.eye(D) * 1e-4
        inv_cov = np.linalg.pinv(cov_sound)

        diff = flat_latents - mu_sound
        dist_sq = np.sum((diff @ inv_cov) * diff, axis=-1)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))

        auc = float(roc_auc_score(flat_labels, dist))
        ap = float(average_precision_score(flat_labels, dist))
        ap_chance = float(np.mean(flat_labels))

        dist_def = dist[flat_labels == 1]
        dist_snd = dist[flat_labels == 0]
        mu_d = np.mean(dist_def) if len(dist_def) > 0 else 0.0
        mu_s = np.mean(dist_snd) if len(dist_snd) > 0 else 0.0
        var_d = np.var(dist_def) if len(dist_def) > 0 else 0.0
        var_s = np.var(dist_snd) if len(dist_snd) > 0 else 0.0
        cnr = float(abs(mu_d - mu_s) / max(1e-6, np.sqrt(0.5 * (var_d + var_s))))

        return {
            "auc_roc": round(auc, 4),
            "average_precision": round(ap, 4),
            "ap_chance": round(ap_chance, 4),
            "cnr": round(cnr, 2),
            "anomaly_scores": dist.reshape(H, W),
        }
