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
from scipy import ndimage as ndi

from ..downstream.test_time_adaptation import whiten_matrix
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


def compute_otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Computes Otsu's optimal threshold on a 1D continuous array by maximizing
    inter-class variance between background and foreground distributions.
    """
    vals = np.asarray(values).reshape(-1)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.5
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin + 1e-7:
        return vmin

    counts, bin_edges = np.histogram(vals, bins=n_bins, range=(vmin, vmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = float(np.sum(counts))
    if total == 0:
        return 0.5

    p = counts / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    valid = denom > 1e-10
    sigma_b_sq = np.zeros_like(omega)
    sigma_b_sq[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    best_idx = int(np.argmax(sigma_b_sq))
    return float(bin_centers[best_idx])


HALO_RADIUS_PX = 2       # boundary ring excluded from readout fitting/mining
HARD_NEG_PER_SCAN = 2000  # top P0 false alarms mined as hard negatives per scan


def halo_ring(gt_mask: np.ndarray, radius: int = HALO_RADIUS_PX) -> np.ndarray:
    """Boolean ring around defect boundaries (dilated defect minus defect).

    Halo pixels are partial-volume mixtures: the CAD mask calls them sound while
    the measured field already responds to the flaw. Mining them as confident
    negatives taught P2 to suppress defect edges (train AUC collapsed to 0.37),
    so both probe fitting and hard-negative mining skip this ring.
    """
    defect = (gt_mask == 1)
    if not np.any(defect):
        return np.zeros_like(defect, dtype=bool)
    dil = ndi.binary_dilation(defect, iterations=int(max(1, radius)))
    return (dil & ~defect)


def whiten_stats(bg: np.ndarray, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """(mu, W) with W = (cov + eps*I)^(-1/2) estimated on a background pixel set."""
    bg = np.asarray(bg, dtype=np.float64)
    mu = bg.mean(axis=0).astype(np.float32)
    cov = np.cov(bg, rowvar=False)
    return mu, whiten_matrix(cov, eps=eps)


def unsup_bg_stats(flat_z: np.ndarray, valid_mask: Optional[np.ndarray] = None,
                   eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Background stats over ALL valid pixels (no GT, no anomaly cut)."""
    zv = flat_z[valid_mask] if valid_mask is not None else flat_z
    return whiten_stats(zv, eps=eps)


def scan_adaptive_features(
    flat_z: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    keep_q: float = 0.75,
    eps: float = 1e-2,
) -> np.ndarray:
    """P1u_trim_s2: per-scan unsupervised background whitening (deployable default).

    Estimates (mu, W) from the lowest-|energy| keep_q fraction of valid pixels
    (zero GT labels) and returns (z - mu) @ W. This is the same estimator as the
    official TTA path, so training and inference cannot drift apart.
    """
    from ..downstream.test_time_adaptation import unsup_trimmed_bg_stats

    mu, W = unsup_trimmed_bg_stats(flat_z, keep_q=keep_q, eps=eps, valid_mask=valid_mask)
    return (flat_z - mu).dot(W).astype(np.float32)


def mine_hard_negatives(
    gt_mask: np.ndarray,
    scores: np.ndarray,
    k: int = HARD_NEG_PER_SCAN,
    halo_radius: int = HALO_RADIUS_PX,
) -> np.ndarray:
    """P2h: indices of the k highest-scoring sound pixels, halo ring excluded.

    `scores` is a per-pixel false-alarm score from the base probe (e.g. P0
    predicted probability). Mining these structured background regions as
    negatives teaches the probe to suppress scan-specific clutter instead of
    memorising it as defect evidence.
    """
    y = gt_mask.reshape(-1)
    ring = halo_ring(gt_mask, radius=halo_radius).reshape(-1)
    sound = np.where((y == 0) & ~ring)[0]
    if len(sound) == 0:
        return np.zeros(0, dtype=np.int64)
    kk = min(int(k), len(sound))
    return sound[np.argpartition(-np.asarray(scores).reshape(-1)[sound], kk - 1)[:kk]]


def compute_instance_pod(
    score_map: np.ndarray,
    flaw_config: Dict[str, Any],
    threshold: float = 0.5,
    margin_px: int = 2,
) -> Dict[str, Any]:
    """
    Computes Instance-level Probability of Detection (POD) / Defect Hit Rate.
    Evaluates whether each known CAD flaw has a maximum detection score in its
    neighborhood exceeding the given threshold.

    Parameters:
    - score_map: [H, W] continuous score (probability or anomaly distance)
    - flaw_config: dictionary with 'features' list from CAD ground truth config
    - threshold: decision threshold for calling a hit
    - margin_px: pixel buffer margin around the CAD flaw boundary

    Returns:
    - dict with pod (hit rate), num_hits, num_defects, hit_details
    """
    if score_map is None or flaw_config is None or "features" not in flaw_config:
        return {"pod": 0.0, "num_hits": 0, "num_defects": 0, "threshold": float(threshold), "hit_details": []}

    H, W = score_map.shape[:2]
    features = flaw_config.get("features", [])
    hit_details = []
    num_hits = 0
    num_defects = 0

    for feat in features:
        # Check if feature is a genuine defect (not sound metal or sound rivet)
        kind = str(feat.get("kind", "")).lower()
        depth = feat.get("depth")
        # Defect criteria: either depth > 0 or kind indicates corrosion/defect
        if depth is not None:
            try:
                is_defect = float(depth) > 0.0
            except (ValueError, TypeError):
                is_defect = False
        else:
            is_defect = ("corrosion" in kind or "defect" in kind) and ("only" not in kind)

        if not is_defect:
            continue

        num_defects += 1
        fid = feat.get("id", num_defects)

        # Flaw center
        if feat.get("corrosionX") is not None and feat.get("corrosionY") is not None:
            cx = float(feat["corrosionX"])
            cy = float(feat["corrosionY"])
        elif feat.get("x") is not None and feat.get("y") is not None:
            cx = float(feat["x"])
            cy = float(feat["y"])
        else:
            continue

        dia = float(feat.get("diameter", 0.0) or 0.0)
        rad = max(2.0, dia / 2.0)
        eff_rad = rad + margin_px

        # Bounding box coordinates with clipping
        r0 = max(0, int(np.floor(cy - eff_rad)))
        r1 = min(H, int(np.ceil(cy + eff_rad + 1)))
        c0 = max(0, int(np.floor(cx - eff_rad)))
        c1 = min(W, int(np.ceil(cx + eff_rad + 1)))

        if r1 <= r0 or c1 <= c0:
            hit_details.append({
                "id": fid,
                "diameter": dia,
                "depth": depth,
                "peak_score": 0.0,
                "hit": False,
            })
            continue

        patch = score_map[r0:r1, c0:c1]
        grid_r, grid_c = np.mgrid[r0:r1, c0:c1]
        dist_sq = (grid_r - cy) ** 2 + (grid_c - cx) ** 2
        in_circle = dist_sq <= (eff_rad ** 2)

        if np.any(in_circle):
            peak_val = float(np.max(patch[in_circle]))
        else:
            peak_val = float(np.max(patch))

        is_hit = bool(peak_val >= threshold)
        if is_hit:
            num_hits += 1

        hit_details.append({
            "id": fid,
            "diameter": dia,
            "depth": depth,
            "peak_score": round(peak_val, 4),
            "hit": is_hit,
        })

    pod = float(num_hits / num_defects) if num_defects > 0 else 0.0
    return {
        "pod": round(pod, 4),
        "num_hits": num_hits,
        "num_defects": num_defects,
        "threshold": round(float(threshold), 4),
        "hit_details": hit_details,
    }


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
        size_log_target: bool = False,
    ) -> Dict[str, Any]:
        """
        Fits global downstream readouts on representations from the training pool:
        1. StandardScaler + LogisticRegression(class_weight="balanced") for Defect Detection
        2. StandardScaler + MLPClassifier for Non-linear Defect Detection
        3. StandardScaler + Ridge(alpha=10.0) for Quantitative Flaw Sizing (diameter in mm)

        size_log_target=True fits the regressors on log1p(diameter) instead of raw
        mm: defect diameters span 2-14 mm, so a raw L2 target lets the large flaws
        dominate the fit and the small ones collapse toward the mean (H31: sizing
        worse than a constant). `readouts["size_target"]` records the transform so
        inference inverts it exactly.
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
            y_fit = np.log1p(y_size.astype(np.float64)) if size_log_target else y_size
            ridge = Ridge(alpha=10.0, random_state=self.random_state)
            ridge.fit(X_size_s, y_fit)
            readouts["scaler_size"] = scaler_size
            readouts["ridge_size"] = ridge
            readouts["size_target"] = "log1p" if size_log_target else "identity"

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
        raw_pred = ridge.predict(feats_s)
        if readouts.get("size_target") == "log1p":
            pred_all = np.maximum(0.0, np.expm1(raw_pred))
        else:
            pred_all = np.maximum(0.0, raw_pred)

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
            return {
                "iou": 0.0,
                "dice": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "optimal_threshold": 0.5,
                "iou_05": 0.0,
                "dice_05": 0.0,
                "sound_fpr_05": 0.0,
                "threshold_otsu": 0.5,
                "iou_otsu": 0.0,
                "dice_otsu": 0.0,
                "sound_fpr_otsu": 0.0,
                "threshold_opt": 0.5,
                "iou_opt": 0.0,
                "dice_opt": 0.0,
                "sound_fpr_opt": 0.0,
                "sound_metal_fpr": 0.0,
            }

        sound_idx = (y_true == 0)
        n_sound = int(np.sum(sound_idx))

        # 1. Fixed standard threshold 0.5
        pred_05 = (p_val >= 0.5).astype(int)
        iou_05 = float(jaccard_score(y_true, pred_05, zero_division=0))
        dice_05 = float(f1_score(y_true, pred_05, zero_division=0))
        sound_fpr_05 = float(np.mean(pred_05[sound_idx])) if n_sound > 0 else 0.0

        # 2. Otsu adaptive threshold
        tau_otsu = compute_otsu_threshold(p_val)
        pred_otsu = (p_val >= tau_otsu).astype(int)
        iou_otsu = float(jaccard_score(y_true, pred_otsu, zero_division=0))
        dice_otsu = float(f1_score(y_true, pred_otsu, zero_division=0))
        sound_fpr_otsu = float(np.mean(pred_otsu[sound_idx])) if n_sound > 0 else 0.0

        # 3. Optimal threshold search
        if threshold is not None:
            best_tau = float(threshold)
            y_pred = (p_val >= best_tau).astype(int)
            best_iou = float(jaccard_score(y_true, y_pred, zero_division=0))
            best_dice = float(f1_score(y_true, y_pred, zero_division=0))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            sound_fpr_opt = float(np.mean(y_pred[sound_idx])) if n_sound > 0 else 0.0
        else:
            taus = np.linspace(0.1, 0.95, 35)
            p_pos = p_val[y_true == 1]
            p_neg = p_val[sound_idx]
            n_pos = len(p_pos)
            n_neg = len(p_neg)

            if n_pos > 0 and n_neg > 0:
                tp = np.sum(p_pos[:, None] >= taus[None, :], axis=0).astype(np.float64)
                fp = np.sum(p_neg[:, None] >= taus[None, :], axis=0).astype(np.float64)
                fn = n_pos - tp

                precisions = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp > 0))
                recalls = np.divide(tp, n_pos, out=np.zeros_like(tp), where=(n_pos > 0))
                dices = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn > 0))
                ious = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn > 0))
                fprs = np.divide(fp, n_neg, out=np.zeros_like(fp), where=(n_neg > 0))

                best_idx = int(np.argmax(dices))
                best_dice = float(dices[best_idx])
                best_tau = float(taus[best_idx])
                best_iou = float(ious[best_idx])
                prec = float(precisions[best_idx])
                rec = float(recalls[best_idx])
                sound_fpr_opt = float(fprs[best_idx])
            else:
                best_dice, best_iou, best_tau = 0.0, 0.0, 0.5
                prec, rec, sound_fpr_opt = 0.0, 0.0, 0.0

        return {
            "iou": round(best_iou, 4),
            "dice": round(best_dice, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "optimal_threshold": round(best_tau, 4),
            "iou_05": round(iou_05, 4),
            "dice_05": round(dice_05, 4),
            "sound_fpr_05": round(sound_fpr_05, 4),
            "threshold_otsu": round(tau_otsu, 4),
            "iou_otsu": round(iou_otsu, 4),
            "dice_otsu": round(dice_otsu, 4),
            "sound_fpr_otsu": round(sound_fpr_otsu, 4),
            "threshold_opt": round(best_tau, 4),
            "iou_opt": round(best_iou, 4),
            "dice_opt": round(best_dice, 4),
            "f1_max": round(best_dice, 4),
            "sound_fpr_opt": round(sound_fpr_opt, 4),
            "sound_metal_fpr": round(sound_fpr_05, 4),
        }

    @staticmethod
    def compute_otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
        return compute_otsu_threshold(values, n_bins)

    @staticmethod
    def compute_instance_pod(
        score_map: np.ndarray,
        flaw_config: Dict[str, Any],
        threshold: float = 0.5,
        margin_px: int = 2,
    ) -> Dict[str, Any]:
        return compute_instance_pod(score_map, flaw_config, threshold, margin_px)

    # =========================================================================
    # Task 2: Quantitative Physical Sizing (Conditional Hurdle Regression)
    # =========================================================================
    def benchmark_sizing_regression(
        self,
        features: np.ndarray,
        size_map: np.ndarray,  # continuous defect diameter or depth in mm
        focus_defects_only: bool = True,
        log_target: bool = False,
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

        # Fit in log1p space (wide 2-14 mm range) and invert for the reported metrics.
        y_fit = np.log1p(y.astype(np.float64)) if log_target else y
        inv = (lambda v: np.maximum(0.0, np.expm1(v))) if log_target else (lambda v: v)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        linear_r2, linear_mae, linear_rmse = [], [], []
        mlp_r2, mlp_mae, mlp_rmse = [], [], []

        for train_idx, test_idx in kf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_fit[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. Ridge Regressor
            ridge = Ridge(alpha=10.0, random_state=self.random_state)
            ridge.fit(X_tr_s, y_tr)
            y_pred_lr = inv(ridge.predict(X_te_s))

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
            y_pred_mlp = inv(mlp.predict(X_te_s))

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
        flat_latents = np.nan_to_num(latents.reshape(-1, D).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        flat_labels = (ground_truth_mask.reshape(-1) > 0.5).astype(np.int64)

        sound_idx = (flat_labels == 0)
        z_sound = flat_latents[sound_idx]
        mu_sound = np.mean(z_sound, axis=0)
        cov_sound = np.cov(z_sound, rowvar=False) + np.eye(D) * 1e-4
        try:
            inv_cov = np.linalg.pinv(cov_sound, hermitian=True)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(D)

        diff = flat_latents - mu_sound
        dist_sq = np.sum((diff @ inv_cov) * diff, axis=-1)
        dist = np.nan_to_num(np.sqrt(np.maximum(dist_sq, 0.0)), nan=0.0, posinf=0.0, neginf=0.0)

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
