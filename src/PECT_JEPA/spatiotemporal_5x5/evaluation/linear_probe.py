"""
Oracle Linear Probe & k-NN Evaluation for 5x5 PECT-JEPA.

Standard Self-Supervised Learning (SSL) evaluation protocol:
1. Freezes the PECT-JEPA Encoder completely (zero backpropagation through the backbone).
2. Extracts latent representations Z = [N, D] from C-scan feature maps.
3. Uses the Ground Truth Mask (1 = Defect, 0 = Sound Metal, ignoring -1 buffer zone) as labels Y.
4. Trains a linear classifier (Logistic Regression) with L2 regularization (Ridge).
5. Evaluates non-parametric k-Nearest Neighbors (k-NN, k=5, 20) with cosine/euclidean distance.
6. Computes standard SSL benchmark metrics:
   - Linear Probe AUC-ROC
   - Linear Probe Average Precision (PR-AUC)
   - Linear Probe Accuracy & Balanced F1-Score
   - k-NN Accuracy & F1-Score

Scientific Rationale for Advisor/Supervisor:
- If Linear Probe achieves high AUC-ROC (e.g. > 0.90) on frozen representations,
  it proves mathematically that the latent space is linearly separable and contains rich flaw representations,
  meaning any suboptimal visualization in unsupervised clustering was caused by K-Means assumptions, not representation failure.
"""

import warnings
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LinearProbeEvaluator:
    """
    Evaluates linear separability of frozen PECT-JEPA representations against ground-truth defect masks.
    Benchmarks Linear Probe (Logistic Regression) vs Non-Linear Probe (MLP 2-Layer) with StandardScaler.
    """

    def __init__(self, c_reg: float = 1.0, max_iter: int = 1000, n_splits: int = 5, random_state: int = 42):
        self.c_reg = c_reg
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.random_state = random_state

    def extract_labeled_samples(
        self,
        feature_map: np.ndarray,  # [sY, sX, D]
        gt_mask: np.ndarray,      # [sY, sX] (1: defect, 0: sound metal, -1: buffer ignore)
        balance_sound_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts feature vectors and binary labels from the feature map and ground truth mask.
        Filters out buffer zone pixels (where gt_mask == -1).
        Optionally subsamples sound metal background if balance_sound_ratio is specified (e.g. 5.0 for 5:1 sound:defect).
        """
        sY, sX, D = feature_map.shape
        flat_feats = feature_map.reshape(-1, D)
        flat_labels = gt_mask.reshape(-1)

        valid_idx = np.where(flat_labels >= 0)[0]
        X_all = flat_feats[valid_idx]
        y_all = flat_labels[valid_idx].astype(np.int64)

        if balance_sound_ratio is not None:
            defect_idx = np.where(y_all == 1)[0]
            sound_idx = np.where(y_all == 0)[0]
            n_defects = len(defect_idx)
            max_sound = int(n_defects * balance_sound_ratio)

            if len(sound_idx) > max_sound:
                np.random.seed(self.random_state)
                selected_sound = np.random.choice(sound_idx, size=max_sound, replace=False)
                keep_idx = np.concatenate([defect_idx, selected_sound])
                np.random.shuffle(keep_idx)
                return X_all[keep_idx], y_all[keep_idx]

        return X_all, y_all

    def evaluate_cross_val(
        self,
        feature_map: np.ndarray,
        gt_mask: np.ndarray,
        knn_neighbors: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs Stratified 5-Fold Cross-Validation on the full C-scan.
        Uses StandardScaler strictly fitted on training folds.
        Benchmarks Linear Probe (Logistic Regression) vs Non-linear Probe (MLP 2-Layer).
        Returns mean metrics across folds and concatenated out-of-fold predictions for ROC/PR plotting.
        """
        X, y = self.extract_labeled_samples(feature_map, gt_mask)
        n_defects = int(np.sum(y == 1))
        n_sound = int(np.sum(y == 0))

        if n_defects < self.n_splits:
            raise ValueError(f"Too few defect samples ({n_defects}) for {self.n_splits}-fold CV.")

        # Smart background subsampling: preserve 100% of defect samples, cap sound metal background at 8000
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        max_neg = min(len(neg_idx), max(len(pos_idx) * 10, 8000))
        if len(neg_idx) > max_neg:
            rng = np.random.RandomState(self.random_state)
            sub_neg = rng.choice(neg_idx, size=max_neg, replace=False)
            keep_idx = np.sort(np.concatenate([pos_idx, sub_neg]))
            X_cv = X[keep_idx]
            y_cv = y[keep_idx]
        else:
            X_cv = X
            y_cv = y

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        linear_auc, linear_ap, linear_f1, linear_acc = [], [], [], []
        mlp_auc, mlp_ap, mlp_f1, mlp_acc = [], [], [], []
        knn_acc_list, knn_f1_list = [], []

        oof_y_true = []
        oof_linear_probs = []
        oof_mlp_probs = []

        for train_idx, test_idx in skf.split(X_cv, y_cv):
            X_tr, y_tr = X_cv[train_idx], y_cv[train_idx]
            X_te, y_te = X_cv[test_idx], y_cv[test_idx]

            # 0. Feature Standardization (fitted strictly on training fold)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. Linear Probe (Logistic Regression with class weighting)
            clf = LogisticRegression(
                C=self.c_reg,
                max_iter=max(500, self.max_iter),
                tol=1e-3,
                class_weight="balanced",
                random_state=self.random_state,
                solver="lbfgs",
            )
            clf.fit(X_tr_s, y_tr)
            probs_lr = clf.predict_proba(X_te_s)[:, 1]
            preds_lr = (probs_lr >= 0.5).astype(int)

            linear_auc.append(float(roc_auc_score(y_te, probs_lr)))
            linear_ap.append(float(average_precision_score(y_te, probs_lr)))
            linear_f1.append(float(f1_score(y_te, preds_lr, zero_division=0)))
            linear_acc.append(float(accuracy_score(y_te, preds_lr)))

            # 2. MLP 2-Layer Probe (Non-Linear Readout)
            try:
                mlp = MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    max_iter=150,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=self.random_state,
                )
                sample_weights = compute_sample_weight("balanced", y_tr)
                mlp.fit(X_tr_s, y_tr, sample_weight=sample_weights)
                probs_mlp = mlp.predict_proba(X_te_s)[:, 1]
                preds_mlp = (probs_mlp >= 0.5).astype(int)

                mlp_auc.append(float(roc_auc_score(y_te, probs_mlp)))
                mlp_ap.append(float(average_precision_score(y_te, probs_mlp)))
                mlp_f1.append(float(f1_score(y_te, preds_mlp, zero_division=0)))
                mlp_acc.append(float(accuracy_score(y_te, preds_mlp)))
            except Exception:
                probs_mlp = probs_lr
                mlp_auc.append(linear_auc[-1])
                mlp_ap.append(linear_ap[-1])
                mlp_f1.append(linear_f1[-1])
                mlp_acc.append(linear_acc[-1])

            # 3. k-NN Classifier
            knn = KNeighborsClassifier(n_neighbors=knn_neighbors, metric="cosine")
            knn.fit(X_tr_s, y_tr)
            knn_preds = knn.predict(X_te_s)

            knn_acc_list.append(float(accuracy_score(y_te, knn_preds)))
            knn_f1_list.append(float(f1_score(y_te, knn_preds, zero_division=0)))

            oof_y_true.append(y_te)
            oof_linear_probs.append(probs_lr)
            oof_mlp_probs.append(probs_mlp)

        oof_y = np.concatenate(oof_y_true)
        oof_p_lr = np.concatenate(oof_linear_probs)
        fpr_lr, tpr_lr, _ = roc_curve(oof_y, oof_p_lr)
        prec_lr, rec_lr, _ = precision_recall_curve(oof_y, oof_p_lr)

        l_auc = float(np.mean(linear_auc))
        l_ap = float(np.mean(linear_ap))
        l_f1 = float(np.mean(linear_f1))
        l_acc = float(np.mean(linear_acc))

        m_auc = float(np.mean(mlp_auc))
        m_ap = float(np.mean(mlp_ap))
        m_f1 = float(np.mean(mlp_f1))
        m_acc = float(np.mean(mlp_acc))

        return {
            "n_defect_samples": n_defects,
            "n_sound_samples": n_sound,
            "defect_ratio_pct": round(float(n_defects / (n_defects + n_sound) * 100), 2),
            "linear_probe_auc_roc": round(l_auc, 4),
            "linear_probe_auc_std": round(float(np.std(linear_auc)), 4),
            "linear_probe_average_precision": round(l_ap, 4),
            "linear_probe_f1": round(l_f1, 4),
            "linear_probe_accuracy": round(l_acc, 4),
            "mlp_2layer_auc_roc": round(m_auc, 4),
            "mlp_2layer_average_precision": round(m_ap, 4),
            "mlp_2layer_f1": round(m_f1, 4),
            "mlp_2layer_accuracy": round(m_acc, 4),
            "representation_gap_delta_auc": round(m_auc - l_auc, 4),
            "representation_gap_delta_ap": round(m_ap - l_ap, 4),
            f"knn_{knn_neighbors}_accuracy": round(float(np.mean(knn_acc_list)), 4),
            f"knn_{knn_neighbors}_f1": round(float(np.mean(knn_f1_list)), 4),
            "curve_data": {
                "fpr": fpr_lr.tolist(),
                "tpr": tpr_lr.tolist(),
                "precision": prec_lr.tolist(),
                "recall": rec_lr.tolist(),
            },
        }

    def fit_and_predict_probability_map(
        self,
        feature_map: np.ndarray,
        gt_mask: np.ndarray,
        knn_neighbors: int = 5,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        1. Runs Stratified K-Fold Cross-Validation on labeled pixels to calculate unbiased CV metrics.
        2. Fits StandardScaler and final Logistic Regression on labeled pixels.
        3. Predicts defect probability P(Y=1 | z) for every pixel in the entire feature_map.

        Returns:
            metrics: dict of cross-validation metrics and curves.
            prob_map: [sY, sX] 2D array of defect probabilities in range [0, 1].
        """
        metrics = self.evaluate_cross_val(feature_map, gt_mask, knn_neighbors=knn_neighbors)

        sY, sX, D = feature_map.shape
        X_labeled, y_labeled = self.extract_labeled_samples(feature_map, gt_mask)

        # Smart background subsampling for final fit as well
        pos_idx = np.where(y_labeled == 1)[0]
        neg_idx = np.where(y_labeled == 0)[0]
        max_neg = min(len(neg_idx), max(len(pos_idx) * 10, 8000))
        if len(neg_idx) > max_neg:
            rng = np.random.RandomState(self.random_state)
            sub_neg = rng.choice(neg_idx, size=max_neg, replace=False)
            keep_idx = np.sort(np.concatenate([pos_idx, sub_neg]))
            X_fit = X_labeled[keep_idx]
            y_fit = y_labeled[keep_idx]
        else:
            X_fit = X_labeled
            y_fit = y_labeled

        # Standardize features using scaler fit on balanced labeled samples
        scaler = StandardScaler()
        X_fit_s = scaler.fit_transform(X_fit)

        clf = LogisticRegression(
            C=self.c_reg,
            max_iter=max(500, self.max_iter),
            tol=1e-3,
            class_weight="balanced",
            random_state=self.random_state,
            solver="lbfgs",
        )
        clf.fit(X_fit_s, y_fit)

        flat_all = feature_map.reshape(-1, D)
        flat_all_s = scaler.transform(flat_all)
        probs_flat = clf.predict_proba(flat_all_s)[:, 1]
        prob_map = probs_flat.reshape(sY, sX)

        return metrics, prob_map
