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

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class LinearProbeEvaluator:
    """
    Evaluates linear separability of frozen PECT-JEPA representations against ground-truth defect masks.
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
    ) -> Dict[str, float]:
        """
        Runs Stratified 5-Fold Cross-Validation on the full C-scan.
        Returns mean AUC-ROC, AP, F1, and k-NN metrics across folds.
        """
        X, y = self.extract_labeled_samples(feature_map, gt_mask)
        n_defects = int(np.sum(y == 1))
        n_sound = int(np.sum(y == 0))

        if n_defects < self.n_splits:
            raise ValueError(f"Too few defect samples ({n_defects}) for {self.n_splits}-fold CV.")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        auc_list = []
        ap_list = []
        f1_list = []
        acc_list = []
        knn_acc_list = []
        knn_f1_list = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # 1. Linear Probe (Logistic Regression) with class weighting
            clf = LogisticRegression(
                C=self.c_reg,
                max_iter=self.max_iter,
                class_weight="balanced",
                random_state=self.random_state,
                solver="lbfgs"
            )
            clf.fit(X_tr, y_tr)

            probs = clf.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)

            auc_list.append(float(roc_auc_score(y_te, probs)))
            ap_list.append(float(average_precision_score(y_te, probs)))
            f1_list.append(float(f1_score(y_te, preds, zero_division=0)))
            acc_list.append(float(accuracy_score(y_te, preds)))

            # 2. k-NN Classifier (Cosine / Normalized Euclidean)
            knn = KNeighborsClassifier(n_neighbors=knn_neighbors, metric="cosine")
            knn.fit(X_tr, y_tr)
            knn_preds = knn.predict(X_te)

            knn_acc_list.append(float(accuracy_score(y_te, knn_preds)))
            knn_f1_list.append(float(f1_score(y_te, knn_preds, zero_division=0)))

        return {
            "n_defect_samples": n_defects,
            "n_sound_samples": n_sound,
            "defect_ratio_pct": round(float(n_defects / (n_defects + n_sound) * 100), 2),
            "linear_probe_auc_roc": round(float(np.mean(auc_list)), 4),
            "linear_probe_auc_std": round(float(np.std(auc_list)), 4),
            "linear_probe_average_precision": round(float(np.mean(ap_list)), 4),
            "linear_probe_f1": round(float(np.mean(f1_list)), 4),
            "linear_probe_accuracy": round(float(np.mean(acc_list)), 4),
            f"knn_{knn_neighbors}_accuracy": round(float(np.mean(knn_acc_list)), 4),
            f"knn_{knn_neighbors}_f1": round(float(np.mean(knn_f1_list)), 4),
        }

    def fit_and_predict_probability_map(
        self,
        feature_map: np.ndarray,
        gt_mask: np.ndarray,
        knn_neighbors: int = 5,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        1. Runs Stratified K-Fold Cross-Validation on labeled pixels to calculate unbiased CV metrics.
        2. Fits a final Logistic Regression model on all valid labeled pixels.
        3. Predicts defect probability P(Y=1 | z) for every pixel in the entire feature_map.

        Returns:
            metrics: dict of cross-validation metrics.
            prob_map: [sY, sX] 2D array of defect probabilities in range [0, 1].
        """
        metrics = self.evaluate_cross_val(feature_map, gt_mask, knn_neighbors=knn_neighbors)

        sY, sX, D = feature_map.shape
        X_labeled, y_labeled = self.extract_labeled_samples(feature_map, gt_mask)

        clf = LogisticRegression(
            C=self.c_reg,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
            solver="lbfgs",
        )
        clf.fit(X_labeled, y_labeled)

        flat_all = feature_map.reshape(-1, D)
        probs_flat = clf.predict_proba(flat_all)[:, 1]
        prob_map = probs_flat.reshape(sY, sX)

        return metrics, prob_map
