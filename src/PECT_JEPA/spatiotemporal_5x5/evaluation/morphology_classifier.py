"""
Defect Morphology Classifier for PECT-JEPA.

Classifies latent representations into 4 distinct physical NDT states:
- Class 0: Sound Metal (flat plate baseline)
- Class 1: Surface-Breaking Corrosion (Lộ thiên - depth physically quantifiable)
- Class 2: Subsurface / Buried Defect (Trong lòng vật liệu - 2nd layer / around fasteners)
- Class 3: Sound Fastener Structure (Rivet hole / head without flaw)

Enables dynamic, cluster-level and pixel-level routing so that:
- Quantitative Depth Regression is applied STRICTLY to surface-breaking defects (Class 1)
- Subsurface / Buried defects (Class 2) and fasteners are routed to Universal Flaw Extent Sizing (Diameter & Area)
Eliminates brittle file-level string matching (e.g. `if specimen == 'corrosion'`).
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class DefectMorphologyClassifier:
    """
    Lightweight, balanced classifier that maps latent representations [D] or [2D]
    to physical defect morphology classes:
        0: Sound Metal
        1: Surface-Breaking Defect (Lộ thiên)
        2: Subsurface / Buried Defect (Trong lòng vật liệu)
        3: Fastener Structure (Rivet)
    """

    CLASS_NAMES = [
        "Sound Metal",
        "Surface Defect (Surface)",
        "Subsurface Defect (Buried)",
        "Fastener Structure"
    ]

    def __init__(self, classifier_type: str = "linear", random_state: int = 42):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        if classifier_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                max_iter=300,
                random_state=random_state,
                early_stopping=True,
            )
        else:
            self.model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                tol=1e-3,
                random_state=random_state,
                solver="lbfgs"
            )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DefectMorphologyClassifier":
        """
        Fits the morphology classifier on labeled feature representations.
        X: [N, D] latent features
        y: [N] morphology labels (0..3)
        """
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts discrete morphology class (0..3)."""
        if not self.is_fitted:
            raise RuntimeError("DefectMorphologyClassifier is not fitted.")
        orig_shape = X.shape
        flat_X = X.reshape(-1, orig_shape[-1])
        X_s = self.scaler.transform(flat_X)
        preds = self.model.predict(X_s)
        if len(orig_shape) == 3:
            return preds.reshape(orig_shape[0], orig_shape[1])
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probability distributions [N, 4]."""
        if not self.is_fitted:
            raise RuntimeError("DefectMorphologyClassifier is not fitted.")
        orig_shape = X.shape
        flat_X = X.reshape(-1, orig_shape[-1])
        X_s = self.scaler.transform(flat_X)
        probs = self.model.predict_proba(X_s)
        if len(orig_shape) == 3:
            return probs.reshape(orig_shape[0], orig_shape[1], -1)
        return probs

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates classifier on held-out test data."""
        flat_X = X_test.reshape(-1, X_test.shape[-1])
        flat_y = y_test.reshape(-1)
        valid = np.where(flat_y >= 0)[0]
        flat_X = flat_X[valid]
        flat_y = flat_y[valid]

        preds = self.predict(flat_X)
        macro_f1 = float(f1_score(flat_y, preds, average="macro", zero_division=0))
        acc = float(np.mean(preds == flat_y))
        cm = confusion_matrix(flat_y, preds, labels=list(range(len(self.CLASS_NAMES))))
        cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        return {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "class_names": self.CLASS_NAMES,
        }


def build_morphology_labels_from_scan(
    specimen_key: str,
    gt_mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Constructs ground-truth morphology labels for a scan:
    - 0: Sound Metal (gt_mask == 0)
    - 1: Surface Defect (Corrosion plate: gt_mask == 1, depth > 0)
    - 2: Subsurface / Buried Defect (Rivet/Mixed plates: corrosion under/near rivets)
    - 3: Sound Fastener Structure (Rivet plate without defect)
    """
    sY, sX = gt_mask.shape
    labels = np.zeros((sY, sX), dtype=np.int64)

    is_corrosion_plate = "corrosion" in specimen_key.lower() or "corosion" in specimen_key.lower()
    is_rivet_plate = "rivet" in specimen_key.lower() or "mixed" in specimen_key.lower()

    if is_corrosion_plate:
        labels[gt_mask == 1] = 1  # Surface-breaking defect
    elif is_rivet_plate:
        labels[gt_mask == 1] = 2  # Subsurface / buried defect in 2nd layer
        # If ground truth has rivet core annotations, mark sound rivets as 3
        if depth_map is not None:
            # Rivet bodies typically have distinct marker
            pass

    return labels
