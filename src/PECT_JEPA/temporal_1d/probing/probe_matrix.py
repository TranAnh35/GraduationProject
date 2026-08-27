"""
Probe matrix & representation diagnostics for PECT-JEPA (Stage C).

Tools:
  - extract_pooled_features: frozen-encoder pooled features [N, D]
  - effective_rank: entropy-based rank (collapse early-warning)
  - linear_cka: representational similarity between two feature sets
  - LinearProbe: closed-form ridge one-vs-rest classification probe
  - run_probe_matrix: per-factor probe accuracies (sensor/waveform/liftoff/defect)

Interpretation contract (physics grounding):
  * Probes on NUISANCE factors (sensor, waveform, liftoff) should be WEAK.
  * Probes on DEFECT factors should be STRONG.
"""

from typing import Dict, Optional, List
import numpy as np
import torch


@torch.no_grad()
def extract_pooled_features(
    model,
    loader,
    device: torch.device = torch.device("cpu"),
    max_batches: Optional[int] = None
) -> np.ndarray:
    """Extract pooled frozen features [N, D] from a DataLoader of 1D batches."""
    model.eval()
    feats: List[np.ndarray] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["data"] if isinstance(batch, dict) else batch[0]
        x = x.to(device)
        h = model.extract_features(x, pool=True)  # [B, D]
        feats.append(h.cpu().numpy())
    return np.concatenate(feats, axis=0)


def effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank (Roy & Vetterli): exp(Shannon entropy of
    normalized squared singular values). A collapsed embedding has rank ~1.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), eps)
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Linear CKA between two feature matrices with the same number of rows."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")) + eps
    return float(num / den)


class LinearProbe:
    """
    Closed-form ridge regression probe onto one-hot class targets.
    No sklearn dependency. Prediction = argmax of linear outputs.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.W: Optional[np.ndarray] = None
        self.mu_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        classes = np.unique(y)
        self.classes_ = classes
        D = X.shape[1]
        self.mu_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mu_
        Y = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
        for j, c in enumerate(classes):
            Y[y == c, j] = 1.0
        Y = Y - Y.mean(axis=0, keepdims=True)
        A = Xc.T @ Xc + self.ridge * np.eye(D)
        self.W = np.linalg.solve(A, Xc.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.mu_ is not None
        Xc = X - self.mu_  # NOTE: train-set mean, NOT the prediction-set mean
        idx = np.argmax(Xc @ self.W, axis=1)
        return self.classes_[idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


def run_probe_matrix(
    features: np.ndarray,
    labels: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge: float = 1e-3
) -> Dict[str, float]:
    """
    Train one LinearProbe per factor and return test accuracies.

    Args:
        features: [N, D] pooled features
        labels: {factor_name: [N] int labels}
        train_mask / test_mask: boolean [N] masks (no row overlap)
    """
    results: Dict[str, float] = {}
    for factor, y in labels.items():
        if len(np.unique(y[train_mask])) < 2:
            results[factor] = float("nan")
            continue
        probe = LinearProbe(ridge=ridge).fit(features[train_mask], y[train_mask])
        results[factor] = probe.score(features[test_mask], y[test_mask])
    return results
