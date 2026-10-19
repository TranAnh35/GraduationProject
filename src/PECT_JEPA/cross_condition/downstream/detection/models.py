"""
Downstream Defect Detection Models.

Provides:
- Standard Linear Probe (Logistic Regression)
- 2-Layer MLP Probe
- PyTorch Linear and MLP modules for end-to-end tensor batch inference.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class PyTorchLinearProbe(nn.Module):
    """PyTorch implementation of Linear Probe with optional batch normalization."""
    def __init__(self, in_dim: int, num_classes: int = 1):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim, affine=True)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        return self.fc(self.norm(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


class PyTorchMLPProbe(nn.Module):
    """PyTorch 2-layer MLP probe."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)


class DetectionProbeBundle:
    """Wrapper holding trained sklearn detection models with unified numpy inference."""
    def __init__(
        self,
        scaler: StandardScaler,
        linear_probe: LogisticRegression,
        mlp_probe: Optional[MLPClassifier] = None,
    ):
        self.scaler = scaler
        self.linear_probe = linear_probe
        self.mlp_probe = mlp_probe

    def predict_linear(self, X: np.ndarray) -> np.ndarray:
        """Predicts defect probability using Linear Probe. X: [N, D]"""
        Xs = self.scaler.transform(X)
        if hasattr(self.linear_probe, "predict_proba"):
            return self.linear_probe.predict_proba(Xs)[:, 1]
        # Fallback to decision function sigmoid
        df = self.linear_probe.decision_function(Xs)
        return 1.0 / (1.0 + np.exp(-df))

    def predict_mlp(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predicts defect probability using MLP Probe. X: [N, D]"""
        if self.mlp_probe is None:
            return None
        Xs = self.scaler.transform(X)
        return self.mlp_probe.predict_proba(Xs)[:, 1]
