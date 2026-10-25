"""
Downstream Flaw Sizing Models.

Provides:
- Ridge Flaw Sizing Regressor (Ridge regression with StandardScaler)
- MLP Flaw Sizing Regressor
- PyTorch implementation for tensor batch inference.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class PyTorchSizingRegressor(nn.Module):
    """PyTorch MLP Regressor for flaw depth/diameter sizing in mm."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # Defect depth/diameter is strictly non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SizingRegressorBundle:
    """Wrapper holding trained sklearn sizing models with unified numpy inference."""
    def __init__(
        self,
        scaler: StandardScaler,
        ridge_regressor: Ridge,
        mlp_regressor: Optional[MLPRegressor] = None,
        target_transform: str = "identity",
    ):
        self.scaler = scaler
        self.ridge_regressor = ridge_regressor
        self.mlp_regressor = mlp_regressor
        # "log1p" => the regressors were fitted on log1p(mm); invert at inference.
        self.target_transform = target_transform

    def _invert(self, pred: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return np.maximum(np.expm1(pred), 0.0)
        return np.maximum(pred, 0.0)

    def predict_ridge(self, X: np.ndarray) -> np.ndarray:
        """Predicts flaw dimension (mm) using Ridge Regressor. X: [N, D]"""
        Xs = self.scaler.transform(X)
        return self._invert(self.ridge_regressor.predict(Xs))

    def predict_mlp(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predicts flaw dimension (mm) using MLP Regressor. X: [N, D]"""
        if self.mlp_regressor is None:
            return None
        Xs = self.scaler.transform(X)
        return self._invert(self.mlp_regressor.predict(Xs))
