"""
Downstream tasks package for Cross-Condition PECT-JEPA.

Organized by tasks:
- detection/ : Linear Probe, MLP Probe
- sizing/    : Flaw Sizing Regressors
- domain_adaptation/ : UDA models (CORAL, DANN)
"""

from .detection.models import DetectionProbeBundle, PyTorchLinearProbe, PyTorchMLPProbe
from .sizing.models import SizingRegressorBundle, PyTorchSizingRegressor
from .domain_adaptation.coral import CORALAdapter, coral_loss
from .domain_adaptation.dann import DANNModel

__all__ = [
    "DetectionProbeBundle",
    "PyTorchLinearProbe",
    "PyTorchMLPProbe",
    "SizingRegressorBundle",
    "PyTorchSizingRegressor",
    "CORALAdapter",
    "coral_loss",
    "DANNModel",
]
