"""
Downstream tasks package for Cross-Condition PECT-JEPA.

Organized by tasks:
- detection/ : Linear Probe, MLP Probe
- sizing/    : Flaw Sizing Regressors
- domain_adaptation/ : UDA models (CORAL, DANN, DAN, CDAN, SHOT)
"""

from .detection.models import DetectionProbeBundle, PyTorchLinearProbe, PyTorchMLPProbe
from .sizing.models import SizingRegressorBundle, PyTorchSizingRegressor
from .domain_adaptation.coral import CORALAdapter, coral_loss
from .domain_adaptation.dann import DANNModel
from .domain_adaptation.dan_mmd import DANModel, MultiKernelMMD
from .domain_adaptation.cdan import CDANModel
from .domain_adaptation.shot import SHOTModel

__all__ = [
    "DetectionProbeBundle",
    "PyTorchLinearProbe",
    "PyTorchMLPProbe",
    "SizingRegressorBundle",
    "PyTorchSizingRegressor",
    "CORALAdapter",
    "coral_loss",
    "DANNModel",
    "DANModel",
    "MultiKernelMMD",
    "CDANModel",
    "SHOTModel",
]
