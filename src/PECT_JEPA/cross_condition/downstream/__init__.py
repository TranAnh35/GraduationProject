"""
Downstream tasks package for Cross-Condition PECT-JEPA.

Organized by tasks:
- detection/ : Linear Probe, MLP Probe
- sizing/    : Flaw Sizing Regressors
- test_time_adaptation : Test-Time Adaptation & Spatial Self-Calibration (TTA)
"""

from .detection.models import DetectionProbeBundle, PyTorchLinearProbe, PyTorchMLPProbe
from .sizing.models import SizingRegressorBundle, PyTorchSizingRegressor
from .test_time_adaptation import (
    whiten_matrix,
    unsup_trimmed_bg_stats,
    TestTimeSelfCalibrator,
)

__all__ = [
    "DetectionProbeBundle",
    "PyTorchLinearProbe",
    "PyTorchMLPProbe",
    "SizingRegressorBundle",
    "PyTorchSizingRegressor",
    "whiten_matrix",
    "unsup_trimmed_bg_stats",
    "TestTimeSelfCalibrator",
]
