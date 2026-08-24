"""
PECT-JEPA package containing two independent research directions:
1. Spatio-Temporal 3D PECT-JEPA (spatiotomporal_3d)
2. 1D Temporal PECT-JEPA / TS-JEPA (temporal_1d)
"""

from .spatiotomporal_3d.models.jepa import PECT_JEPA
from .temporal_1d.models.jepa_1d import PECT_JEPA_1D

__all__ = ["PECT_JEPA", "PECT_JEPA_1D"]
