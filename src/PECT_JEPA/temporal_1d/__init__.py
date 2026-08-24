"""
1D Temporal PECT-JEPA (TS-JEPA) Package.
Self-Supervised Representation Learning on 1D Transient PECT Waveforms.
"""

from .models.jepa_1d import PECT_JEPA_1D
from .configs.config import Temporal1DConfig, get_default_config_1d

__all__ = ["PECT_JEPA_1D", "Temporal1DConfig", "get_default_config_1d"]
