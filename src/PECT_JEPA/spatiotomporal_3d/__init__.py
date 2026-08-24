"""
PECT-JEPA: Self-Supervised Joint Embedding Predictive Architecture for Pulsed Eddy Current Testing.
"""

from .models.jepa import PECT_JEPA
from .configs.config import PECTJEPAConfig, get_default_config
from .data.dataset import PECTDataset, read_tdms_scan
from .training.trainer import JEPATrainer

__version__ = "0.1.0"

__all__ = [
    "PECT_JEPA",
    "PECTJEPAConfig",
    "get_default_config",
    "PECTDataset",
    "read_tdms_scan",
    "JEPATrainer",
]
