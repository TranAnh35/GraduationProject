"""
PECT-JEPA package containing two independent research directions:
1. Spatio-Temporal 3D PECT-JEPA (spatiotomporal_3d)
2. 1D Temporal PECT-JEPA / TS-JEPA (temporal_1d)
"""

__all__ = ["PECT_JEPA", "PECT_JEPA_1D"]


def __getattr__(name):
    if name == "PECT_JEPA":
        from .spatiotomporal_3d.models.jepa import PECT_JEPA
        return PECT_JEPA
    elif name == "PECT_JEPA_1D":
        from .temporal_1d.models.jepa_1d import PECT_JEPA_1D
        return PECT_JEPA_1D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
