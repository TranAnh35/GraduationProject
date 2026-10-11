"""
Spatial topology definitions for PECT probe extraction.
"""

from typing import Tuple
import numpy as np


def get_spatial_topology_offsets(
    topology: str = "concentric_star",
    star_radii: Tuple[int, int, int] = (1, 3, 7),
    grid_size: int = 5
) -> np.ndarray:
    """
    Returns 2D integer spatial pixel offsets [25, 2] for probe extraction.
    Concentric star topology:
      - Center probe: (0, 0) (idx 0)
      - Inner ring (r1 = 1 mm): 8 probes (indices 1..8) [Core Target: 9 probes]
      - Middle ring (r2 = 3 mm): 8 probes (indices 9..16)
      - Outer ring (r3 = 7 mm): 8 probes (indices 17..24) [Outer Context: 16 probes]
    """
    if topology == "concentric_star":
        offsets = [(0, 0)]
        for r in star_radii:
            # 8 directions: 4 cardinal + 4 diagonal
            ring = [
                (r, 0), (-r, 0), (0, r), (0, -r),
                (r, r), (r, -r), (-r, r), (-r, -r),
            ]
            offsets.extend(ring)
        return np.array(offsets, dtype=np.int64)

    # Fallback dense 5x5
    h = grid_size // 2
    offsets = []
    for r in range(-h, h + 1):
        for c in range(-h, h + 1):
            offsets.append((r, c))
    return np.array(offsets, dtype=np.int64)


class ConcentricStarTopology:
    """
    Class helper for concentric star 25-probe layout.
    """
    def __init__(self, star_radii: Tuple[int, int, int] = (1, 3, 7)):
        self.star_radii = star_radii
        self.offsets = get_spatial_topology_offsets("concentric_star", star_radii=star_radii)
        # Create standard 5x5 mapping matrix (identity indices 0..24 shaped as [5, 5])
        self.grid_indices = np.arange(25, dtype=np.int64).reshape(5, 5)
