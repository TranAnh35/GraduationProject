"""
Physics-Grounded Spatial Sampling Topologies for PECT-JEPA.

Supports:
1. 'concentric_star': 25 omnidirectional probes across 3 concentric physical rings
   (Center + Ring 1 at r=1mm + Ring 2 at r=3mm + Ring 3 at r=7mm), spanning 14mm x 14mm
   to match the physical PECT coil footprint (10-18mm) and break spatial smoothing shortcuts.
2. 'dense_5x5': Standard Cartesian 5x5 grid (25 points, 1mm spacing, spanning 4mm x 4mm).
"""

from typing import Tuple
import numpy as np
import torch


def get_spatial_topology_offsets(
    topology: str = "concentric_star",
    star_radii: Tuple[int, int, int] = (1, 3, 7),
    grid_size: int = 5,
) -> np.ndarray:
    """
    Returns array of (row_offset, col_offset) for the 25 spatial probes.
    Shape: [25, 2] in int32.
    Center probe is always at index 0: (0, 0).
    """
    topology = str(topology).lower().strip()

    if topology in ("concentric_star", "star", "octagram", "star_25"):
        r_inner, r_mid, r_outer = star_radii
        offsets = [(0, 0)]  # Index 0: Center inspection point

        # 8 radial ray directions: (dr, dc)
        # N, NE, E, SE, S, SW, W, NW
        dirs = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

        # Ring 1: Inner micro-local core (r_inner, default 1mm)
        for dr, dc in dirs:
            offsets.append((dr * r_inner, dc * r_inner))

        # Ring 2: Flaw boundary transition (r_mid, default 3mm)
        for dr, dc in dirs:
            offsets.append((dr * r_mid, dc * r_mid))

        # Ring 3: Outer coil footprint & sound metal reference (r_outer, default 7mm)
        diag_outer = int(round(r_outer * 0.7071))  # Euclidean distance sqrt(50) = 7.07mm
        for dr, dc in dirs:
            if abs(dr) == 1 and abs(dc) == 1:
                offsets.append((dr * diag_outer, dc * diag_outer))
            else:
                offsets.append((dr * r_outer, dc * r_outer))

        assert len(offsets) == 25, f"Expected 25 star offsets, got {len(offsets)}"
        return np.array(offsets, dtype=np.int32)

    elif topology in ("dense_5x5", "grid_5x5", "standard", "dense"):
        offsets = []
        pad = grid_size // 2  # 2 for 5x5
        for r in range(-pad, pad + 1):
            for c in range(-pad, pad + 1):
                offsets.append((r, c))
        # Ensure center (0, 0) is at index 0 for consistent center readout
        center_idx = pad * grid_size + pad  # index 12
        center = offsets.pop(center_idx)
        offsets.insert(0, center)
        assert len(offsets) == 25, f"Expected 25 grid offsets, got {len(offsets)}"
        return np.array(offsets, dtype=np.int32)

    else:
        raise ValueError(f"Unknown spatial topology: {topology}. Choose 'concentric_star' or 'dense_5x5'.")


def compute_physical_distance_matrix(offsets: np.ndarray, step_mm: float = 1.0) -> torch.Tensor:
    """
    Computes pairwise physical Euclidean distance matrix [25, 25] in millimeters.
    Args:
        offsets: [25, 2] array of (row, col) coordinates.
        step_mm: Physical scanning grid step in mm (default 1.0mm).
    Returns:
        dist_matrix: [25, 25] torch.FloatTensor of Euclidean distances in mm.
    """
    coords_mm = offsets.astype(np.float32) * step_mm  # [25, 2]
    diff = coords_mm[:, np.newaxis, :] - coords_mm[np.newaxis, :, :]  # [25, 25, 2]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # [25, 25]
    return torch.from_numpy(dist).float()
