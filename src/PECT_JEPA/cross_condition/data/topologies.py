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


def get_contiguous_star_masks(
    star_radii: Tuple[int, int, int] = (1, 3, 7),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds 16 spatially contiguous, strictly non-enclosing (9 target, 16 context)
    partitions on the 25-probe concentric star topology:
      - Neither context points lie inside the target region nor target points
        lie surrounded inside the context region (both regions always touch the
        outer boundary r3 and form single connected angular wedges).
      - Family A (8 orientations, k=0..7):
          Target (9 probes): 3 adjacent CCW rays across all 3 rings (r1, r2, r3).
          Context (16 probes): Center (0,0) + 5 complementary adjacent CCW rays across (r1, r2, r3).
      - Family B (8 orientations, k=0..7):
          Target (9 probes): Center (0,0) + 2 adjacent CCW rays across (r1, r2, r3)
                             + 1 adjacent CCW ray on inner/middle rings (r1, r2).
          Context (16 probes): Outer r3 probe of the 3rd ray + 5 complementary adjacent CCW rays.
    Returns:
        target_bank:  [16, 9] int64 array of target probe indices in 0..24.
        context_bank: [16, 16] int64 array of context probe indices in 0..24.
    """
    # Counter-clockwise order of the 8 local ray indices within each ring of get_spatial_topology_offsets:
    # 0:(r,0)[0 deg], 4:(r,r)[45 deg], 2:(0,r)[90 deg], 6:(-r,r)[135 deg],
    # 1:(-r,0)[180 deg], 7:(-r,-r)[225 deg], 3:(0,-r)[270 deg], 5:(r,-r)[315 deg]
    ccw_rays = [0, 4, 2, 6, 1, 7, 3, 5]

    def ray_probes(ccw_idx: int) -> Tuple[int, int, int]:
        d = ccw_rays[ccw_idx % 8]
        return (1 + d, 9 + d, 17 + d)  # (r1, r2, r3)

    target_list = []
    context_list = []
    all_set = set(range(25))

    # Family A: 8 half-plane outer wedges (Center 0 in Context)
    for k in range(8):
        tgt = []
        for step in range(3):
            tgt.extend(ray_probes(k + step))
        ctx = [0]
        for step in range(3, 8):
            ctx.extend(ray_probes(k + step))
        assert len(tgt) == 9 and len(ctx) == 16 and (set(tgt) | set(ctx)) == all_set
        target_list.append(tgt)
        context_list.append(ctx)

    # Family B: 8 apex-inclusive wedges (Center 0 in Target, both sets reach outer boundary r3)
    for k in range(8):
        r_m1 = ray_probes(k)
        r_m2 = ray_probes(k + 1)
        r_m3 = ray_probes(k + 2)
        tgt = [0, r_m1[0], r_m1[1], r_m1[2], r_m2[0], r_m2[1], r_m2[2], r_m3[0], r_m3[1]]
        ctx = [r_m3[2]]
        for step in range(3, 8):
            ctx.extend(ray_probes(k + step))
        assert len(tgt) == 9 and len(ctx) == 16 and (set(tgt) | set(ctx)) == all_set
        target_list.append(tgt)
        context_list.append(ctx)

    return np.array(target_list, dtype=np.int64), np.array(context_list, dtype=np.int64)

