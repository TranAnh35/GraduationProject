"""
Contiguous Cluster Masker for 5x5 Spatial Grid in PECT-JEPA.

Generates random contiguous clusters of 10-15 masked points (40% - 60%)
on a 5x5 spatial grid using random-walk neighbor expansion with
Hole-Filling & Island Elimination:
- Eliminates enclosed interior "donut holes" (context points surrounded by mask).
- Eliminates isolated context points of size 1 (points with 0 context neighbors).
- Guarantees Target is a single connected component.
- Guarantees uniform batch tensor dimensions [B, N_ctx] and [B, N_tgt].
- Eliminates the Laplace interpolation shortcut for Self-Supervised Learning.
"""

import random
from collections import deque
from typing import Tuple, List, Set, Optional
import numpy as np
import torch


class ContiguousClusterMasker5x5:
    """
    Random-walk contiguous cluster masker for 5x5 grid (25 tokens)
    equipped with Hole-Filling and Island Pruning.
    """

    def __init__(self, min_masked: int = 10, max_masked: int = 15, grid_size: int = 5):
        self.min_masked = min_masked
        self.max_masked = max_masked
        self.grid_size = grid_size
        self.total_tokens = grid_size * grid_size  # 25
        self.all_pts: Set[Tuple[int, int]] = set((i, j) for i in range(grid_size) for j in range(grid_size))

        if not (0 < min_masked <= max_masked < self.total_tokens):
            raise ValueError(f"Invalid mask bounds: {min_masked} - {max_masked} for grid size {grid_size}")

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def _is_connected(self, pts_set: Set[Tuple[int, int]]) -> bool:
        """Check if a set of 2D grid coordinates forms a single connected component."""
        if not pts_set:
            return True
        start = next(iter(pts_set))
        visited = {start}
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nb in self._get_neighbors(*curr):
                if nb in pts_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == len(pts_set)

    def _find_components(self, pts_set: Set[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Find all 4-connected components in a set of grid coordinates."""
        visited = set()
        components = []
        for p in pts_set:
            if p not in visited:
                comp = [p]
                queue = deque([p])
                visited.add(p)
                while queue:
                    curr = queue.popleft()
                    for nb in self._get_neighbors(*curr):
                        if nb in pts_set and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                            comp.append(nb)
                components.append(comp)
        return components

    def _touches_boundary(self, p: Tuple[int, int]) -> bool:
        """Check if point is on the 5x5 grid outer boundary."""
        return p[0] == 0 or p[0] == self.grid_size - 1 or p[1] == 0 or p[1] == self.grid_size - 1

    def _is_valid_mask(self, tgt_set: Set[Tuple[int, int]]) -> bool:
        """
        Validate that:
        1. Target cluster is a single connected component.
        2. Context set has NO isolated points (every context point has >= 1 context neighbor).
        3. Context set has NO interior donut holes (every context component touches the boundary).
        """
        if not self._is_connected(tgt_set):
            return False
        ctx_set = self.all_pts - tgt_set
        for p in ctx_set:
            if not any(nb in ctx_set for nb in self._get_neighbors(*p)):
                return False
        comps = self._find_components(ctx_set)
        for c in comps:
            if not any(self._touches_boundary(p) for p in c):
                return False
        return True

    def sample_one(self, rng: random.Random, num_mask: int, max_attempts: int = 50) -> Tuple[List[int], List[int]]:
        """
        Sample 1 contiguous cluster of target points with hole-filling.
        Ensures:
        1. Target cluster is contiguous (1 connected component).
        2. No context points are enclosed/surrounded by the mask (no donut holes).
        3. No isolated context points of size 1 (0 context neighbors).
        4. Target size strictly equals `num_mask`.

        Returns:
            (context_indices, target_indices) as lists of 1D token indices [0 .. 24].
        """
        for _ in range(max_attempts):
            start_x = rng.randint(0, self.grid_size - 1)
            start_y = rng.randint(0, self.grid_size - 1)
            masked_set: Set[Tuple[int, int]] = {(start_x, start_y)}
            frontier: List[Tuple[int, int]] = self._get_neighbors(start_x, start_y)

            # 1. Random walk expansion
            while len(masked_set) < num_mask and frontier:
                idx = rng.randint(0, len(frontier) - 1)
                cx, cy = frontier.pop(idx)
                if (cx, cy) not in masked_set:
                    masked_set.add((cx, cy))
                    for nx, ny in self._get_neighbors(cx, cy):
                        if (nx, ny) not in masked_set and (nx, ny) not in frontier:
                            frontier.append((nx, ny))

            if len(masked_set) < num_mask:
                continue

            # 2. Hole-Filling:
            # Fill any enclosed interior holes or isolated context points
            ctx_set = self.all_pts - masked_set
            comps = self._find_components(ctx_set)
            for comp in comps:
                touches_b = any(self._touches_boundary(p) for p in comp)
                if (not touches_b) or (len(comp) == 1):
                    for p in comp:
                        masked_set.add(p)

            # 3. Pruning: Prune back down to exact num_mask while preserving validity
            stuck = False
            while len(masked_set) > num_mask:
                candidates = [p for p in masked_set if self._is_valid_mask(masked_set - {p})]
                if not candidates:
                    stuck = True
                    break
                masked_set.remove(rng.choice(candidates))

            if not stuck and len(masked_set) == num_mask and self._is_valid_mask(masked_set):
                tgt = sorted([x * self.grid_size + y for (x, y) in masked_set])
                ctx = [i for i in range(self.total_tokens) if i not in tgt]
                return ctx, tgt

        # Deterministic fallback slice if attempts exhausted (never leaves isolated points)
        fallback: Set[Tuple[int, int]] = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                fallback.add((i, j))
                if len(fallback) == num_mask:
                    tgt = sorted([x * self.grid_size + y for (x, y) in fallback])
                    ctx = [x for x in range(self.total_tokens) if x not in tgt]
                    return ctx, tgt

        tgt = sorted([x * self.grid_size + y for (x, y) in masked_set])
        ctx = [i for i in range(self.total_tokens) if i not in tgt]
        return ctx, tgt

    def sample_mask(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample contiguous cluster masks for a batch.
        Uses a constant `num_mask` for all items in the batch so output tensors are rectangular.

        Returns:
            context_indices: [B, N_ctx] long tensor
            target_indices:  [B, N_tgt] long tensor
            mask_bool:       [B, 25] boolean tensor (True = target/masked)
        """
        rng = random.Random(seed) if seed is not None else random
        num_mask = rng.randint(self.min_masked, self.max_masked)

        ctx_all = []
        tgt_all = []
        mask_grid_all = np.zeros((batch_size, self.total_tokens), dtype=bool)

        for b in range(batch_size):
            sub_rng = random.Random(seed * 10007 + b) if seed is not None else rng
            ctx, tgt = self.sample_one(sub_rng, num_mask)
            ctx_all.append(ctx)
            tgt_all.append(tgt)
            mask_grid_all[b, tgt] = True

        context_indices = torch.tensor(ctx_all, dtype=torch.long, device=device)
        target_indices = torch.tensor(tgt_all, dtype=torch.long, device=device)
        mask_bool = torch.from_numpy(mask_grid_all).to(device)
        return context_indices, target_indices, mask_bool
