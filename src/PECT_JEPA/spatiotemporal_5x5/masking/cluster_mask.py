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

    def __init__(
        self,
        min_masked: int = 10,
        max_masked: int = 15,
        grid_size: int = 5,
        use_mask_bank: bool = True,
        bank_size: int = 2048,
    ):
        self.min_masked = min_masked
        self.max_masked = max_masked
        self.grid_size = grid_size
        self.total_tokens = grid_size * grid_size  # 25
        self.all_pts: Set[Tuple[int, int]] = set((i, j) for i in range(grid_size) for j in range(grid_size))
        self.use_mask_bank = use_mask_bank
        self.bank_size = bank_size
        self._banks: dict = {}

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

    def _get_or_create_bank(self, num_mask: int, device: torch.device):
        dev_key = (str(device), num_mask)
        if dev_key not in self._banks:
            rng = random.Random(42 + num_mask * 1009)
            ctx_all = []
            tgt_all = []
            mask_grid_all = np.zeros((self.bank_size, self.total_tokens), dtype=bool)
            for _ in range(self.bank_size):
                ctx, tgt = self.sample_one(rng, num_mask)
                ctx_all.append(ctx)
                tgt_all.append(tgt)
                mask_grid_all[len(ctx_all) - 1, tgt] = True

            ctx_t = torch.tensor(ctx_all, dtype=torch.long, device=device)
            tgt_t = torch.tensor(tgt_all, dtype=torch.long, device=device)
            bool_t = torch.from_numpy(mask_grid_all).to(device)
            self._banks[dev_key] = (ctx_t, tgt_t, bool_t)
        return self._banks[dev_key]

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
        if seed is None and self.use_mask_bank:
            num_mask = random.randint(self.min_masked, self.max_masked)
            ctx_bank, tgt_bank, bool_bank = self._get_or_create_bank(num_mask, device)
            idx = torch.randint(0, self.bank_size, (batch_size,), device=device)
            return ctx_bank[idx], tgt_bank[idx], bool_bank[idx]

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


class SpatiotemporalDiffusionMasker5x5:
    """
    3D Spatiotemporal-Diffusion Masker for 5x5 PECT-JEPA (50 tokens: 25 spatial x 2 scales).

    Combines:
      1. Spatial Cluster Masking:
         Selects a contiguous cluster of `num_spatial_cluster` points (default 8) on the 5x5 grid
         using random-walk neighbor expansion.
         For these spatial points, BOTH Shallow and Deep tokens are masked (8 * 2 = 16 tokens).
         Forces the Predictor to learn lateral eddy current deflection around defects.
      2. Cross-Diffusion Masking:
         From the remaining (25 - num_spatial_cluster) points, selects `num_cross_diffusion` points (default 8).
         For these points, ONLY the Deep token is masked, keeping the Shallow (surface) token visible!
         Forces the Predictor to learn the electromagnetic diffusion Green's function from surface to depth.
         Breaks the spatial copying shortcut on sound metal (since shallow != deep).

    Total tokens: 50.
    Total targets: num_spatial_cluster * 2 + num_cross_diffusion = 16 + 8 = 24 tokens (48%).
    Total context: 50 - 24 = 26 tokens (52%).
    Outputs rectangular, non-ragged tensors: [B, 26] and [B, 24].
    """

    def __init__(
        self,
        grid_size: int = 5,
        num_spatial_cluster: int = 8,
        num_cross_diffusion: int = 8,
        use_mask_bank: bool = True,
        bank_size: int = 2048,
    ):
        self.grid_size = grid_size
        self.total_spatial = grid_size * grid_size  # 25
        self.total_tokens = self.total_spatial * 2  # 50
        self.num_spatial_cluster = min(num_spatial_cluster, self.total_spatial - 2)
        self.num_cross_diffusion = min(num_cross_diffusion, self.total_spatial - self.num_spatial_cluster)
        self.num_tgt = self.num_spatial_cluster * 2 + self.num_cross_diffusion
        self.num_ctx = self.total_tokens - self.num_tgt
        self.use_mask_bank = use_mask_bank
        self.bank_size = bank_size
        self._banks: dict = {}

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        nbs = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                nbs.append((nx, ny))
        return nbs

    def sample_one(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        all_pts = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        sx = rng.randint(0, self.grid_size - 1)
        sy = rng.randint(0, self.grid_size - 1)
        cluster: Set[Tuple[int, int]] = {(sx, sy)}
        frontier: List[Tuple[int, int]] = self._get_neighbors(sx, sy)

        # 1. Random walk cluster expansion
        while len(cluster) < self.num_spatial_cluster and frontier:
            idx = rng.randint(0, len(frontier) - 1)
            cx, cy = frontier.pop(idx)
            if (cx, cy) not in cluster:
                cluster.add((cx, cy))
                for nb in self._get_neighbors(cx, cy):
                    if nb not in cluster and nb not in frontier:
                        frontier.append(nb)

        # If frontier exhausted early, fill from remaining
        remaining = [p for p in all_pts if p not in cluster]
        while len(cluster) < self.num_spatial_cluster and remaining:
            p = remaining.pop(rng.randint(0, len(remaining) - 1))
            cluster.add(p)

        # Target tokens from spatial cluster (both shallow and deep)
        tgt_tokens: List[int] = []
        for (x, y) in cluster:
            sp_idx = x * self.grid_size + y
            tgt_tokens.append(sp_idx * 2)      # shallow token
            tgt_tokens.append(sp_idx * 2 + 1)  # deep token

        # 2. Cross-diffusion masking on remaining visible spatial points
        remaining_pts = [p for p in all_pts if p not in cluster]
        cross_pts = rng.sample(remaining_pts, min(self.num_cross_diffusion, len(remaining_pts)))
        for (x, y) in cross_pts:
            sp_idx = x * self.grid_size + y
            tgt_tokens.append(sp_idx * 2 + 1)  # mask deep token only

        tgt_set = set(tgt_tokens)
        ctx_tokens = [i for i in range(self.total_tokens) if i not in tgt_set]

        return sorted(ctx_tokens), sorted(tgt_tokens)

    def _get_or_create_bank(self, device: torch.device):
        dev_key = str(device)
        if dev_key not in self._banks:
            rng = random.Random(42 + 50)
            ctx_all = []
            tgt_all = []
            mask_grid_all = np.zeros((self.bank_size, self.total_tokens), dtype=bool)
            for _ in range(self.bank_size):
                ctx, tgt = self.sample_one(rng)
                ctx_all.append(ctx)
                tgt_all.append(tgt)
                mask_grid_all[len(ctx_all) - 1, tgt] = True

            ctx_t = torch.tensor(ctx_all, dtype=torch.long, device=device)
            tgt_t = torch.tensor(tgt_all, dtype=torch.long, device=device)
            bool_t = torch.from_numpy(mask_grid_all).to(device)
            self._banks[dev_key] = (ctx_t, tgt_t, bool_t)
        return self._banks[dev_key]

    def sample_mask(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if seed is None and self.use_mask_bank:
            ctx_bank, tgt_bank, bool_bank = self._get_or_create_bank(device)
            idx = torch.randint(0, self.bank_size, (batch_size,), device=device)
            return ctx_bank[idx], tgt_bank[idx], bool_bank[idx]

        ctx_all = []
        tgt_all = []
        mask_grid_all = np.zeros((batch_size, self.total_tokens), dtype=bool)

        for b in range(batch_size):
            sub_rng = random.Random(seed * 10007 + b) if seed is not None else random.Random()
            ctx, tgt = self.sample_one(sub_rng)
            ctx_all.append(ctx)
            tgt_all.append(tgt)
            mask_grid_all[b, tgt] = True

        context_indices = torch.tensor(ctx_all, dtype=torch.long, device=device)
        target_indices = torch.tensor(tgt_all, dtype=torch.long, device=device)
        mask_bool = torch.from_numpy(mask_grid_all).to(device)
        return context_indices, target_indices, mask_bool


class ComplementarySpatiotemporalMasker5x5:
    """
    Complementary Spatiotemporal Masker (CST-Masker) for 5x5 PECT-JEPA.
    Operates on 100 spatiotemporal tokens (25 spatial probe points * 4 temporal diffusion stages).

    Principle (Resolving the Spatial Copying Shortcut on Sound Metal):
      1. Spatial Cluster Partition:
         Selects a contiguous cluster of `num_spatial_cluster` points (default 8) on the 5x5 grid
         using random-walk neighbor expansion with hole-filling.
         Target spatial points: S_tgt (8 points)
         Context spatial points: S_ctx (17 points)
      2. Asymmetric / Complementary Temporal Masking:
         At Context points S_ctx: LATE temporal diffusion tokens (tau in {2, 3}) are ALSO MASKED!
         Context Encoder only observes EARLY excitation tokens (tau in {0, 1}) -> 17 * 2 = 34 tokens.
         Target Encoder represents LATE diffusion tokens (tau in {2, 3}) at S_tgt -> 8 * 2 = 16 tokens.

      Guarantees:
        - Late diffusion tokens tau in {2, 3} are completely absent from all context locations S_ctx.
        - Predictor cannot copy horizontally from adjacent sound metal because NO neighbor has late tokens.
        - Predictor cannot do 1D temporal extrapolation at the same spatial point because S_tgt not in S_ctx.
        - Forces predictor to learn the 3D Green's function propagator G(Delta s, Delta tau).
    """

    def __init__(
        self,
        grid_size: int = 5,
        num_temporal_stages: int = 4,
        num_spatial_cluster: int = 8,
        mode: str = "causal",
        use_mask_bank: bool = True,
        bank_size: int = 2048,
    ):
        self.grid_size = grid_size
        self.total_spatial = grid_size * grid_size  # 25
        self.num_temporal_stages = num_temporal_stages  # 4
        self.total_tokens = self.total_spatial * num_temporal_stages  # 100
        self.num_spatial_cluster = min(num_spatial_cluster, self.total_spatial - 2)
        self.mode = mode
        self.use_mask_bank = use_mask_bank
        self.bank_size = bank_size
        self._banks: dict = {}

        # Base spatial cluster generator
        self.spatial_masker = ContiguousClusterMasker5x5(
            min_masked=self.num_spatial_cluster,
            max_masked=self.num_spatial_cluster,
            grid_size=grid_size,
            use_mask_bank=use_mask_bank,
            bank_size=bank_size,
        )

        # Token counts
        self.num_sp_tgt = self.num_spatial_cluster  # 8
        self.num_sp_ctx = self.total_spatial - self.num_sp_tgt  # 17
        self.num_temporal_ctx = num_temporal_stages // 2  # 2
        self.num_temporal_tgt = num_temporal_stages // 2  # 2

        self.num_ctx = self.num_sp_ctx * self.num_temporal_ctx  # 17 * 2 = 34
        self.num_tgt = self.num_sp_tgt * self.num_temporal_tgt  # 8 * 2 = 16

    def sample_one(self, rng: random.Random) -> Tuple[List[int], List[int]]:
        # 1. Sample contiguous spatial cluster for target
        sp_ctx_list, sp_tgt_list = self.spatial_masker.sample_one(rng, self.num_spatial_cluster)

        # 2. Determine temporal partitioning
        if self.mode == "random":
            # Random complementary 2 vs 2 stages
            all_stages = list(range(self.num_temporal_stages))
            rng.shuffle(all_stages)
            t_ctx = sorted(all_stages[:self.num_temporal_ctx])
            t_tgt = sorted(all_stages[self.num_temporal_ctx:])
        elif self.mode in ("surface_to_depth", "spectral_diffusion", "spectral"):
            # Surface to deep: observe surface & shallow (2, 3) -> predict deep subsurface (0, 1)
            t_ctx = [2, 3]
            t_tgt = [0, 1]
        else:
            # Causal: early excitation (0, 1) -> late diffusion (2, 3)
            t_ctx = list(range(self.num_temporal_ctx))  # [0, 1]
            t_tgt = list(range(self.num_temporal_ctx, self.num_temporal_stages))  # [2, 3]

        # 3. Context tokens: at each context spatial point, only t_ctx stages
        ctx_tokens = []
        for s in sp_ctx_list:
            for tau in t_ctx:
                ctx_tokens.append(s * self.num_temporal_stages + tau)

        # 4. Target tokens: at each target spatial point, only t_tgt stages
        tgt_tokens = []
        for s in sp_tgt_list:
            for tau in t_tgt:
                tgt_tokens.append(s * self.num_temporal_stages + tau)

        return sorted(ctx_tokens), sorted(tgt_tokens)

    def _get_or_create_bank(self, device: torch.device):
        dev_key = str(device)
        if dev_key not in self._banks:
            rng = random.Random(42 + 100)
            ctx_all = []
            tgt_all = []
            mask_grid_all = np.zeros((self.bank_size, self.total_tokens), dtype=bool)
            for _ in range(self.bank_size):
                ctx, tgt = self.sample_one(rng)
                ctx_all.append(ctx)
                tgt_all.append(tgt)
                mask_grid_all[len(ctx_all) - 1, tgt] = True

            ctx_t = torch.tensor(ctx_all, dtype=torch.long, device=device)
            tgt_t = torch.tensor(tgt_all, dtype=torch.long, device=device)
            bool_t = torch.from_numpy(mask_grid_all).to(device)
            self._banks[dev_key] = (ctx_t, tgt_t, bool_t)
        return self._banks[dev_key]

    def sample_mask(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if seed is None and self.use_mask_bank:
            ctx_bank, tgt_bank, bool_bank = self._get_or_create_bank(device)
            idx = torch.randint(0, self.bank_size, (batch_size,), device=device)
            return ctx_bank[idx], tgt_bank[idx], bool_bank[idx]

        ctx_all = []
        tgt_all = []
        mask_grid_all = np.zeros((batch_size, self.total_tokens), dtype=bool)

        for b in range(batch_size):
            sub_rng = random.Random(seed * 10007 + b) if seed is not None else random.Random()
            ctx, tgt = self.sample_one(sub_rng)
            ctx_all.append(ctx)
            tgt_all.append(tgt)
            mask_grid_all[b, tgt] = True

        context_indices = torch.tensor(ctx_all, dtype=torch.long, device=device)
        target_indices = torch.tensor(tgt_all, dtype=torch.long, device=device)
        mask_bool = torch.from_numpy(mask_grid_all).to(device)
        return context_indices, target_indices, mask_bool


def build_masker_5x5(config):
    """
    Factory function to construct masker based on config and tokenizer type.
    - 25 tokens (continuous_stf, spatial_grid, etc.): ContiguousClusterMasker5x5
    - 50 tokens (dual_scale_diffusion): SpatiotemporalDiffusionMasker5x5
    - 100 tokens (spatiotemporal_patch): ComplementarySpatiotemporalMasker5x5 (CST)
    """
    tokenizer_type = getattr(config, "tokenizer_type", "spatiotemporal_patch")
    masker_type = getattr(config, "masker_type", "auto")
    use_mask_bank = getattr(config, "use_mask_bank", True)
    bank_size = getattr(config, "mask_bank_size", 2048)

    if tokenizer_type in ("continuous_stf", "continuous_filterbank", "spatial_grid", "time_only", "dual_domain_attention", "dual_domain"):
        return ContiguousClusterMasker5x5(
            min_masked=config.min_masked,
            max_masked=config.max_masked,
            grid_size=config.grid_size,
            use_mask_bank=use_mask_bank,
            bank_size=bank_size,
        )
    elif tokenizer_type in ("dual_scale_diffusion", "dual_scale") and masker_type != "contiguous_cluster":
        return SpatiotemporalDiffusionMasker5x5(
            grid_size=config.grid_size,
            num_spatial_cluster=getattr(config, "num_spatial_cluster", 8),
            num_cross_diffusion=getattr(config, "num_cross_diffusion", 8),
            use_mask_bank=use_mask_bank,
            bank_size=bank_size,
        )
    elif tokenizer_type in ("spatio_spectral", "skin_depth"):
        return ComplementarySpatiotemporalMasker5x5(
            grid_size=config.grid_size,
            num_temporal_stages=getattr(config, "num_scales", 4),
            num_spatial_cluster=getattr(config, "num_spatial_cluster", 8),
            mode=getattr(config, "cst_mask_mode", "surface_to_depth"),
            use_mask_bank=use_mask_bank,
            bank_size=bank_size,
        )
    else:
        return ComplementarySpatiotemporalMasker5x5(
            grid_size=config.grid_size,
            num_temporal_stages=getattr(config, "num_temporal_stages", 4),
            num_spatial_cluster=getattr(config, "num_spatial_cluster", 8),
            mode=getattr(config, "cst_mask_mode", "causal"),
            use_mask_bank=use_mask_bank,
            bank_size=bank_size,
        )


