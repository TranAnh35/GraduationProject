"""
Dynamic Spatio-Temporal Block Masking for PECT-JEPA (Module D, Section 9).
Dynamically samples 3D spatio-temporal blocks (B_h x B_w x B_t) in the token grid.
Guarantees exact target mask ratio r in [min_mask_ratio, max_mask_ratio],
ensures visible context remains, and provides strict disjointness.
"""

import math
import random
import numpy as np
import torch
from typing import Tuple, List, Optional


class DynamicSpatioTemporalBlockMasker:
    """
    Module D: Dynamic Spatio-Temporal Block Masker.
    Generates dynamic 3D mask blocks of size (B_h, B_w, B_t) in the token grid (H_t, W_t, K_t).
    Enforces target token budget based on min_mask_ratio and max_mask_ratio.
    """
    def __init__(
        self,
        spatial_block_h: int = 4,
        spatial_block_w: int = 4,
        temporal_block_t: int = 1,
        min_mask_ratio: float = 0.20,
        max_mask_ratio: float = 0.50,
        num_blocks: Optional[int] = None
    ):
        self.B_h = spatial_block_h
        self.B_w = spatial_block_w
        self.B_t = temporal_block_t
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def sample_mask(
        self,
        batch_size: int,
        grid_shape: Tuple[int, int, int],
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample dynamic 3D spatio-temporal masks for a batch.

        Args:
            batch_size: B
            grid_shape: (H_t, W_t, K_t)
            device: torch device
            seed: optional random seed for reproducibility

        Returns:
            context_indices: [B, N_ctx] tensor of token indices for visible context
            target_indices: [B, N_tgt] tensor of token indices for masked targets
            mask_3d: [B, H_t, W_t, K_t] boolean mask (True = masked / target, False = context)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        H_t, W_t, K_t = grid_shape
        total_tokens = H_t * W_t * K_t

        # 1. Sample target mask ratio for the batch
        target_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        target_budget = int(round(total_tokens * target_ratio))
        # Ensure at least 1 block and at least some visible context
        target_budget = max(self.B_h * self.B_w * self.B_t, min(target_budget, total_tokens - 1))

        all_context_indices = []
        all_target_indices = []
        masks_3d = []

        for b in range(batch_size):
            mask_grid = np.zeros((H_t, W_t, K_t), dtype=bool)

            # Sample 3D blocks until target budget is covered
            attempts = 0
            while np.sum(mask_grid) < target_budget and attempts < 200:
                attempts += 1

                # Dynamic block dimensions with slight variability around (B_h, B_w, B_t)
                b_h = min(self.B_h, H_t)
                b_w = min(self.B_w, W_t)
                b_t = min(self.B_t, K_t)

                h_max = max(0, H_t - b_h)
                w_max = max(0, W_t - b_w)
                t_max = max(0, K_t - b_t)

                h_start = random.randint(0, h_max) if h_max > 0 else 0
                w_start = random.randint(0, w_max) if w_max > 0 else 0
                t_start = random.randint(0, t_max) if t_max > 0 else 0

                mask_grid[
                    h_start : h_start + b_h,
                    w_start : w_start + b_w,
                    t_start : t_start + b_t
                ] = True

            # Get flat indices of masked tokens
            target_flat = np.flatnonzero(mask_grid)

            # Exact budget alignment: randomly sample exactly target_budget tokens from target_flat
            if len(target_flat) >= target_budget:
                chosen_target = np.random.choice(target_flat, size=target_budget, replace=False)
            else:
                # If short, supplement randomly from unmasked
                unmasked = np.flatnonzero(~mask_grid)
                needed = target_budget - len(target_flat)
                supp = np.random.choice(unmasked, size=needed, replace=False)
                chosen_target = np.concatenate([target_flat, supp])

            # Reconstruct exact boolean mask
            exact_mask = np.zeros(total_tokens, dtype=bool)
            exact_mask[chosen_target] = True
            mask_3d_exact = exact_mask.reshape((H_t, W_t, K_t))

            # Disjoint visible context indices
            chosen_context = np.flatnonzero(~exact_mask)

            # Sort indices for deterministic ordering within each sample
            chosen_target = np.sort(chosen_target)
            chosen_context = np.sort(chosen_context)

            masks_3d.append(mask_3d_exact)
            all_target_indices.append(chosen_target)
            all_context_indices.append(chosen_context)

        # Convert to batch tensors
        context_indices = torch.from_numpy(np.stack(all_context_indices, axis=0)).long().to(device)
        target_indices = torch.from_numpy(np.stack(all_target_indices, axis=0)).long().to(device)
        masks_tensor = torch.from_numpy(np.stack(masks_3d, axis=0)).to(device)

        return context_indices, target_indices, masks_tensor
