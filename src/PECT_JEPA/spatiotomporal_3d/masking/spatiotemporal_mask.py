"""
Dynamic Frame-by-Frame Block Masker for PECT-JEPA v0.2 (Module 2, implement.md).
Implements Frame-by-Frame Dynamic Slicing:
1. Selects K random frames out of T_c frames to be masked (remaining T_c - K frames are 100% visible).
2. At each selected frame t, places 1 solid rectangular spatial block (B_h x B_w) at an independent random coordinate (X_t, Y_t).
3. Guarantees 100% strict block integrity without random sub-sampling or holes.
"""

import random
import numpy as np
import torch
from typing import Tuple, Optional, List


class DynamicSpatioTemporalBlockMasker:
    """
    Module 2: Dynamic Frame-by-Frame Block Masker.
    Generates dynamic rectangular blocks on K randomly selected temporal frames.
    """
    def __init__(
        self,
        spatial_block_h: int = 8,
        spatial_block_w: int = 8,
        num_masked_frames: Optional[int] = 8,
        min_masked_frames: int = 4,
        max_masked_frames: int = 10
    ):
        self.B_h = spatial_block_h
        self.B_w = spatial_block_w
        self.num_masked_frames = num_masked_frames
        self.min_masked_frames = min_masked_frames
        self.max_masked_frames = max_masked_frames

    def sample_mask(
        self,
        batch_size: int,
        grid_shape: Tuple[int, int, int],
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample dynamic Frame-by-Frame 3D block masks for a batch.

        Args:
            batch_size: B
            grid_shape: (H_t, W_t, T_c) e.g. (37, 37, 16)
            device: torch device
            seed: optional local random seed

        Returns:
            context_indices: [B, N_ctx] tensor of 1D flat indices for visible context tokens
            target_indices: [B, N_tgt] tensor of 1D flat indices for masked target tokens
            mask_3d: [B, H_t, W_t, T_c] boolean mask (True = Target, False = Context)
        """
        rng = random.Random(seed) if seed is not None else random

        H_t, W_t, T_c = grid_shape
        total_tokens = H_t * W_t * T_c

        b_h = min(self.B_h, H_t)
        b_w = min(self.B_w, W_t)

        # 1. Determine number of frames K to mask for this batch
        if self.num_masked_frames is not None:
            K_frames = min(self.num_masked_frames, T_c - 1)
        else:
            K_frames = rng.randint(self.min_masked_frames, min(self.max_masked_frames, T_c - 1))
        K_frames = max(1, K_frames)

        all_context_indices = []
        all_target_indices = []
        masks_3d = []

        h_max = max(0, H_t - b_h)
        w_max = max(0, W_t - b_w)

        for b in range(batch_size):
            mask_grid = np.zeros((H_t, W_t, T_c), dtype=bool)

            # 2. Select K random frames out of T_c
            chosen_frames = rng.sample(range(T_c), K_frames)

            # 3. On each selected frame t, place 1 independent random block (B_h x B_w)
            for t in chosen_frames:
                x_t = rng.randint(0, h_max) if h_max > 0 else 0
                y_t = rng.randint(0, w_max) if w_max > 0 else 0

                # Mark exact solid rectangular block: [x_t : x_t + b_h, y_t : y_t + b_w, t]
                mask_grid[x_t : x_t + b_h, y_t : y_t + b_w, t] = True

            # 4. Extract 1D flat indices (Strict Block Integrity)
            target_flat = np.flatnonzero(mask_grid)
            context_flat = np.flatnonzero(~mask_grid)

            masks_3d.append(mask_grid)
            all_target_indices.append(target_flat)
            all_context_indices.append(context_flat)

        # Batch indexing tensors (equal length K_frames * b_h * b_w across batch)
        context_indices = torch.from_numpy(np.stack(all_context_indices, axis=0)).long().to(device)
        target_indices = torch.from_numpy(np.stack(all_target_indices, axis=0)).long().to(device)
        masks_tensor = torch.from_numpy(np.stack(masks_3d, axis=0)).to(device)

        return context_indices, target_indices, masks_tensor
