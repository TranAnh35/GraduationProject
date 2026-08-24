"""
Dynamic 1D Temporal Masker for PECT-JEPA (implement.md, Section 4.4).
Supports:
1. 'late_decay': Physics-informed masking. Keeps early-time transient (0..4) as Context,
   masks late-time diffusion decay (5..15) as Target (~70% mask ratio).
2. 'random_patch': Random 1D patch masking of 70% patches.
"""

import random
from typing import Tuple, Optional
import numpy as np
import torch


class Dynamic1DBlockMasker:
    """
    1D Temporal Masker.
    Generates 1D patch masks for sequence of N=16 patches.
    """
    def __init__(
        self,
        num_patches: int = 16,
        mask_strategy: str = "late_decay",
        mask_ratio: float = 0.70,
        num_visible_early: int = 5
    ):
        self.num_patches = num_patches
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self.num_visible_early = num_visible_early
        self.num_target_patches = int(round(num_patches * mask_ratio))

    def sample_mask(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample 1D temporal mask.

        Args:
            batch_size: B
            device: torch device
            seed: optional local seed

        Returns:
            context_indices: [B, N_ctx] tensor of visible context patch indices
            target_indices: [B, N_tgt] tensor of masked target patch indices
            mask_1d: [B, num_patches] boolean tensor (True = Target, False = Context)
        """
        rng = random.Random(seed) if seed is not None else random

        all_context_indices = []
        all_target_indices = []
        masks_1d = []

        for b in range(batch_size):
            if self.mask_strategy == "late_decay":
                # First num_visible_early patches are Context, rest are Target
                ctx = list(range(self.num_visible_early))
                tgt = list(range(self.num_visible_early, self.num_patches))
            else:
                # Randomly sample Target patches
                tgt = sorted(rng.sample(range(self.num_patches), self.num_target_patches))
                ctx = [i for i in range(self.num_patches) if i not in tgt]

            mask_bool = np.zeros(self.num_patches, dtype=bool)
            mask_bool[tgt] = True

            all_context_indices.append(ctx)
            all_target_indices.append(tgt)
            masks_1d.append(mask_bool)

        context_indices = torch.tensor(all_context_indices, dtype=torch.long, device=device)
        target_indices = torch.tensor(all_target_indices, dtype=torch.long, device=device)
        mask_tensor = torch.from_numpy(np.stack(masks_1d, axis=0)).to(device)

        return context_indices, target_indices, mask_tensor
