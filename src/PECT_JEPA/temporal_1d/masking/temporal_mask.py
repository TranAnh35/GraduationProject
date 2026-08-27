"""
Multi-strategy physics-informed 1D masker for PECT-JEPA (Stage A3).

Strategies (all use a constant N_ctx = num_visible, N_tgt = num_patches - num_visible
so batch tensors stay uniform):
  - 'late_decay'    : context = first N_ctx patches (us-head), target = ms-tail.
                      Physics: predict late-time diffusion state from early-time response.
  - 'head_from_tail': context = last N_ctx patches, target = head. Reverse-causal:
                      cannot be solved by monotone-decay extrapolation, forces the
                      representation to encode surface / stimulus information.
  - 'random_patch'  : random N_ctx visible patches. Forces local temporal dynamics.

`Dynamic1DBlockMasker` is kept as a backward-compatible alias.
"""

import random
from typing import Dict, Optional, Tuple
import numpy as np
import torch

VALID_STRATEGIES = ("late_decay", "random_patch", "head_from_tail")


class MultiStrategy1DMasker:
    """
    Mixed-strategy 1D temporal masker with per-sample strategy sampling.

    Returns:
        context_indices [B, N_ctx], target_indices [B, N_tgt],
        mask_1d [B, num_patches] (True = target), strategy_ids [B].
    """

    def __init__(
        self,
        num_patches: int = 16,
        strategy_probs: Optional[Dict[str, float]] = None,
        num_visible: int = 5
    ):
        probs = dict(strategy_probs) if strategy_probs else {
            "late_decay": 0.4, "random_patch": 0.3, "head_from_tail": 0.3
        }
        for s in probs:
            if s not in VALID_STRATEGIES:
                raise ValueError(f"Unknown masking strategy: {s}")
        total = sum(probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"strategy_probs must sum to 1.0, got {total}")
        self.num_patches = num_patches
        self.num_visible = num_visible
        if not (0 < num_visible < num_patches):
            raise ValueError("num_visible must be in (0, num_patches)")
        self.num_target = num_patches - num_visible
        self.strategies = list(probs.keys())
        self.probs = np.array([probs[s] for s in self.strategies], dtype=np.float64)
        self.probs /= self.probs.sum()

    def _sample_one(self, strategy: str, rng: random.Random) -> Tuple[list, list]:
        N, V = self.num_patches, self.num_visible
        if strategy == "late_decay":
            ctx = list(range(V))
            tgt = list(range(V, N))
        elif strategy == "head_from_tail":
            ctx = list(range(N - V, N))
            tgt = list(range(0, N - V))
        else:  # random_patch
            tgt = sorted(rng.sample(range(N), self.num_target))
            ctx = [i for i in range(N) if i not in tgt]
        return ctx, tgt

    def sample_mask(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size: B
            device: target device
            seed: optional seed for reproducibility

        Returns:
            context_indices: [B, N_ctx] long
            target_indices:  [B, N_tgt] long
            mask_1d:         [B, num_patches] bool (True = target)
            strategy_ids:    [B] long (index into self.strategies)
        """
        rng = random.Random(seed) if seed is not None else random

        ctx_all, tgt_all, mask_all, sid_all = [], [], [], []
        chosen = rng.choices(self.strategies, weights=self.probs.tolist(), k=batch_size)

        for b, strategy in enumerate(chosen):
            # per-sample independent randomness: derive deterministic sub-seed if seeded
            sub_rng = random.Random(seed * 100003 + b) if seed is not None else rng
            ctx, tgt = self._sample_one(strategy, sub_rng)
            m = np.zeros(self.num_patches, dtype=bool)
            m[tgt] = True
            ctx_all.append(ctx)
            tgt_all.append(tgt)
            mask_all.append(m)
            sid_all.append(self.strategies.index(strategy))

        return (
            torch.tensor(ctx_all, dtype=torch.long, device=device),
            torch.tensor(tgt_all, dtype=torch.long, device=device),
            torch.from_numpy(np.stack(mask_all, axis=0)).to(device),
            torch.tensor(sid_all, dtype=torch.long, device=device),
        )


# Backward-compatible alias (legacy name)
Dynamic1DBlockMasker = MultiStrategy1DMasker
