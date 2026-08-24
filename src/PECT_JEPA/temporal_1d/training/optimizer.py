"""
Optimizer and Schedulers for 1D Temporal PECT-JEPA.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any


def build_optimizer_1d(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.05
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with weight decay applied to 2D+ parameters only.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "mask_token" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


class WarmupCosineLRScheduler1D:
    """Cosine learning rate schedule with linear warmup."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)

    def step(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            lr = self.base_lr * (current_step + 1) / max(1, self.warmup_steps)
        else:
            progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class EMAScheduler1D:
    """Cosine schedule for Target Encoder EMA momentum."""
    def __init__(
        self,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
        total_steps: int = 10000,
        use_schedule: bool = True
    ):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_steps = total_steps
        self.use_schedule = use_schedule

    def get_momentum(self, current_step: int) -> float:
        if not self.use_schedule:
            return self.base_momentum
        progress = min(1.0, current_step / max(1, self.total_steps))
        momentum = self.final_momentum - (self.final_momentum - self.base_momentum) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return momentum
