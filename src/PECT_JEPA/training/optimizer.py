"""
Optimizer and Learning Rate Scheduler setup for PECT-JEPA.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.05
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer excluding LayerNorm and bias terms from weight decay.
    Target encoder parameters are excluded from optimization.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)
    return optimizer


class WarmupCosineLRScheduler:
    """Cosine learning rate scheduler with linear warmup."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 3e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 100,
        total_steps: int = 1000
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)

    def step(self, step: int):
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
