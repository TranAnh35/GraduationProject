"""
Exponential Moving Average (EMA) momentum scheduler for PECT-JEPA Target Encoder (Section 12).
"""

import math


class EMAScheduler:
    """
    Computes momentum schedule for EMA Target Encoder updates.
    Supports constant momentum and cosine momentum schedules.
    """
    def __init__(
        self,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
        total_steps: int = 1000,
        use_schedule: bool = True
    ):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_steps = max(1, total_steps)
        self.use_schedule = use_schedule

    def get_momentum(self, step: int) -> float:
        if not self.use_schedule:
            return self.base_momentum

        # Cosine ramp from base_momentum to final_momentum
        step = min(step, self.total_steps)
        progress = step / self.total_steps
        momentum = self.final_momentum - (self.final_momentum - self.base_momentum) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(momentum)
