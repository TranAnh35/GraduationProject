"""
Configuration definitions for Unified 5x5 Spatiotemporal PECT-JEPA.
"""

from dataclasses import dataclass, asdict, field, fields
from typing import Optional, Dict, Any
import json
import os


@dataclass
class Spatiotemporal5x5Config:
    # ------------------------------------------------------------------ Data
    data_dir: str = "data"
    grid_size: int = 5                  # 5x5 spatial grid (25 scan points)
    time_samples: int = 500             # Raw uniform samples per waveform
    t_total_ms: float = 5.0             # Total physical record duration in ms
    resample_mode: str = "linear"       # 'linear' (128 pts) | 'dual_channel' (256 pts)
    temporal_samples: int = 128         # Resampled length
    log_time_samples: int = 128         # Backward compatibility alias
    t_start_frac: float = 0.02          # Grid start fraction (for log mode)
    num_channels: int = 1               # 1 channel (pure linear waveform)
    in_channels: int = 128              # 1 channel * 128 samples = 128 temporal features
    normalization: str = "global_peak"  # 'global_peak' | 'zscore' | 'peak_early' | 'min_max'
    early_window_frac: float = 0.10
    raster_correction: bool = True
    crop_border: int = 10               # Crop outer boundary pixels to remove air/edge effect (10 pixels each side)
    use_memmap: bool = True
    cache_dir: str = ".cache/pect_5x5_mmap"
    eps: float = 1e-8
    max_files: Optional[int] = None

    # --------------------------------------------------------------- Masking
    min_masked: int = 10                # 40% of 25 tokens
    max_masked: int = 15                # 60% of 25 tokens

    # ------------------------------------------------------------ Architecture
    embed_dim: int = 128                # D
    pos_embed_type: str = "learnable_2d" # 'learnable_2d' | 'sinusoidal_2d'
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_depth: int = 2
    predictor_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    use_momentum_schedule: bool = True

    # ------------------------------------------------------------------- Loss
    loss_type: str = "smooth_l1"         # 'smooth_l1' | 'l1' | 'l2' | 'cosine'
    var_weight: float = 1.0              # VICReg variance hinge weight
    cov_weight: float = 0.5              # VICReg covariance decorrelation weight
    var_gamma: float = 1.0               # Target variance standard deviation

    # --------------------------------------------------------------- Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    epochs: int = 50
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"

    # --------------------------------------------------- Experiment & Logging
    exp_name: str = "pect_jepa_5x5_base"
    log_dir: str = "experiments/5x5"
    save_dir: str = "checkpoints/pect_jepa_5x5"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "PECT_JEPA_5x5"
    wandb_entity: Optional[str] = None
    log_histograms: bool = False
    log_interval: int = 20
    val_interval: int = 1
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spatiotemporal5x5Config":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, file_path: str) -> "Spatiotemporal5x5Config":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_config_5x5() -> Spatiotemporal5x5Config:
    return Spatiotemporal5x5Config()
