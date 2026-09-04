"""
Configuration definitions for 1D Temporal PECT-JEPA (TS-JEPA).

Stage A customizations (physics-informed, per design blueprint):
  A1. Two-channel input: [normalized waveform, log-envelope], log-time resampled
      to T'=128; shape-preserving 'peak_early' normalization (default).
  A2. Log-time tokenization + physical (log-time) positional embeddings.
  A3. Multi-strategy physics-informed masking (late_decay / random_patch /
      head_from_tail).
  A4. Anti-collapse: normalized L2 + variance hinge + covariance regularization.
  A5. File-balanced batch sampling.

Sized for a single RTX 3050 (4 GB VRAM): embed_dim=128, depth=4, batch=256, AMP on.
"""

from dataclasses import dataclass, asdict, field, fields
from typing import Optional, Dict, Any
import json
import os


@dataclass
class Temporal1DConfig:
    # ------------------------------------------------------------------ Data
    data_dir: str = "data"
    time_samples: int = 500          # Raw uniform samples per waveform (T_raw)
    t_total_ms: float = 5.0         # Total acquisition duration in ms (physical time axis)
    log_time_samples: int = 128      # T' : log-time resampled length per channel
    t_start_frac: float = 0.02       # Log-time grid starts at 2% of the record
    num_channels: int = 2            # Ch0: normalized waveform, Ch1: standardized log-envelope
    normalization: str = "peak_early"  # 'peak_early' | 'zscore' | 'min_max'
    early_window_frac: float = 0.10  # Fraction of the record used as the "early-time" window
    tokenizer_mode: str = "resampled"  # 'resampled' (B1/B2: log-time -> 128) | 'raw' (B0: raw + pad)
    pad_mode: str = "edge"           # B0 only: 'edge' | 'reflect' | 'zero' (pad at record tail)
    raw_padded_length: int = 512     # B0 only: padded length, must be divisible by num_patches
    raster_correction: bool = True   # Rearrange bidirectional raster lines (TDMS)
    use_memmap: bool = True
    cache_dir: str = ".cache/pect_1d_mmap"
    eps: float = 1e-8
    max_files: Optional[int] = None  # Limit number of files (debug / pilot runs)

    # ------------------------------------------------------------- Tokenizer
    num_patches: int = 16            # N tokens along log-time
    embed_dim: int = 128             # D
    pos_embed_type: str = "physical_log_time"  # 'physical_log_time' | 'sinusoidal'
    dropout: float = 0.0

    # ---------------------------------------------------------------- Masking
    # Mixed-strategy physics-informed masking (A3). Probabilities must sum to 1.0.
    strategy_probs: Dict[str, float] = field(default_factory=lambda: {
        "late_decay": 0.4,
        "random_patch": 0.3,
        "head_from_tail": 0.3,
    })
    num_visible: int = 5             # N_ctx per sample (constant across strategies)

    # ------------------------------------------------------------ Architecture
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_depth: int = 2
    predictor_heads: int = 4
    mlp_ratio: float = 4.0
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    use_momentum_schedule: bool = True

    # ------------------------------------------------------------------- Loss
    loss_type: str = "normalized_l2"   # JEPA term: 'normalized_l2' | 'cosine' | 'smooth_l1'
    var_weight: float = 1.0            # VICReg-style variance hinge weight (A4)
    cov_weight: float = 0.04           # Covariance decorrelation weight (A4)
    var_gamma: float = 1.0             # Target std for the variance hinge

    # --------------------------------------------------------------- Training
    batch_size: int = 256             # 4 GB VRAM friendly
    file_balanced: bool = True        # A5: file-balanced sampling
    k_per_file: int = 8               # Samples per file within one batch
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    epochs: int = 50
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"
    # --------------------------------------------------- Experiment & Logging
    exp_name: str = "jepa_1d_base"
    log_dir: str = "experiments/1d"
    save_dir: str = "checkpoints/pect_jepa_1d"
    log_interval: int = 20
    val_interval: int = 1
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "PECT_JEPA_1D"
    wandb_entity: Optional[str] = None
    log_histograms: bool = False
    log_image_interval: int = 5
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Temporal1DConfig":
        # Tolerant loading: ignore keys from older configs (e.g. patch_length, stride,
        # mask_strategy, mask_ratio, num_visible_early) so old checkpoints still load.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, file_path: str) -> "Temporal1DConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_default_config_1d() -> Temporal1DConfig:
    """Returns the default 1D Temporal PECT-JEPA configuration (Stage A)."""
    return Temporal1DConfig()
