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
    crop_border: int = 15               # Crop outer boundary pixels to remove air/edge effect (15 pixels each side)
    apply_lowpass: bool = True          # Zero-phase Butterworth lowpass filter to eliminate high-frequency EMI
    lowpass_cutoff: float = 2500.0      # Cutoff frequency in Hz (2.5 kHz preserves 99.8% diffusion energy)
    lowpass_order: int = 4              # 4th-order zero-phase filter (filtfilt)
    use_memmap: bool = True
    cache_dir: str = ".cache/pect_5x5_mmap"
    eps: float = 1e-8
    max_files: Optional[int] = None

    # --------------------------------------------------------------- Masking
    masker_type: str = "spatiotemporal_diffusion" # 'spatiotemporal_diffusion' (3D Space+Diffusion) | 'contiguous_cluster' (2D)
    min_masked: int = 10                # Backward compatibility alias
    max_masked: int = 15                # Backward compatibility alias
    num_spatial_cluster: int = 8        # Number of spatial grid points in cluster (x2 tokens = 16 tokens)
    num_cross_diffusion: int = 8        # Number of cross-scale deep masked points (8 tokens) -> Total 24 targets, 26 context

    # ------------------------------------------------------------ Architecture
    tokenizer_type: str = "dual_scale_diffusion" # 'dual_scale_diffusion' (Shallow vs Deep 50 tokens) | 'dual_domain_attention' | 'dual_domain' | 'time_only'
    tokenizer_heads: int = 4            # Number of attention heads for dual-domain fusion
    num_freq_bins: int = 14             # Number of FFT frequency bins (1..14, default 14 = 0-2800 Hz)
    spectral_features: str = "phase_and_mag" # 'phase_and_mag' | 'phase_only'
    phase_snr_tapering: bool = True     # Magnitude-weighted phase tapering to suppress noise floor
    phase_noise_floor: float = 0.05     # Signal magnitude threshold for phase tapering
    embed_dim: int = 64                 # D (64 provides optimal capacity for 1.46M samples/epoch, 4 heads with dk=16)
    pos_embed_type: str = "learnable_2d" # 'learnable_2d' | 'sinusoidal_2d'
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_type: str = "operator_diffusion" # 'operator_diffusion' (Physics Neural Operator with Green's Attention Bias) | 'standard'
    predictor_depth: int = 2
    predictor_heads: int = 4
    diffusion_gamma_init: float = 1.0   # Spatial diffusion attenuation rate gamma for Green's attention bias
    diffusion_beta_init: float = 0.5    # Cross-scale depth barrier beta for Green's attention bias
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ema_momentum: float = 0.990         # Start at 0.990 for fast early target evolution
    ema_momentum_end: float = 0.999      # Capped at 0.999 (never 1.0) to keep target encoder dynamic
    use_momentum_schedule: bool = True

    # ------------------------------------------------------------------- Loss
    loss_type: str = "l1"                # Pure I-JEPA L1 loss (preserves non-vanishing unit gradient)
    liftoff_invar_weight: float = 0.05   # Lift-off invariance loss weight (decouples lift-off from depth)
    phase_align_weight: float = 0.05     # Phase-depth monotonicity loss weight (enforces monotonic depth manifold)
    var_weight: float = 1.0              # VICReg coordinate-wise variance hinge weight (Bardes et al., ICLR 2022)
    cov_weight: float = 1.0              # VICReg covariance decorrelation penalty weight (Bardes et al., ICLR 2022)
    var_gamma: float = 1.0               # VICReg target standard deviation threshold gamma (anchors coordinate scale)
    uniformity_weight: float = 0.0       # Hypersphere Uniformity loss weight (0.0 to prevent artificial repulsion of sound metal)
    uniformity_t: float = 2.0            # Gaussian potential parameter t for hypersphere uniformity
    uniformity_subsample: int = 1024     # Subsample size for stable, memory-efficient pairwise similarity
    norm_floor_weight: float = 0.1       # Norm Floor Barrier weight to prevent zero-vector collapse (||z|| >= target)
    norm_floor_target: float = 1.0       # Minimum expected representation L2 norm target

    # --------------------------------------------------------------- Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    epochs: int = 30                     # 30 epochs ensures sustained representation convergence
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"

    # --------------------------------------------------- Experiment & Logging
    exp_name: str = "pect_jepa_5x5_base"
    log_dir: str = "experiments/5x5"
    save_dir: Optional[str] = None       # If None/default, automatically unified into <log_dir>/<exp_name>/checkpoints
    add_timestamp: bool = False          # False keeps fixed exp_name folder for seamless resume; True appends timestamp
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "PECT_JEPA_5x5"
    wandb_entity: Optional[str] = None
    log_histograms: bool = False
    log_interval: int = 20
    val_interval: int = 1
    diagnostics_interval: int = 1        # Run physics-grounded training diagnostics every N epochs (0 to disable)
    early_stopping_metric: str = "val_loss_pred"  # 'val_loss_pred' | 'val_loss'
    early_stopping_patience: int = 10    # Stop training early if monitored metric fails to improve for N epochs (0 = disabled)
    early_stopping_warmup: int = 0       # Grace period: pause early stopping patience counter during first N epochs (0 = disabled)
    seed: int = 42
    resume: Optional[str] = None         # Checkpoint path or 'latest' / 'auto' / 'best' to resume from

    @property
    def experiment_dir(self) -> str:
        return os.path.join(self.log_dir, self.exp_name)

    @property
    def checkpoint_dir(self) -> str:
        if self.save_dir and self.save_dir != "auto":
            return self.save_dir
        return os.path.join(self.experiment_dir, "checkpoints")

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
