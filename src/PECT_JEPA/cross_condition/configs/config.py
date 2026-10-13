"""
Configuration for Cross-Condition PECT-JEPA.
Designed specifically for cross-condition predictive SSL on NDT eddy current inspection.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class CrossConditionConfig:
    # Experiment metadata
    exp_name: str = "cross_condition_jepa"
    add_timestamp: bool = True
    base_exp_dir: str = "experiments/cross_condition/runs"

    # Model architecture
    embed_dim: int = 64
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_depth: int = 2
    predictor_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Physical Tokenizer
    spatial_topology: str = "concentric_star"
    star_radii: Tuple[int, int, int] = (1, 3, 7)
    grid_size: int = 5
    temporal_samples: int = 128
    num_harmonics: int = 16

    # Masking specification (9 core target probes, 16 outer context probes)
    num_target_probes: int = 9
    num_context_probes: int = 16
    use_condition_token: bool = False  # If False, disables c_b condition routing (pure JEPA)
    condition_modulation: str = "none"  # 'query_projection', 'cross_attention', 'adaln', 'additive', 'none'
    tokenizer_type: str = "dual_domain"  # 'dual_domain' (baseline EXP-44), 'energy_adaptive', 'continuous_scale_invariant'
    relative_target: bool = False  # If False, predicts full-field target H_tgt preserving natural target variance

    # Disentangled Dual-Subspace [z_c, z_s] (Carrier vs Scattering)
    latent_disentanglement: bool = False  # If True, splits latent D into [z_c, z_s]
    carrier_dim: int = 32                 # Dimension of carrier/background subspace z_c
    scattering_dim: int = 32              # Dimension of scattering/defect subspace z_s
    carrier_flat_weight: float = 1.0      # Spatial flatness penalty on z_c within C-scan
    scattering_var_weight: float = 1.0    # VICReg variance penalty on z_s (hinge std >= 1.0)
    scattering_cov_weight: float = 1.0    # VICReg covariance decorrelation penalty on z_s (prevents 1D collapse)
    # EXP-49: Cross-condition invariance (L_inv) + cross-subspace lock (L_xcov)
    inv_weight: float = 0.0               # MSE(z_s(x_a), stopgrad(z_s(x_b))): drives condition drift OUT of z_s
    xcov_weight: float = 0.0              # ||XCov(z_c, z_s)||^2: locks the split, forces z_c to pick up the remainder
    downstream_latent_type: str = "scattering"  # 'scattering' (z_s only, drift-free), 'full' ([z_c, z_s]), 'carrier' (z_c)

    # Loss weights
    pred_loss_type: str = "l1"  # Pure L1 loss preserving subtle flaw gradients
    scan_centered_vicreg: bool = True  # If True, centers representations per scan before computing VICReg (forces within-scan variance)
    var_weight: float = 0.5     # Gentle VICReg variance scale anchor (std >= 1.0, prevents 2D collapse)
    cov_weight: float = 0.0     # Covariance decorrelation penalty (0.0 allows natural 6D-12D rank)
    var_gamma: float = 1.0

    # Training parameters
    batch_size: int = 256
    num_partitions: int = 10    # K partitions per macro-cycle (K sub-epochs = 100% full universe pass)
    # samples_per_epoch: if None, computed automatically from total_exhaustive_samples // num_partitions
    samples_per_epoch: Optional[int] = None
    grid_stride: int = 1        # 1 = all 72,900 pixels per C-scan; 2 = stride 2 subsample (18,225 px)
    epochs: int = 100
    warmup_epochs: int = 10     # Number of warmup epochs (defaults to K partitions = 1 full macro-cycle)
    early_stopping_patience: int = 15  # Early stopping patience in epochs post-warmup (0 to disable)
    early_stopping_metric: str = "composite"  # 'composite' (AUC + SNR + Loss), 'val_auc', 'latent_snr', 'loss'
    early_stopping_min_delta: float = 1e-4
    eval_val_every: int = 1     # Evaluate validation metrics every N epochs
    lr: float = 4e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.999
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    resume_checkpoint: Optional[str] = None

    # Unified Experiment Directory Paths (All checkpoints & evaluations inside same folder)
    data_dir: str = "data"
    transform_csv: str = "data/ground_truth/raw_tdms_rotate_crop.csv"
    exp_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    eval_dir: Optional[str] = None
    visualizations_dir: Optional[str] = None
