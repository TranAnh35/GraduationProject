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

    # Loss weights
    pred_loss_type: str = "l1"  # Pure L1 loss preserving subtle flaw gradients
    var_weight: float = 1.0     # VICReg variance hinge on Context Encoder
    cov_weight: float = 1.0     # VICReg covariance penalty on Context Encoder
    var_gamma: float = 1.0

    # Training parameters
    batch_size: int = 256
    # samples_per_epoch: if None, automatically calculated based on total pair combinations
    # Default: 400,000 samples/epoch ensures dense exploration of all pairs and coordinates
    samples_per_epoch: Optional[int] = 409600
    epochs: int = 40
    lr: float = 4e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.999
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    resume_checkpoint: Optional[str] = None

    # Data and Paths
    data_dir: str = "data"
    transform_csv: str = "data/ground_truth/raw_tdms_rotate_crop.csv"
    checkpoint_dir: str = "experiments/cross_condition/checkpoints"
    results_dir: str = "experiments/cross_condition/results"
    visualizations_dir: str = "experiments/cross_condition/visualizations"
