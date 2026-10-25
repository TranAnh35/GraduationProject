"""
Configuration for Cross-Condition PECT-JEPA.

Single canonical pipeline (exp70 lineage):
  - Unified 64D latent (no [z_c, z_s] split, no two-stream backbone).
  - Energy-adaptive dual-domain tokenizer: per-probe unit-variance waveform shape
    + patch-consistent polarity alignment + bi-directional spatial energy contrast.
  - Dodd-Deeds spectral condition token c_B (0% DC), AdaLN or cross-attention routing.
  - Contiguous non-enclosing wedge masking on a concentric star topology.

Legacy branches from exp44-exp69 (latent_disentanglement / two_stream_backbone /
diffusion_attention / ref_patch_mode='legacy' / inv_oracle_mask / alternative
tokenizers) were removed; their implementations remain in git commit 131889a.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


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

    # Physical tokenizer (concentric star 25-probe topology)
    spatial_topology: str = "concentric_star"
    star_radii: Tuple[int, int, int] = (2, 4, 7)
    grid_size: int = 5
    temporal_samples: int = 128
    num_harmonics: int = 16
    # Global background energy anchor: appends Delta E_global = log(1+E_probe) -
    # log(1+E_bg_ref) using the scan's 4 defect-free corner patches as the flat
    # background reference. Delta E_local (vs the patch's own probe ring) loses
    # contrast inside wide defects (Rivet_v2, 12-14 mm) because the context ring
    # falls inside the defect; the corner anchor does not.
    global_bg_energy_anchor: bool = False

    # Tokenizer & spatial aggregation architecture options (EXP-72).
    # 'linear' + 'mean' reproduce the exp70/exp71 checkpoint surface exactly
    # (no extra parameters), so legacy checkpoints still load with strict=True.
    tokenizer_arch: str = "linear"        # 'linear' (legacy) | 'multiscale_conv' (EXP-72B)
    spatial_pool_mode: str = "mean"       # 'mean' (legacy) | 'radial' (EXP-72A Green radial-decay + center attention)
    radial_decay_init: float = 2.5        # Initial Green spatial decay alpha in w(r) ~ exp(-alpha * (r/r_max)^2)

    # Masking specification (concentric star topology)
    num_target_probes: int = 9
    num_context_probes: int = 16
    num_target_rays: int = 3              # contiguous rays masked (3 => 9 tgt / 16 ctx)
    continuous_pos_embed: bool = True     # continuous 2D coordinate MLP pos_mlp(dr, dc)
    contiguous_masking: bool = True       # non-enclosing wedge masking (no ctx inside tgt)

    # Self-supervised condition token c_B (Dodd-Deeds spectral signature)
    use_condition_token: bool = True      # False => pure JEPA, no c_B routing (ablation)
    condition_modulation: str = "cross_attention"  # 'cross_attention' | 'adaln'
    relative_target: bool = False         # True: predict Delta H = H_tgt - c_B (legacy exp68)

    # Loss weights
    pred_loss_type: str = "l1"            # 'l1' | 'normalized_l1' (probe-wise, scale-invariant)
    anti_hijack_ratio: float = 0.0        # cap L_inv only at ratio * pred_loss (0 = uncapped); VICReg is never capped (FM22)
    scan_centered_vicreg: bool = True     # center representations per scan before VICReg
    var_weight: float = 0.5               # VICReg variance hinge (std >= var_gamma)
    cov_weight: float = 0.2               # VICReg covariance decorrelation
    var_gamma: float = 1.0

    # Cross-condition invariance L_inv (condition drift OUT of the latent)
    inv_weight: float = 0.0               # 0 = disabled
    inv_start_epoch: int = 1              # curriculum: first epoch with L_inv > 0
    inv_ramp_epochs: int = 0              # cosine ramp length (0 = constant from start)
    scan_centered_inv: bool = True        # subtract per-scan mean before cosine L_inv
    inv_cosine: bool = True               # scale-invariant 1 - cos (kills shrink shortcut)
    # BYOL-style asymmetric predictor: L_inv = cos(inv_head(z_A), stopgrad(z_B)).
    # Without it, a scan-centered cosine L_inv directly pulls the encoder into a
    # shared low-D subspace (FM6: rank 23D -> 1.2D in exp57a).
    inv_predictor: bool = False
    # Unsupervised background gating for L_inv: only the keep_q lowest-|Delta E|
    # tokens (sound metal) are aligned, so defect-edge scattering keeps its
    # condition-specific contrast instead of being averaged away.
    inv_trim_q: float = 0.75

    # Data split protocol (DOWNSTREAM only)
    # 'cross_specimen'  : 16-file Base Domain (Corrosion, Rivet_v1) -> 12 train / 4 val, 65 test
    # 'cross_condition' : 24-file Base Domain (Corrosion, Rivet_v1, Rivet_v2) -> 18 train / 6 val, 57 test
    # JEPA pretraining has no scope flag: it is label-free and always trains on
    # every TDMS file under data_dir. `split_mode` only defines the downstream
    # train/val/test roles (fit / telemetry / score) applied through the frozen
    # encoder.
    split_mode: str = "cross_specimen"
    batch_size: int = 256
    num_partitions: int = 10    # K partitions per macro-cycle
    samples_per_epoch: Optional[int] = None
    grid_stride: int = 4        # 1 = all 72,900 pixels per C-scan; 4 = 4,624 px
    epochs: int = 100
    warmup_epochs: int = 10
    early_stopping_patience: int = 15
    early_stopping_metric: str = "composite"  # 'composite', 'val_auc', 'latent_snr', 'loss'
    early_stopping_min_delta: float = 1e-4
    # Only save best when R_eff >= ratio * max-R_eff-so-far (0 disables): a
    # rank-degenerate val gain must not be counted as an improvement.
    best_reff_floor_ratio: float = 0.5
    eval_val_every: int = 1
    lr: float = 4e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.999
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    resume_checkpoint: Optional[str] = None
    # Divergence guard: stop when val loss rises far above its running best.
    diverge_patience: int = 0     # 0 disables the guard
    diverge_factor: float = 1.5

    # Unified experiment directory paths
    data_dir: str = "data"
    transform_csv: str = "data/ground_truth/raw_tdms_rotate_crop.csv"
    exp_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    eval_dir: Optional[str] = None
    visualizations_dir: Optional[str] = None
