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

    # Masking specification (concentric star topology)
    num_target_probes: int = 9
    num_context_probes: int = 16
    num_target_rays: int = 3              # Number of contiguous rays masked (3 = 9 target / 16 context [36%]; 5 = 15 target / 10 context [60%])
    continuous_pos_embed: bool = False    # Option A: continuous 2D coordinate MLP pos_mlp(dr, dc) instead of discrete pos_embed table
    contiguous_masking: bool = False      # Contiguous non-enclosing wedge masking (avoids context inside target and vice versa)
    predictor_scattering_only: bool = False  # If True, predictor reads only z_s (blocks carrier z_c shortcut)
    scan_centered_inv: bool = False       # If True, subtracts scan/batch mean before cosine L_inv (kills LN_bias needle bypass)
    patch_energy_norm: bool = False       # If True (or tokenizer_type='patch_energy_adaptive'), normalizes temporal amplitude by shared 25-probe patch std (dim=(-2,-1)) instead of per-probe (dim=-1), preserving within-patch spatial contrast (fixes FM11)
    median_patch_energy: bool = False     # If True, keeps per-probe unit-variance x_norm for waveform shape while normalizing amplitude/spectral anchors (v_pp, energy_t, mag) by robust 25-probe median std (fixes FM11 without FM13 outer-ring coupling)
    clean_phasor_token: bool = False      # If True, uses canonical peak-polarity alignment + 5 diffusion energy arrival quantiles + root-saliency complex phasors (eliminates FM16 stopband +-pi phase-wrap noise and FM18 TMR polarity inversion)
    subspace_norm: bool = False           # If True, normalizes [z_c, z_s] with independent LayerNorm(D_c) and bias-free LayerNorm(D_s, bias=False) instead of shared LayerNorm(D) (eliminates FM17 z_s variance starvation and FM8 DC needle)
    relative_energy_anchor: bool = False  # If True, appends spatial energy ratio log(1 + E_i) - log(1 + E_context) to time branch without outer-ring boundary coupling (fixes FM11 on Rivet_v2 & Sizing without FM13/FM14)
    patch_polarity_align: bool = False    # If True, aligns patch polarity using center probe 0 dominant peak without checkerboard flips or root-saliency spectral starvation (resolves TMR polarity while preserving broadband Dodd-Deeds phase)
    bi_directional_relative_energy: bool = False  # If True, appends both signed Delta E and absolute magnitude |Delta E| to time branch (eliminates Rivet_v2 half-space sign inversion in latent space)
    diffusion_attention: bool = False  # Direction A: Physically-grounded Green's diffusion distance-decay attention bias
    two_stream_backbone: bool = False  # Direction B: Asymmetric two-stream decoupled encoder backbone (carrier z_c vs scattering z_s)
    carrier_coupling_mode: str = "additive"  # 'additive' (legacy), 'multiplicative' (pure scaling), 'none' (fully decoupled)
    grounded_condition_pred: bool = False  # Direction C: Condition transfer function grounding with scale-normalized directional target prediction
    use_condition_token: bool = False  # If False, disables c_b condition routing (pure JEPA)
    relative_condition_operator: bool = False  # If True, condition token is domain transfer operator T(c_A, c_B) preventing shortcut DC leakage
    condition_modulation: str = "none"  # 'query_projection', 'cross_attention', 'adaln', 'pure_film', 'additive', 'none'
    tokenizer_type: str = "dual_domain"  # 'dual_domain' (baseline EXP-44), 'energy_adaptive', 'patch_energy_adaptive', 'waveform_invariant'
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
    # P0-1 curriculum + geometry for L_inv / L_xcov:
    # raw MSE rewards amplitude shrinkage (the exp49 collapse). Two guards:
    # (a) schedule: weight stays 0 before *_start_epoch, then cosine-ramps to
    #     full over *_ramp_epochs (0 = constant from start_epoch, legacy);
    # (b) cosine geometry: 1 - cos(z_s(A), sg(z_s(B))) is scale-invariant, so
    #     shrinking the representation no longer reduces the loss.
    inv_start_epoch: int = 1              # first epoch where L_inv may be > 0
    inv_ramp_epochs: int = 0              # cosine ramp length (0 = legacy constant)
    xcov_start_epoch: int = 1
    xcov_ramp_epochs: int = 0
    inv_cosine: bool = False              # True: scale-invariant cosine L_inv
    # Asymmetric inv (BYOL-style): current L_inv pulls encoder(A) output
    # directly toward encoder(B) output. Without a predictor in between this
    # is BYOL-minus-predictor, a known-collapsing configuration: the EMA
    # target tracks A, A aligns to B, both spiral into a shared low-D
    # subspace (rank 10 -> 1.4 in every inv run). With this flag, alignment
    # work moves into a predictor head P: loss(P(z_s(A)), stopgrad(z_s(B)));
    # gradient still reaches the A-encoder but P absorbs the collapse
    # pressure, letting the encoder keep diverse directions.
    inv_predictor: bool = False           # True: route L_inv via inv_head MLP
    # Stage-0 selective inv (ORACLE mask, diagnostic only): compute L_inv
    # over sound-metal tokens only (GT-derived mask via sound_mask batch
    # key). Tests whether blind alignment washes defect-coupled signal
    # (exp51: all 9 OOD cells degraded). Supervised leak by construction:
    # upper-bound ablation, NEVER the deployable recipe (Stage 1 replaces
    # the oracle with an unsupervised sound estimator).
    inv_oracle_mask: bool = False         # True: masked inv + inv_sound_frac log
    # Grounding job for z_c (exp48 left z_c with no direct objective, so the
    # split was decorative): predict Condition-B token from pooled z_c.
    zc_cond_weight: float = 0.0           # MSE(head(mean(z_c)), stopgrad(c_b))
    # Fix 1 (needle geometry): the DC offset lives in the null space of every
    # loss (shared by both JEPA branches so it cancels in differences;
    # VICReg/xcov center before measuring). Penalize batch-token mean-square
    # per subspace so direction (cosine) becomes meaningful again. Global mean
    # (not per-scan): per-scan condition bias in z_c is legitimate signal and
    # survives, only the shared giant bias is killed.
    offset_weight: float = 0.0            # (||mean(z_c)||^2 + ||mean(z_s)||^2)/D
    # Fix 4 (right granularity): VICReg guards token-level variance but the
    # readout uses POOLED features ([center, mean-rest]), where collapse still
    # happens. Var-hinge on the pooled scattering vector itself.
    pool_var_weight: float = 0.0          # hinge(std(pooled) >= var_gamma)
    # Divergence guard: stop when val recon rises far above its running best
    # for `diverge_patience` consecutive evals (train/val divergence = shortcut
    # or overfit, e.g. exp49 val 0.0275 -> 0.06 while train stayed flat).
    diverge_patience: int = 3             # 0 disables the guard
    diverge_factor: float = 1.5           # trigger when val > best * factor
    downstream_latent_type: str = "scattering"  # 'scattering' (z_s only, drift-free), 'full' ([z_c, z_s]), 'carrier' (z_c)

    # Loss weights
    pred_loss_type: str = "l1"  # Pure L1 loss preserving subtle flaw gradients
    pred_magnitude_weight: float = 0.0    # Scale-invariant log-energy magnitude penalty |log(||pred||) - log(||tgt||)|
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
    # R_eff floor for best-saving (run 161630: rank collapsed 13 -> 1.2 under
    # L_inv while val AUC kept rising; composite kept saving degenerate
    # weights). Only save best when R_eff >= ratio * max-R_eff-so-far
    # (post-warmup). Rank-degenerate metric gains don't count. 0 disables.
    best_reff_floor_ratio: float = 0.5
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
