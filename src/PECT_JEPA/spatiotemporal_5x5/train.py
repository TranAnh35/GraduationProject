"""
CLI Training Script for Unified 5x5 Spatiotemporal PECT-JEPA.

Features:
- Local 5x5 spatial grid extraction preserving physical continuity
- Contiguous Cluster Masking (40% - 60% irregular cluster)
- Unified Tokenizer + 2D spatial positional embedding
- Multi-worker DataLoader with File-Balanced Sampling
- Mixed Precision (AMP FP16) and Warmup Cosine Decay
- VICReg anti-collapse regularized Smooth L1 loss

Usage:
    # Standard training on GPU
    python -m src.PECT_JEPA.spatiotemporal_5x5.train --data_dir data --epochs 50 --batch_size 256

    # Pilot debug run
    python -m src.PECT_JEPA.spatiotemporal_5x5.train --data_dir data --max_files 4 --epochs 3 --exp_name pilot_5x5
"""

import argparse
import json
import os
import sys
import types
from typing import Any, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Defensive safeguard for HPC clusters where torch._dynamo has broken imports or NumPy 2.x conflicts
try:
    import torch._dynamo
except Exception:
    fake_dynamo = types.ModuleType("torch._dynamo")
    fake_dynamo.disable = lambda fn=None, *args, **kwargs: (fn if fn is not None else (lambda f: f))
    sys.modules["torch._dynamo"] = fake_dynamo

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.dataset import (
    PECT5x5Dataset,
    FileBalancedBatchSampler5x5,
    collate_5x5_batch,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import find_all_tdms_files
from src.PECT_JEPA.spatiotemporal_5x5.data.split import get_dataset_split, extract_file_metadata
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.training.trainer import Trainer5x5
from src.PECT_JEPA.spatiotemporal_5x5.utils.logger import PECTExperimentLogger5x5


def get_optimal_num_workers(requested_workers: Any = "auto") -> int:
    """Dynamically determine optimal num_workers based on available CPU cores."""
    if requested_workers is not None and str(requested_workers).lower() != "auto":
        try:
            return max(0, int(requested_workers))
        except ValueError:
            pass

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if slurm_cpus:
        try:
            n_slurm = int(str(slurm_cpus).split("(")[0])
            return max(1, min(4, n_slurm - 1))
        except Exception:
            pass

    try:
        available_cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cores = os.cpu_count() or 4

    return min(4, max(2, available_cores // 2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Unified 5x5 Spatiotemporal PECT-JEPA Training")
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    p.add_argument("--epochs", type=int, default=10, help="Total training epochs (default: 10)")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size (recommended: 128 - 512 for 5x5)")
    p.add_argument("--k_per_file", type=int, default=8, help="Points per file in file-balanced sampler")
    p.add_argument("--num_workers", type=str, default="auto", help="Number of CPU workers (integer or 'auto')")
    p.add_argument("--preload_ram", type=lambda v: v.lower() == "true", default=True,
                   help="Preload cached data into RAM (default: True to eliminate disk I/O seek latency)")
    p.add_argument("--spatial_topology", type=str, default="concentric_star",
                   choices=["concentric_star", "dense_5x5"],
                   help="Spatial sampling topology: 'concentric_star' (EXP-13: 25 probes over 14x14mm, default) or 'dense_5x5'")
    p.add_argument("--star_radii", type=int, nargs=3, default=[1, 3, 7],
                   help="Radii in mm for concentric star rings (default: 1 3 7)")
    p.add_argument("--max_files", type=int, default=None, help="Limit number of TDMS files for debug")
    p.add_argument("--resample_mode", type=str, default="linear", choices=["linear", "dual_channel"],
                   help="Resampling mode: 'linear' (128 samples) or 'dual_channel' (256 samples)")
    p.add_argument("--in_channels", type=int, default=128, help="Number of temporal channels (128 for linear, 256 for dual)")
    p.add_argument("--crop_border", type=int, default=15,
                   help="Number of outer boundary pixels to crop on each edge (default: 15 to remove air/edge effect)")
    p.add_argument("--normalization", type=str, default="global_peak", choices=["global_peak", "zscore", "peak_early", "min_max"])
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Base learning rate")
    p.add_argument("--loss_type", type=str, default="l1", choices=["l1", "smooth_l1", "l2", "cosine"],
                   help="JEPA latent prediction loss function (default: l1)")
    p.add_argument("--var_weight", type=float, default=1.0,
                   help="VICReg coordinate-wise variance hinge weight (Bardes et al., ICLR 2022, default: 1.0)")
    p.add_argument("--cov_weight", type=float, default=1.0,
                   help="VICReg covariance decorrelation penalty weight (Bardes et al., ICLR 2022, default: 1.0)")
    p.add_argument("--var_gamma", type=float, default=1.0,
                   help="VICReg target standard deviation threshold gamma (anchors coordinate scale, default: 1.0)")
    p.add_argument("--uniformity_weight", type=float, default=0.0,
                   help="Hypersphere Uniformity loss weight (Wang & Isola, ICML 2020, default: 0.0)")
    p.add_argument("--uniformity_t", type=float, default=2.0,
                   help="Gaussian potential parameter t for hypersphere uniformity (default: 2.0)")
    p.add_argument("--uniformity_subsample", type=int, default=1024,
                   help="Subsample size of tokens for hypersphere uniformity (default: 1024)")
    p.add_argument("--norm_floor_weight", type=float, default=0.1,
                   help="Norm-floor barrier weight to prevent zero-vector collapse (default: 0.1)")
    p.add_argument("--norm_floor_target", type=float, default=1.0,
                   help="Minimum target L2 norm of representations (default: 1.0)")
    p.add_argument("--tokenizer_type", type=str, default="spatio_spectral",
                   choices=["spatio_spectral", "skin_depth", "dual_domain_attention", "spatiotemporal_patch", "st_patch", "continuous_stf", "continuous_filterbank", "dual_scale_diffusion", "dual_domain", "time_only", "spatial_grid"],
                   help="Tokenizer architecture: 'spatio_spectral' (EXP-12: 100 skin-depth tokens, default), 'dual_domain_attention', etc.")
    p.add_argument("--num_scales", type=int, default=4,
                   help="Number of physical skin-depth scales for spatio_spectral tokenizer (default: 4)")
    p.add_argument("--masker_type", type=str, default="auto",
                   choices=["auto", "complementary_st", "spatiotemporal_diffusion", "contiguous_cluster"],
                   help="Masker strategy: 'auto' (ContiguousCluster for 25 tok, CST for 100 tok, default), 'complementary_st', or 'contiguous_cluster'")
    p.add_argument("--num_temporal_stages", type=int, default=4,
                   help="Number of chronological diffusion stages for spatiotemporal_patch tokenizer and CST masker (default: 4)")
    p.add_argument("--cst_mask_mode", type=str, default="surface_to_depth", choices=["surface_to_depth", "causal", "random"],
                   help="CST masking temporal partition mode: 'surface_to_depth' (default), 'causal', or 'random'")
    p.add_argument("--predictor_type", type=str, default="residual_diffusion",
                   choices=["residual_diffusion", "residual", "parabolic_diffusion", "operator_diffusion", "standard"],
                   help="Predictor architecture: 'residual_diffusion' (Residual Diffusion Predictor, default), 'parabolic_diffusion', 'operator_diffusion', or 'standard'")
    p.add_argument("--use_target_ema", type=lambda v: v.lower() == "true", default=False,
                   help="Use EMA target encoder (default: False for Single Shared Encoder + Stop-Gradient Target)")
    p.add_argument("--adaptive_disturbance_weight", type=float, default=2.0,
                   help="Field-Disturbance Adaptive Loss weight kappa to address 95% sound metal imbalance (default: 2.0)")
    p.add_argument("--temporal_mono_weight", type=float, default=0.05,
                   help="Temporal diffusion delay monotonicity loss weight (default: 0.05)")
    p.add_argument("--diffusion_gamma_init", type=float, default=1.0,
                   help="Initial spatial diffusion attenuation coefficient gamma for Green's attention bias (default: 1.0)")
    p.add_argument("--diffusion_alpha_init", type=float, default=0.5,
                   help="Initial geometric dispersion scale alpha for Parabolic Green's attention bias (default: 0.5)")
    p.add_argument("--diffusion_beta_init", type=float, default=0.5,
                   help="Initial cross-scale vertical diffusion barrier beta for legacy operator diffusion (default: 0.5)")
    p.add_argument("--fluct_weight", type=float, default=2.0,
                   help="Context-Referenced Fluctuation Loss weight for magnifying defect contrast (default: 2.0)")
    p.add_argument("--liftoff_invar_weight", type=float, default=0.0,
                   help="Physical lift-off invariance loss weight (default: 0.0 for pure JEPA)")
    p.add_argument("--phase_align_weight", type=float, default=0.0,
                   help="Self-supervised phase-depth monotonicity alignment loss weight (default: 0.0 for pure JEPA)")
    p.add_argument("--num_freq_bins", type=int, default=14,
                   help="Number of FFT frequency bins for spectral branch (default: 14, covers 0-2800 Hz)")
    p.add_argument("--spectral_features", type=str, default="phase_and_mag", choices=["phase_and_mag", "phase_only"],
                   help="Spectral features for dual-domain tokenizer: 'phase_and_mag' or 'phase_only' (default: phase_and_mag)")
    p.add_argument("--phase_snr_tapering", type=lambda v: v.lower() == "true", default=True,
                   help="Magnitude-weighted phase tapering for dual-domain tokenizer (default: True)")
    p.add_argument("--phase_noise_floor", type=float, default=0.05,
                   help="Signal magnitude threshold for phase tapering (default: 0.05)")
    p.add_argument("--apply_lowpass", type=lambda v: v.lower() == "true", default=True,
                   help="Apply zero-phase Butterworth lowpass filter to waveforms (default: True)")
    p.add_argument("--lowpass_cutoff", type=float, default=2500.0,
                   help="Lowpass filter cutoff frequency in Hz (default: 2500.0)")
    p.add_argument("--lowpass_order", type=int, default=4,
                   help="Lowpass filter order (default: 4)")
    p.add_argument("--ema_momentum", type=float, default=0.990, help="Target encoder base EMA momentum (default: 0.990)")
    p.add_argument("--ema_momentum_end", type=float, default=0.999,
                   help="Target encoder final EMA momentum cap (default: 0.999; never 1.0 to keep targets dynamic)")
    p.add_argument("--embed_dim", type=int, default=64, help="Latent embedding dimension D (default: 64)")
    p.add_argument("--encoder_depth", type=int, default=4, help="Context/Target encoder Transformer depth")
    p.add_argument("--predictor_depth", type=int, default=2, help="Predictor Transformer depth")
    p.add_argument("--use_radial_attention_bias", type=lambda v: v.lower() == "true", default=True,
                   help="Enable continuous physical radial distance attention bias in ContextEncoder (default: True)")
    p.add_argument("--use_phase_curvature", type=lambda v: v.lower() == "true", default=True,
                   help="Enable harmonic radial phase curvature kappa_theta(f) in SpatioSpectralTokenizer (default: True)")
    p.add_argument("--feature_extraction_mode", type=str, default="unified", choices=["unified", "context"],
                   help="Feature representation mode for C-scan feature extraction (default: unified)")
    p.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--mixed_precision", type=lambda v: v.lower() == "true", default=True, help="Use AMP FP16")
    p.add_argument("--exp_name", type=str, default="exp11_dual_domain_pure_jepa", help="Experiment run name")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Directory to save model checkpoints (default: None -> auto-unified inside experiments/5x5/<exp_name>/checkpoints/)")
    p.add_argument("--add_timestamp", type=lambda v: v.lower() == "true", default=True,
                   help="Append timestamp suffix to exp_name (default: True for isolated run logging)")
    p.add_argument("--split_protocol", type=str, default="compound_ood",
                   choices=["compound_ood", "leave_liftoff", "leave_sensor", "leave_waveform", "leave_specimen", "random"],
                   help="Evaluation/training split protocol: compound_ood (Option A: hold out z3+TMR+Chirp simultaneously), leave_liftoff (LOLO), leave_sensor (LOSO), leave_waveform (LOWO), leave_specimen (LODO), random (default: compound_ood)")
    p.add_argument("--holdout_target", type=str, default="z3",
                   help="Category to hold out for single-factor protocols (e.g. 'z3', 'TMR', 'Chirp', 'Rivet_v2')")
    p.add_argument("--holdout_liftoff", type=str, default="z3",
                   help="Lift-off level to hold out for compound_ood (default: 'z3')")
    p.add_argument("--holdout_sensor", type=str, default="TMR",
                   help="Sensor hardware to hold out for compound_ood (default: 'TMR')")
    p.add_argument("--holdout_waveform", type=str, default="Chirp",
                   help="Waveform shape to hold out for compound_ood (default: 'Chirp')")
    p.add_argument("--val_files_count", type=int, default=4,
                   help="Number of in-domain validation files for compound_ood protocol (default: 4)")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio from training pool (default: 0.1)")
    p.add_argument("--log_dir", type=str, default="experiments/5x5", help="Experiment logs directory")
    p.add_argument("--use_tensorboard", type=lambda v: v.lower() == "true", default=True, help="Enable TensorBoard")
    p.add_argument("--use_wandb", type=lambda v: v.lower() == "true", default=False, help="Enable Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="PECT_JEPA_5x5", help="WandB project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")
    p.add_argument("--log_histograms", type=lambda v: v.lower() == "true", default=False, help="Log histograms to TB")
    p.add_argument("--diagnostics_interval", type=int, default=1,
                   help="Frequency of running physics-grounded foundation diagnostics dashboard (default: 1 = every epoch; 0 = disable)")
    p.add_argument("--early_stopping_metric", type=str, default="val_loss_pred", choices=["val_loss_pred", "val_loss"],
                   help="Metric to monitor for early stopping and best checkpoint saving (default: val_loss_pred)")
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Stop training early if monitored metric fails to improve for N epochs (default: 10; 0 = disabled)")
    p.add_argument("--early_stopping_warmup", type=int, default=5,
                   help="Number of initial epochs during which early stopping patience counter is paused (default: 5)")
    p.add_argument("--eval_after_train", type=lambda v: v.lower() == "true", default=False,
                   help="Automatically run downstream evaluation suite immediately after training completes")
    p.add_argument("--eval_3d", type=lambda v: v.lower() == "true", default=False,
                   help="Enable 3D volumetric defect tomography and topological defect graph in evaluation (default: False)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training from checkpoint: filepath (.pt), 'latest', 'best', or 'auto' (default: None)")

    return p


def main():
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    # If linear mode selected, ensure in_channels matches
    in_channels = args.in_channels
    if args.resample_mode == "linear" and in_channels == 256:
        in_channels = 128

    config = Spatiotemporal5x5Config(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_files=args.max_files,
        resample_mode=args.resample_mode,
        in_channels=in_channels,
        crop_border=args.crop_border,
        normalization=args.normalization,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        liftoff_invar_weight=args.liftoff_invar_weight,
        phase_align_weight=args.phase_align_weight,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        var_gamma=args.var_gamma,
        uniformity_weight=args.uniformity_weight,
        uniformity_t=args.uniformity_t,
        uniformity_subsample=args.uniformity_subsample,
        norm_floor_weight=args.norm_floor_weight,
        norm_floor_target=args.norm_floor_target,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_warmup=args.early_stopping_warmup,
        ema_momentum=args.ema_momentum,
        ema_momentum_end=args.ema_momentum_end,
        tokenizer_type=args.tokenizer_type,
        num_scales=args.num_scales,
        spatial_topology=args.spatial_topology,
        star_radii=tuple(args.star_radii),
        masker_type=args.masker_type,
        num_temporal_stages=args.num_temporal_stages,
        cst_mask_mode=args.cst_mask_mode,
        num_freq_bins=args.num_freq_bins,
        spectral_features=args.spectral_features,
        phase_snr_tapering=args.phase_snr_tapering,
        phase_noise_floor=args.phase_noise_floor,
        apply_lowpass=args.apply_lowpass,
        lowpass_cutoff=args.lowpass_cutoff,
        lowpass_order=args.lowpass_order,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        use_target_ema=args.use_target_ema,
        use_radial_attention_bias=args.use_radial_attention_bias,
        use_phase_curvature=args.use_phase_curvature,
        feature_extraction_mode=args.feature_extraction_mode,
        predictor_type=args.predictor_type,
        predictor_depth=args.predictor_depth,
        diffusion_gamma_init=args.diffusion_gamma_init,
        diffusion_alpha_init=args.diffusion_alpha_init,
        diffusion_beta_init=args.diffusion_beta_init,
        fluct_weight=args.fluct_weight,
        adaptive_disturbance_weight=args.adaptive_disturbance_weight,
        temporal_mono_weight=args.temporal_mono_weight,
        device=args.device,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        add_timestamp=args.add_timestamp,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_histograms=args.log_histograms,
        diagnostics_interval=args.diagnostics_interval,
        resume=args.resume,
    )

    # Initialize unified logger
    logger = PECTExperimentLogger5x5(config)

    if torch.cuda.is_available() and config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Unify checkpoint save_dir inside experiment directory if not explicitly custom
    if config.save_dir is None or config.save_dir in ("checkpoints/pect_jepa_5x5", "auto"):
        config.save_dir = os.path.join(logger.run_dir, "checkpoints")

    os.makedirs(config.save_dir, exist_ok=True)
    config.save_json(os.path.join(logger.run_dir, "config_5x5.json"))
    config.save_json(os.path.join(config.save_dir, "config_5x5.json"))

    # 1. Discover data files
    all_files = find_all_tdms_files(config.data_dir)
    if config.max_files is not None:
        all_files = all_files[: config.max_files]
    if not all_files:
        print(f"[Error] No TDMS files found under: {config.data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(all_files)} total TDMS files in dataset.")
    metadata = [extract_file_metadata(fp) for fp in all_files]
    for k in ("sensor", "waveform", "liftoff", "specimen"):
        vals = sorted({m[k] for m in metadata if k in m and m[k]})
        if vals:
            logger.info(f"  Available {k} categories: {vals}")

    train_files, val_files, test_files, split_summary = get_dataset_split(
        all_files,
        protocol=args.split_protocol,
        holdout_target=args.holdout_target,
        holdout_liftoff=args.holdout_liftoff,
        holdout_sensor=args.holdout_sensor,
        holdout_waveform=args.holdout_waveform,
        val_ratio=args.val_ratio,
        val_files_count=args.val_files_count,
        seed=config.seed,
    )

    if args.split_protocol in ("compound_ood", "tri_ood", "multi_ood", "option_a"):
        logger.info("Dataset Split Protocol: COMPOUND_OOD (Option A - Multi-Domain Shift)")
        logger.info(f"  Holdout factors: Lift-off={split_summary.get('holdout_liftoff')}, Sensor={split_summary.get('holdout_sensor')}, Waveform={split_summary.get('holdout_waveform')}")
        logger.info(f"  Base Domain -> Train: {len(train_files)} files | In-Domain Val: {len(val_files)} files")
        sc = split_summary.get("slice_counts", {})
        logger.info(f"  Held-out Test: {len(test_files)} files (Single OOD: {sc.get('single_liftoff', 0)} z3, {sc.get('single_sensor', 0)} TMR, {sc.get('single_waveform', 0)} Chirp | Compound: {sc.get('compound_double', 0)} double, {sc.get('compound_triple', 0)} triple)")
    else:
        logger.info(f"Dataset Split Protocol: {args.split_protocol.upper()} (Holdout Target: {split_summary.get('holdout_target')})")
        logger.info(f"  Train: {len(train_files)} files | Val: {len(val_files)} files | Held-out Test: {len(test_files)} files")

    # Persist split summary for evaluation reproducibility
    os.makedirs(config.save_dir, exist_ok=True)
    split_summary_path = os.path.join(logger.run_dir, f"{config.exp_name}_split_summary.json")
    with open(split_summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    split_summary_in_ckpt = os.path.join(config.save_dir, f"{config.exp_name}_split_summary.json")
    with open(split_summary_in_ckpt, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info(f"Saved split configuration summary to: {split_summary_path}")

    logger.info(f"Indexing training dataset ({len(train_files)} files, mode={config.resample_mode}, C={config.in_channels})...")
    train_set = PECT5x5Dataset(
        file_paths=train_files,
        grid_size=config.grid_size,
        time_samples=config.time_samples,
        temporal_samples=config.temporal_samples,
        in_channels=config.in_channels,
        resample_mode=config.resample_mode,
        log_time_samples=config.log_time_samples,
        t_start_frac=config.t_start_frac,
        normalization=config.normalization,
        early_window_frac=config.early_window_frac,
        raster_correction=config.raster_correction,
        crop_border=config.crop_border,
        spatial_topology=config.spatial_topology,
        star_radii=config.star_radii,
        apply_lowpass=config.apply_lowpass,
        lowpass_cutoff=config.lowpass_cutoff,
        lowpass_order=config.lowpass_order,
        use_memmap=config.use_memmap,
        preload_ram=args.preload_ram,
        return_meta=False,
        cache_dir=config.cache_dir,
        eps=config.eps,
    )

    logger.info(f"Train Dataset: {len(train_files)} files / {len(train_set):,} 5x5 patches")

    num_workers = get_optimal_num_workers(args.num_workers)
    logger.info(f"DataLoader Worker Allocation: {num_workers} workers")

    train_sampler = FileBalancedBatchSampler5x5(
        dataset=train_set,
        batch_size=config.batch_size,
        k_per_file=args.k_per_file,
        seed=config.seed,
    )

    train_loader_kwargs = {
        "collate_fn": collate_5x5_batch,
        "num_workers": num_workers,
        "pin_memory": (config.device == "cuda" and torch.cuda.is_available()),
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 4 if num_workers <= 2 else 2

    train_loader = DataLoader(train_set, batch_sampler=train_sampler, **train_loader_kwargs)

    val_set = None
    val_loader = None
    if val_files:
        logger.info(f"Indexing validation dataset ({len(val_files)} files, mode={config.resample_mode}, C={config.in_channels})...")
        val_set = PECT5x5Dataset(
            file_paths=val_files,
            grid_size=config.grid_size,
            time_samples=config.time_samples,
            temporal_samples=config.temporal_samples,
            in_channels=config.in_channels,
            resample_mode=config.resample_mode,
            log_time_samples=config.log_time_samples,
            t_start_frac=config.t_start_frac,
            normalization=config.normalization,
            early_window_frac=config.early_window_frac,
            raster_correction=config.raster_correction,
            crop_border=config.crop_border,
            spatial_topology=config.spatial_topology,
            star_radii=config.star_radii,
            apply_lowpass=config.apply_lowpass,
            lowpass_cutoff=config.lowpass_cutoff,
            lowpass_order=config.lowpass_order,
            use_memmap=config.use_memmap,
            preload_ram=args.preload_ram,
            return_meta=False,
            cache_dir=config.cache_dir,
            eps=config.eps,
        )
        logger.info(f"Val Dataset:   {len(val_files)} files / {len(val_set):,} 5x5 patches")

        val_loader_kwargs = {
            "collate_fn": collate_5x5_batch,
            "num_workers": min(2, num_workers),
            "pin_memory": (config.device == "cuda" and torch.cuda.is_available()),
        }
        if min(2, num_workers) > 0:
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 2

        val_g = torch.Generator()
        val_g.manual_seed(config.seed)
        val_loader = DataLoader(
            val_set, batch_size=config.batch_size, shuffle=True, generator=val_g, **val_loader_kwargs
        )
    else:
        logger.info("Val Dataset:   0 files (Validation skipped)")

    # 2. Model initialization
    model = PECT_JEPA_5x5(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PECT_JEPA_5x5 Trainable Parameters: {n_params / 1e6:.2f}M")

    # 4. Trainer
    trainer = Trainer5x5(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        resume_checkpoint=args.resume,
    )

    trainer.fit()

    # Optional automated post-training evaluation suite
    if getattr(args, "eval_after_train", False):
        logger.info("--- Initiating Automated Post-Training Evaluation Suite ---")
        best_ckpt = os.path.join(config.save_dir, "best_model_5x5.pt")
        if not os.path.isfile(best_ckpt):
            best_ckpt = os.path.join(config.save_dir, "latest_model_5x5.pt")

        if os.path.isfile(best_ckpt):
            from .evaluate import main as eval_main
            eval_out_dir = os.path.join(logger.run_dir, "evaluation_results")
            eval_args = [
                "--checkpoint", best_ckpt,
                "--split_summary", split_summary_path,
                "--output_dir", eval_out_dir,
                "--data_dir", config.data_dir,
                "--device", config.device,
                "--crop_border", str(config.crop_border),
            ]
            if getattr(args, "eval_3d", False):
                eval_args.append("--eval_3d")
            old_argv = sys.argv
            try:
                sys.argv = [old_argv[0]] + eval_args
                eval_main()
            except Exception as e:
                logger.warning(f"[Post-Train Eval] Evaluation failed with error: {e}")
            finally:
                sys.argv = old_argv
        else:
            logger.warning(f"[Post-Train Eval] No checkpoint found in {config.save_dir} to evaluate.")


if __name__ == "__main__":
    main()
