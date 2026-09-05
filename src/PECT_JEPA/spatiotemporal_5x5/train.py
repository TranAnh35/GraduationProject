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
    p.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size (recommended: 128 - 512 for 5x5)")
    p.add_argument("--k_per_file", type=int, default=8, help="Points per file in file-balanced sampler")
    p.add_argument("--num_workers", type=str, default="auto", help="Number of CPU workers (integer or 'auto')")
    p.add_argument("--preload_ram", type=lambda v: v.lower() == "true", default=False,
                   help="Preload cached data into RAM (default: False to conserve system memory)")
    p.add_argument("--max_files", type=int, default=None, help="Limit number of TDMS files for debug")
    p.add_argument("--resample_mode", type=str, default="linear", choices=["linear", "dual_channel"],
                   help="Resampling mode: 'linear' (128 samples) or 'dual_channel' (256 samples)")
    p.add_argument("--in_channels", type=int, default=128, help="Number of temporal channels (128 for linear, 256 for dual)")
    p.add_argument("--crop_border", type=int, default=10,
                   help="Number of outer boundary pixels to crop on each edge (default: 10 to remove air/edge effect)")
    p.add_argument("--normalization", type=str, default="global_peak", choices=["global_peak", "zscore", "peak_early", "min_max"])
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Base learning rate")
    p.add_argument("--loss_type", type=str, default="smooth_l1", choices=["smooth_l1", "l1", "l2", "cosine"],
                   help="JEPA latent prediction loss function (default: smooth_l1)")
    p.add_argument("--cov_weight", type=float, default=0.5, help="VICReg covariance penalty weight (default: 0.5)")
    p.add_argument("--var_weight", type=float, default=1.0, help="VICReg variance hinge weight (default: 1.0)")
    p.add_argument("--ema_momentum", type=float, default=0.996, help="Target encoder base EMA momentum (default: 0.996)")
    p.add_argument("--embed_dim", type=int, default=128, help="Latent embedding dimension D (default: 128)")
    p.add_argument("--encoder_depth", type=int, default=4, help="Context/Target encoder Transformer depth")
    p.add_argument("--predictor_depth", type=int, default=2, help="Predictor Transformer depth")
    p.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--mixed_precision", type=lambda v: v.lower() == "true", default=True, help="Use AMP FP16")
    p.add_argument("--exp_name", type=str, default="pect_jepa_5x5_base", help="Experiment run name")
    p.add_argument("--save_dir", type=str, default="checkpoints/pect_jepa_5x5", help="Directory to save model checkpoints")
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
        cov_weight=args.cov_weight,
        var_weight=args.var_weight,
        ema_momentum=args.ema_momentum,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        device=args.device,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_histograms=args.log_histograms,
        resume=args.resume,
    )

    # Initialize unified logger
    logger = PECTExperimentLogger5x5(config)

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
    split_summary_path = os.path.join(config.save_dir, f"{config.exp_name}_split_summary.json")
    with open(split_summary_path, "w", encoding="utf-8") as f:
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
        use_memmap=config.use_memmap,
        preload_ram=args.preload_ram,
        return_meta=False,
        cache_dir=config.cache_dir,
        eps=config.eps,
    )

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
        use_memmap=config.use_memmap,
        preload_ram=args.preload_ram,
        return_meta=False,
        cache_dir=config.cache_dir,
        eps=config.eps,
    )

    logger.info(f"Train Dataset: {len(train_files)} files / {len(train_set):,} 5x5 patches")
    logger.info(f"Val Dataset:   {len(val_files)} files / {len(val_set):,} 5x5 patches")

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

    val_loader_kwargs = {
        "collate_fn": collate_5x5_batch,
        "num_workers": min(2, num_workers),
        "pin_memory": (config.device == "cuda" and torch.cuda.is_available()),
    }
    if min(2, num_workers) > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, **val_loader_kwargs
    )

    # 2. Model initialization
    model = PECT_JEPA_5x5(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PECT_JEPA_5x5 Trainable Parameters: {n_params / 1e6:.2f}M")

    # 3. Trainer
    trainer = Trainer5x5(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        resume_checkpoint=args.resume,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
