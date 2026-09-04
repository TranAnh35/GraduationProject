"""
CLI Training Script for 1D Temporal PECT-JEPA (Stage A).

Features:
- Multi-tier Experiment Tracking (TensorBoard, Structured CSVs, Python logging, optional WandB)
- Full parameter snapshotting (config.json & git_info.json)
- Automatic Best & Latest model saving in both checkpoint and experiment directories.

Usage:
    # Standard training on GPU with TensorBoard logging
    python -m src.PECT_JEPA.temporal_1d.train --data_dir data --epochs 50 --exp_name baseline_peak_early

    # Pilot debug run on 4 files
    python -m src.PECT_JEPA.temporal_1d.train --data_dir data --max_files 4 --epochs 3 --exp_name pilot_test
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.temporal_1d.configs.config import Temporal1DConfig
from src.PECT_JEPA.temporal_1d.data.dataset import (
    PECT1DDataset,
    FileBalancedBatchSampler,
    collate_1d_batch,
)
from src.PECT_JEPA.temporal_1d.data.preprocessing import (
    find_all_tdms_files,
    parse_metadata_from_path,
)
from src.PECT_JEPA.temporal_1d.models.jepa_1d import PECT_JEPA_1D
from src.PECT_JEPA.temporal_1d.training.trainer import JEPATrainer1D
from src.PECT_JEPA.temporal_1d.data.split import split_by_files
from src.PECT_JEPA.temporal_1d.utils.logger import PECTExperimentLogger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("1D Temporal PECT-JEPA (Stage A) training")
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    p.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--k_per_file", type=int, default=8, help="Points per file in file-balanced batch sampler")
    p.add_argument("--max_files", type=int, default=None, help="Limit number of TDMS files for debug")
    p.add_argument("--normalization", type=str, default="peak_early",
                   choices=["peak_early", "zscore", "min_max"])
    p.add_argument("--num_patches", type=int, default=16, help="Number of temporal token patches")
    p.add_argument("--learning_rate", type=float, default=3e-4, help="Base learning rate")
    p.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--mixed_precision", type=lambda v: v.lower() == "true", default=True, help="Use AMP FP16")

    # Logging & Experiment Management Arguments
    p.add_argument("--exp_name", type=str, default="jepa_1d_base", help="Experiment run name")
    p.add_argument("--log_dir", type=str, default="experiments/1d", help="Base directory for experiment logs")
    p.add_argument("--save_dir", type=str, default="checkpoints/pect_jepa_1d", help="Checkpoint save directory")
    p.add_argument("--use_tensorboard", type=lambda v: v.lower() == "true", default=True, help="Enable TensorBoard")
    p.add_argument("--use_wandb", type=lambda v: v.lower() == "true", default=False, help="Enable Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="PECT_JEPA_1D", help="WandB project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")
    p.add_argument("--log_histograms", type=lambda v: v.lower() == "true", default=False, help="Log weight histograms to TB")

    return p


def main():
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    config = Temporal1DConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_per_file=args.k_per_file,
        max_files=args.max_files,
        normalization=args.normalization,
        num_patches=args.num_patches,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        mixed_precision=args.mixed_precision,
        exp_name=args.exp_name,
        log_dir=args.log_dir,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_histograms=args.log_histograms,
    )

    # Initialize unified logger
    logger = PECTExperimentLogger(config)

    # ---- Data Loading -----------------------------------------------------
    all_files = find_all_tdms_files(config.data_dir)
    if config.max_files is not None:
        all_files = all_files[: config.max_files]
    if not all_files:
        logger.error(f"No TDMS files found under: {config.data_dir}")
        raise FileNotFoundError(f"No TDMS files found under: {config.data_dir}")

    logger.info(f"Found {len(all_files)} total TDMS files in dataset.")
    metadata = [parse_metadata_from_path(fp) for fp in all_files]
    for k in ("sensor", "waveform", "liftoff"):
        logger.info(f"  Available {k} categories: {sorted({m[k] for m in metadata})}")

    train_files, val_files, _ = split_by_files(
        all_files, train_ratio=0.8, val_ratio=0.2, seed=config.seed
    )

    tok_pad_to = config.raw_padded_length if config.tokenizer_mode == "raw" else None

    logger.info(f"Indexing training dataset ({len(train_files)} files)...")
    train_set = PECT1DDataset(
        file_paths=train_files,
        time_samples=config.time_samples,
        log_time_samples=config.log_time_samples,
        t_start_frac=config.t_start_frac,
        normalization=config.normalization,
        early_window_frac=config.early_window_frac,
        raster_correction=config.raster_correction,
        use_memmap=config.use_memmap,
        cache_dir=config.cache_dir,
        eps=config.eps,
        pad_to=tok_pad_to,
        pad_mode=config.pad_mode,
    )

    logger.info(f"Indexing validation dataset ({len(val_files)} files)...")
    val_set = PECT1DDataset(
        file_paths=val_files,
        time_samples=config.time_samples,
        log_time_samples=config.log_time_samples,
        t_start_frac=config.t_start_frac,
        normalization=config.normalization,
        early_window_frac=config.early_window_frac,
        raster_correction=config.raster_correction,
        use_memmap=config.use_memmap,
        cache_dir=config.cache_dir,
        eps=config.eps,
        pad_to=tok_pad_to,
        pad_mode=config.pad_mode,
    )

    logger.info(f"Train Dataset: {len(train_files)} files / {len(train_set):,} waveforms")
    logger.info(f"Val Dataset:   {len(val_files)} files / {len(val_set):,} waveforms")

    if config.file_balanced:
        train_sampler = FileBalancedBatchSampler(
            train_set.file_point_counts,
            batch_size=config.batch_size,
            k_per_file=config.k_per_file,
            seed=config.seed,
        )
        train_loader = DataLoader(
            train_set, batch_sampler=train_sampler,
            collate_fn=collate_1d_batch, num_workers=0,
            pin_memory=(config.device == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True,
            collate_fn=collate_1d_batch, num_workers=0,
            pin_memory=(config.device == "cuda"),
        )

    val_loader = DataLoader(
        val_set, batch_size=min(config.batch_size, 128), shuffle=False,
        collate_fn=collate_1d_batch, num_workers=0,
        pin_memory=(config.device == "cuda"),
    )

    # ---- Model & Training ---------------------------------------------------
    model = PECT_JEPA_1D(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PECT_JEPA_1D Trainable Parameters: {n_params / 1e6:.2f}M")

    trainer = JEPATrainer1D(model, train_loader, val_loader, config, logger=logger)
    history = trainer.train()

    history_path = os.path.join(config.save_dir, "history_1d.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved summary history to {history_path}")


if __name__ == "__main__":
    main()
