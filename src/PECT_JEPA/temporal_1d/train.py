"""
CLI Training Script for 1D Temporal PECT-JEPA (Stage A).
Usage:
    python -m src.PECT_JEPA.temporal_1d.train --data_dir data --epochs 50
    python -m src.PECT_JEPA.temporal_1d.train --data_dir data --max_files 4 --epochs 3   # pilot
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("1D Temporal PECT-JEPA (Stage A) training")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--k_per_file", type=int, default=8)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--normalization", type=str, default="peak_early",
                   choices=["peak_early", "zscore", "min_max"])
    p.add_argument("--num_patches", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints/pect_jepa_1d")
    p.add_argument("--mixed_precision", type=lambda v: v.lower() == "true", default=True)
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
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        mixed_precision=args.mixed_precision,
    )

    # ---- Data -------------------------------------------------------------
    all_files = find_all_tdms_files(config.data_dir)
    if config.max_files is not None:
        all_files = all_files[: config.max_files]
    if not all_files:
        raise FileNotFoundError(f"No TDMS files found under: {config.data_dir}")
    print(f"Found {len(all_files)} TDMS files.")

    metadata = [parse_metadata_from_path(fp) for fp in all_files]
    for k in ("sensor", "waveform", "liftoff"):
        print(f"  {k}: {sorted({m[k] for m in metadata})}")

    train_files, val_files, _ = split_by_files(
        all_files, train_ratio=0.8, val_ratio=0.2, seed=config.seed
    )

    tok_pad_to = config.raw_padded_length if config.tokenizer_mode == "raw" else None

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
    print(f"Train: {len(train_files)} files / {len(train_set)} points | "
          f"Val: {len(val_files)} files / {len(val_set)} points")

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

    # ---- Model & training ---------------------------------------------------
    model = PECT_JEPA_1D(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params (trainable): {n_params / 1e6:.2f}M")

    trainer = JEPATrainer1D(model, train_loader, val_loader, config)
    history = trainer.train()

    history_path = os.path.join(config.save_dir, "history_1d.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
