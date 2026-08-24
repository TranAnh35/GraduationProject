"""
CLI Training Script for 1D Temporal PECT-JEPA (TS-JEPA).
Usage:
    python -m src.PECT_JEPA.temporal_1d.train --data_dir data --epochs 50 --batch_size 512
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.temporal_1d.configs.config import get_default_config_1d, Temporal1DConfig
from src.PECT_JEPA.temporal_1d.data.dataset import PECT1DDataset, collate_1d_batch
from src.PECT_JEPA.temporal_1d.data.split import split_by_files
from src.PECT_JEPA.temporal_1d.data.preprocessing import find_all_tdms_files
from src.PECT_JEPA.temporal_1d.models.jepa_1d import PECT_JEPA_1D
from src.PECT_JEPA.temporal_1d.training.trainer import JEPATrainer1D


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D Temporal PECT-JEPA Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Waveform batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patch_length", type=int, default=32, help="Patch length P")
    parser.add_argument("--stride", type=int, default=31, help="Patch stride")
    parser.add_argument("--embed_dim", type=int, default=128, help="Token embedding dimension D")
    parser.add_argument("--mask_strategy", type=str, default="late_decay", choices=["late_decay", "random_patch"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/pect_jepa_1d", help="Checkpoint save directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    config = get_default_config_1d()
    config.data_dir = args.data_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.patch_length = args.patch_length
    config.stride = args.stride
    config.embed_dim = args.embed_dim
    config.mask_strategy = args.mask_strategy
    config.save_dir = args.save_dir
    config.device = args.device
    config.seed = args.seed

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("=" * 65)
    print("1D TEMPORAL PECT-JEPA (TS-JEPA) TRAINING")
    print(f"Data Dir: {config.data_dir}")
    print(f"Patches: {config.num_patches} (length {config.patch_length}, stride {config.stride}) | D: {config.embed_dim}")
    print(f"Strategy: {config.mask_strategy} | Batch Size: {config.batch_size} | Epochs: {config.epochs}")
    print("=" * 65)

    all_files = find_all_tdms_files(config.data_dir)
    print(f"Found {len(all_files)} TDMS files in {config.data_dir}")
    if len(all_files) == 0:
        print("ERROR: No TDMS files found. Please verify data path.")
        return

    train_files, val_files, test_files = split_by_files(
        all_files,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=config.seed
    )
    print(f"Split: {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")

    train_dataset = PECT1DDataset(
        file_paths=train_files,
        time_samples=config.time_samples,
        normalization=config.normalization,
        use_memmap=config.use_memmap,
        cache_dir=config.cache_dir
    )
    val_dataset = PECT1DDataset(
        file_paths=val_files,
        time_samples=config.time_samples,
        normalization=config.normalization,
        use_memmap=config.use_memmap,
        cache_dir=config.cache_dir
    )

    print(f"Total training waveforms: {len(train_dataset):,} | Total validation waveforms: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_1d_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_1d_batch,
        num_workers=0
    )

    model = PECT_JEPA_1D(config)
    trainer = JEPATrainer1D(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    trainer.train()


if __name__ == "__main__":
    main()
