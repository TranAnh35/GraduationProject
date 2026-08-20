"""
Training script for PECT-JEPA.
Usage:
    python -m src.PECT_JEPA.train --data_dir data --epochs 50 --batch_size 2
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.configs.config import get_default_config, PECTJEPAConfig
from src.PECT_JEPA.data.dataset import PECTDataset, find_all_tdms_files, collate_pect_batch
from src.PECT_JEPA.data.split import split_by_files
from src.PECT_JEPA.models.jepa import PECT_JEPA
from src.PECT_JEPA.training.trainer import JEPATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PECT-JEPA Self-Supervised Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="File-level batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--t_prime", type=int, default=64, choices=[64, 128], help="Temporal latent positions (64 or 128)")
    parser.add_argument("--spatial_patch", type=int, default=8, choices=[4, 8, 16], help="Spatial patch size Ps")
    parser.add_argument("--embed_dim", type=int, default=128, help="Token embedding dimension D")
    parser.add_argument("--save_dir", type=str, default="checkpoints/pect_jepa", help="Checkpoint save directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    config = get_default_config()
    config.data.data_dir = args.data_dir
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.temporal_encoder.t_prime = args.t_prime
    config.tokenizer.spatial_patch = args.spatial_patch
    config.tokenizer.embed_dim = args.embed_dim
    config.training.save_dir = args.save_dir
    config.training.device = args.device
    config.training.seed = args.seed

    # Set seeds
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    print("=" * 60)
    print("PECT-JEPA Self-Supervised Training")
    print(f"Data Dir: {config.data.data_dir}")
    print(f"T': {config.temporal_encoder.t_prime} | P_s: {config.tokenizer.spatial_patch} | D: {config.tokenizer.embed_dim}")
    print(f"Epochs: {config.training.epochs} | Batch Size: {config.training.batch_size} | LR: {config.training.learning_rate}")
    print("=" * 60)

    # 1. Discover TDMS files
    all_files = find_all_tdms_files(config.data.data_dir)
    print(f"Found {len(all_files)} TDMS files in {config.data.data_dir}")
    if len(all_files) == 0:
        print("ERROR: No TDMS files found. Please verify data path.")
        return

    # 2. Split by full files (no intra-file data leakage)
    train_files, val_files, test_files = split_by_files(
        all_files,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=config.training.seed
    )
    print(f"Split: {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")

    # 3. Create Datasets and DataLoaders
    train_dataset = PECTDataset(
        file_paths=train_files,
        time_samples=config.temporal_encoder.raw_samples,
        raster_correction=config.data.raster_correction,
        spatial_size=config.data.spatial_size,
        normalization=config.data.normalization
    )
    val_dataset = PECTDataset(
        file_paths=val_files,
        time_samples=config.temporal_encoder.raw_samples,
        raster_correction=config.data.raster_correction,
        spatial_size=config.data.spatial_size,
        normalization=config.data.normalization
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_pect_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_pect_batch,
        num_workers=0
    )

    # 4. Instantiate Model and Trainer
    model = PECT_JEPA(config)
    trainer = JEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 5. Train
    trainer.train()


if __name__ == "__main__":
    main()
