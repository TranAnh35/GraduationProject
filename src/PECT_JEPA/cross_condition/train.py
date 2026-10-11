"""
Official Training Entry Point for Cross-Condition PECT-JEPA.

Usage:
    python -m src.PECT_JEPA.cross_condition.train --epochs 40 --batch_size 256 --samples_per_epoch 102400
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch

from .configs.config import CrossConditionConfig
from .data.paired_dataset import PairedPECTDataset
from .training.trainer import CrossConditionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Official Cross-Condition PECT-JEPA Training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per training step")
    parser.add_argument("--samples_per_epoch", type=int, default=102400, help="Number of paired patches sampled per epoch")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate for Cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension D")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--transform_csv", type=str, default="data/ground_truth/raw_tdms_rotate_crop.csv", help="Geometric transformation CSV")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/cross_condition/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from args
    config = CrossConditionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        data_dir=args.data_dir,
        transform_csv=args.transform_csv,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = torch.device(args.device)
    print(f"Loaded Configuration: {config}")

    # Build Paired Dataset across all 112 directed pairs
    print(f"Loading Base Domain Dataset (16 files, all 112 directed pairs)...", flush=True)
    dataset = PairedPECTDataset(
        data_dir=config.data_dir,
        samples_per_epoch=config.samples_per_epoch,
        transform_csv=config.transform_csv,
    )

    # Initialize Trainer
    trainer = CrossConditionTrainer(config=config, device=device)

    # Start training
    best_ckpt = trainer.train(dataset=dataset, epochs=config.epochs)
    print(f"\nTraining successfully finished! Checkpoint available at: {best_ckpt}")


if __name__ == "__main__":
    main()
