"""
Official Training Entry Point for Cross-Condition PECT-JEPA.

Features:
- Structured run directory with timestamp: experiments/cross_condition/runs/<exp_name>_<YYYYMMDD_HHMMSS>/
- Multi-tier logging: console output, run.log, config.json, git_info.json, metrics_epoch.csv.
- Resume training support: --resume_checkpoint <path_to_checkpoint>.
- Dynamic sample scaling: dense sampling of all 112 directed pairs.
- Real-time unbuffered tqdm progress bar streaming.

Usage:
    python -m src.PECT_JEPA.cross_condition.train --epochs 40 --points_per_pair 4000
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from datetime import datetime
import torch

from .configs.config import CrossConditionConfig
from .data.paired_dataset import PairedPECTDataset
from .training.trainer import CrossConditionTrainer
from .utils.logger import CrossConditionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Official Cross-Condition PECT-JEPA Training")
    parser.add_argument("--exp_name", type=str, default="cross_jepa_base", help="Experiment name")
    parser.add_argument("--add_timestamp", type=lambda v: v.lower() == "true", default=True, help="Append timestamp to experiment directory")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per training step")
    parser.add_argument("--points_per_pair", type=int, default=4000, help="Number of coordinate points sampled per condition pair per epoch (112 pairs * 4000 = ~448k samples)")
    parser.add_argument("--samples_per_epoch", type=int, default=None, help="Explicit samples per epoch (overrides points_per_pair)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate for Cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension D")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--transform_csv", type=str, default="data/ground_truth/raw_tdms_rotate_crop.csv", help="Geometric transformation CSV")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine run directory and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}_{timestamp}" if args.add_timestamp else args.exp_name
    run_dir = os.path.join("experiments", "cross_condition", "runs", run_name)

    # Build config from args
    config = CrossConditionConfig(
        exp_name=args.exp_name,
        add_timestamp=args.add_timestamp,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        data_dir=args.data_dir,
        transform_csv=args.transform_csv,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        results_dir=os.path.join(run_dir, "results"),
        visualizations_dir=os.path.join(run_dir, "visualizations"),
        resume_checkpoint=args.resume_checkpoint,
    )

    # Initialize logger
    logger = CrossConditionLogger(run_dir=run_dir, run_name=run_name, config=config)
    device = torch.device(args.device)

    logger.info(f"Initialized Experiment Run: {run_name}")
    logger.info(f"Experiment Directory: {run_dir}")
    logger.info(f"Configuration: {config}")

    # Build Paired Dataset across all 112 directed pairs
    logger.info(f"Loading Base Domain Dataset (16 files, all 112 directed pairs)...")
    dataset = PairedPECTDataset(
        data_dir=config.data_dir,
        samples_per_epoch=config.samples_per_epoch,
        points_per_pair=args.points_per_pair,
        transform_csv=config.transform_csv,
    )

    # Initialize Trainer with logger
    trainer = CrossConditionTrainer(config=config, device=device, logger=logger)

    # Start training
    best_ckpt = trainer.train(dataset=dataset, epochs=config.epochs)
    logger.info(f"Training successfully finished! Checkpoint available at: {best_ckpt}")
    print(f"\nExperiment Run Finished. All artifacts saved to:\n  {run_dir}\n")


if __name__ == "__main__":
    main()
