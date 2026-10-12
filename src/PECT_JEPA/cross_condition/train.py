"""
Official Training Entry Point for Cross-Condition PECT-JEPA.

Features:
- Exhaustive Data Utilization: Traverses 100% of spatial coordinates (72,900 pixels) across all directed condition pairs without arbitrary dropping.
- Structured Unified Experiment Directory: experiments/cross_condition/runs/<exp_name>_<YYYYMMDD_HHMMSS>/
  - checkpoints/ (best_model.pt, latest_checkpoint.pt)
  - evaluation_results/ (subsequent downstream evaluation stored in same experiment directory)
  - visualizations/ (training loss & diagnostics curves)
  - config.json, run.log, metrics_epoch.csv
- Real-time unbuffered tqdm progress bar streaming.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from datetime import datetime
import numpy as np
import torch

from .configs.config import CrossConditionConfig
from .data.paired_dataset import PairedPECTDataset
from .data.preprocessing import load_canonical_cscan_from_tdms
from .training.trainer import CrossConditionTrainer
from .utils.logger import CrossConditionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Official Cross-Condition PECT-JEPA Training")
    parser.add_argument("--exp_name", type=str, default="cross_jepa_base", help="Experiment name")
    parser.add_argument("--add_timestamp", type=lambda v: v.lower() == "true", default=True, help="Append timestamp to experiment directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per training step")
    parser.add_argument("--num_partitions", type=int, default=10, help="Number of partitions K per macro-cycle (K sub-epochs = 100%% full universe pass)")
    parser.add_argument("--grid_stride", type=int, default=1, help="Spatial pixel grid stride (1 = exhaustive 100%% all 72,900 pixels per scan)")
    parser.add_argument("--samples_per_epoch", type=int, default=None, help="Explicit samples per epoch (default None: 100%% exhaustive dataset traversal)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate for Cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension D")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--var_weight", type=float, default=1.0, help="VICReg variance hinge weight (0.0 to disable)")
    parser.add_argument("--cov_weight", type=float, default=1.0, help="VICReg covariance penalty weight (0.0 to disable)")
    parser.add_argument("--use_condition_token", type=lambda v: v.lower() == "true", default=True, help="Use condition token c_b in predictor")
    parser.add_argument("--condition_modulation", type=str, default="cross_attention", choices=["cross_attention", "adaln", "additive", "none"], help="Modulation type for condition token c_b")
    parser.add_argument("--tokenizer_type", type=str, default="energy_adaptive", choices=["energy_adaptive", "dual_domain"], help="Tokenizer type ('energy_adaptive' with temporal instance-norm or baseline 'dual_domain')")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs (defaults to K partitions = 1 full macro-cycle)")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience in epochs for early stopping post-warmup (0 to disable)")
    parser.add_argument("--early_stopping_metric", type=str, default="composite", choices=["composite", "val_auc", "latent_snr", "loss"], help="Metric to monitor for early stopping and best checkpoint")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Minimum loss reduction delta for early stopping")
    parser.add_argument("--eval_val_every", type=int, default=1, help="Evaluate validation telemetry every N epochs")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine unified experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}_{timestamp}" if args.add_timestamp else args.exp_name
    run_dir = os.path.join("experiments", "cross_condition", "runs", run_name)

    # Build config from args
    config = CrossConditionConfig(
        exp_name=args.exp_name,
        add_timestamp=args.add_timestamp,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_delta=args.early_stopping_min_delta,
        eval_val_every=args.eval_val_every,
        batch_size=args.batch_size,
        grid_stride=args.grid_stride,
        num_partitions=args.num_partitions,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        use_condition_token=args.use_condition_token,
        condition_modulation=args.condition_modulation,
        tokenizer_type=args.tokenizer_type,
        data_dir=args.data_dir,
        transform_csv="data/ground_truth/raw_tdms_rotate_crop.csv",
        exp_dir=run_dir,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        eval_dir=os.path.join(run_dir, "evaluation_results"),
        visualizations_dir=os.path.join(run_dir, "visualizations"),
        resume_checkpoint=args.resume_checkpoint,
    )

    # Initialize logger
    logger = CrossConditionLogger(run_dir=run_dir, run_name=run_name, config=config)
    device = torch.device(args.device)

    logger.info(f"Initialized Unified Experiment Directory: {run_dir}")
    logger.info(f"Configuration: {config}")

    # Pre-load canonical validation scans for real-time downstream health telemetry
    val_scans = []
    val_candidates = [
        ("data/Hall_Pot_Core/Corrosion/Gauss/corosion_frontside_gaussian_300x300_1khz_amp3.88_z1_20260106_200028.tdms", "data/ground_truth/corrosion/corrosion_gt_mask.npy", "Corrosion_Gauss_z1"),
        ("data/Hall_Pot_Core/Rivet_v1/Gauss/rivet_frontside_gaussian_300x300_1khz_z1_20260111_015456.tdms", "data/ground_truth/rivet_v1/rivet_v1_gt_mask.npy", "Rivet_v1_Gauss_z1"),
    ]
    for tdms_p, gt_p, sname in val_candidates:
        if os.path.exists(tdms_p) and os.path.exists(gt_p):
            logger.info(f"Loading Validation Reference Scan: {sname} ({tdms_p})...")
            cscan = load_canonical_cscan_from_tdms(file_path=tdms_p, apply_transforms=True)
            gt_mask = np.load(gt_p)
            val_scans.append({
                "cscan": cscan,
                "gt_mask": gt_mask,
                "name": sname,
            })
    logger.info(f"Loaded {len(val_scans)} Validation Reference Scans for Real-time Downstream Telemetry.")

    # Build Paired Dataset across all directed pairs
    logger.info(f"Loading Base Domain Dataset (Exhaustive grid stride: {config.grid_stride}, K={config.num_partitions} partitions)...")
    dataset = PairedPECTDataset(
        data_dir=config.data_dir,
        grid_stride=config.grid_stride,
        samples_per_epoch=config.samples_per_epoch,
        num_partitions=config.num_partitions,
        transform_csv=config.transform_csv,
    )

    # Initialize Trainer with logger and validation reference scans
    trainer = CrossConditionTrainer(config=config, device=device, logger=logger, val_scans=val_scans)

    # Start training
    best_ckpt = trainer.train(dataset=dataset, epochs=config.epochs)
    logger.info(f"Training successfully finished! Checkpoint saved at: {best_ckpt}")
    print(f"\nExperiment Run Finished. All checkpoints and logs saved to:\n  {run_dir}\n")


if __name__ == "__main__":
    main()
