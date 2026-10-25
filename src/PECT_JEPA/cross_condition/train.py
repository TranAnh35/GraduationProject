"""
Official Training Entry Point for Cross-Condition PECT-JEPA.

Features:
- Exhaustive Data Utilization: Traverses 100% of spatial coordinates across all directed condition pairs without arbitrary dropping.
- Structured Unified Experiment Directory: experiments/cross_condition/runs/<exp_name>_<YYYYMMDD_HHMMSS>/
  - checkpoints/ (best_cross_condition_jepa.pt, latest_cross_condition_jepa.pt)
  - evaluation_results/ (subsequent downstream evaluation stored in same experiment directory)
  - visualizations/ (training loss & diagnostics curves)
  - config.json, run.log, metrics_epoch.csv
- Real-time unbuffered tqdm progress bar streaming.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import glob
import json
from datetime import datetime
import numpy as np
import torch

from .configs.config import CrossConditionConfig
from .data.paired_dataset import PairedPECTDataset
from .data.preprocessing import load_canonical_cscan_from_tdms
from .data.split import (
    SPLIT_MODES,
    extract_file_metadata,
    get_canonical_train_val_test_splits,
)
from .training.trainer import CrossConditionTrainer
from .utils.logger import CrossConditionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Official Cross-Condition PECT-JEPA Training")
    parser.add_argument("--exp_name", type=str, default="cross_jepa_base", help="Experiment name")
    parser.add_argument("--add_timestamp", type=lambda v: v.lower() == "true", default=True, help="Append timestamp to experiment directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per training step")
    parser.add_argument("--num_partitions", type=int, default=10, help="Number of partitions K per macro-cycle (K sub-epochs = 100%% full universe pass)")
    parser.add_argument("--split_mode", type=str, default="cross_specimen", choices=list(SPLIT_MODES), help="Downstream split protocol (telemetry / fit / score roles): 'cross_specimen' = 16-file Base Domain, 'cross_condition' = 24-file Base Domain incl. Rivet_v2. JEPA pretraining itself always uses all TDMS files")
    parser.add_argument("--grid_stride", type=int, default=1, help="Spatial pixel grid stride (1 = exhaustive 100%% all 72,900 pixels per scan, 4 = stride 4 subsampling)")
    parser.add_argument("--star_radii", type=int, nargs=3, default=[2, 4, 7], help="Concentric star ring radii in px (canonical default 2 4 7; legacy 1 3 7)")
    parser.add_argument("--samples_per_epoch", type=int, default=None, help="Explicit samples per epoch (default None: 100%% exhaustive dataset traversal)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate for Cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension D")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--var_weight", type=float, default=0.5, help="VICReg variance hinge weight (0.5 gentle scale anchor)")
    parser.add_argument("--cov_weight", type=float, default=0.2, help="VICReg covariance penalty weight (0.0 to disable)")
    parser.add_argument("--var_gamma", type=float, default=1.0, help="Target standard deviation threshold for VICReg variance hinge")
    parser.add_argument("--relative_target", type=lambda v: v.lower() == "true", default=False, help="Predict relative flaw perturbation Delta H (default False: full target H_tgt)")
    parser.add_argument("--use_condition_token", type=lambda v: v.lower() == "true", default=True, help="Use condition token c_B in predictor")
    parser.add_argument("--condition_modulation", type=str, default="cross_attention", choices=["cross_attention", "adaln"], help="Condition-token routing: c_B joins the predictor KV memory, or AdaLN-modulates the predictor blocks")
    parser.add_argument("--global_bg_energy_anchor", type=lambda v: v.lower() == "true", default=False, help="Append Delta E_global = log(1+E_probe) - log(1+E_corner_bg) to the time branch (wide defects: Rivet_v2)")
    parser.add_argument("--tokenizer_arch", type=str, default="linear", choices=["linear", "multiscale_conv"], help="Tokenizer architecture: 'linear' (legacy flat projection) or 'multiscale_conv' (multi-scale 1D temporal conv stem + continuous phasor)")
    parser.add_argument("--spatial_pool_mode", type=str, default="mean", choices=["mean", "radial"], help="Spatial context aggregation mode: 'mean' (uniform 24-probe average) or 'radial' (Green radial-decay + center-guided attention)")
    parser.add_argument("--radial_decay_init", type=float, default=2.5, help="Initial radial decay alpha for spatial_pool_mode='radial'")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs (defaults to K partitions = 1 full macro-cycle)")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience in epochs for early stopping post-warmup (0 to disable)")
    parser.add_argument("--diverge_patience", type=int, default=3, help="Stop when val recon exceeds factor*best for N consecutive evals (0 to disable)")
    parser.add_argument("--diverge_factor", type=float, default=1.5, help="Val divergence trigger factor over running best")
    parser.add_argument("--early_stopping_metric", type=str, default="composite", choices=["composite", "val_auc", "latent_snr", "loss"], help="Metric to monitor for early stopping and best checkpoint")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Minimum loss reduction delta for early stopping")
    parser.add_argument("--best_reff_floor_ratio", type=float, default=0.5, help="Only save best when R_eff >= ratio * max-R_eff-so-far (0 to disable)")
    parser.add_argument("--eval_val_every", type=int, default=1, help="Evaluate validation telemetry every N epochs")
    parser.add_argument("--inv_weight", type=float, default=0.0, help="Cross-condition invariance weight (cosine or MSE between paired representations)")
    parser.add_argument("--inv_cosine", type=lambda v: v.lower() == "true", default=True, help="Scale-invariant cosine L_inv instead of raw MSE (kills shrink-to-zero shortcut)")
    parser.add_argument("--inv_predictor", type=lambda v: v.lower() == "true", default=False, help="Asymmetric inv: route L_inv via inv_head MLP (BYOL-style, encoder never directly pulled; fixes FM6 rank collapse)")
    parser.add_argument("--inv_trim_q", type=float, default=0.75, help="Background gate for L_inv: keep the keep_q lowest-|Delta E| tokens (1.0 = blind, no gating)")
    parser.add_argument("--inv_start_epoch", type=int, default=1, help="First epoch where L_inv may be > 0 (e.g. 11 to skip warmup)")
    parser.add_argument("--inv_ramp_epochs", type=int, default=0, help="Cosine ramp length for L_inv (0 = constant from start)")
    parser.add_argument("--scan_centered_inv", type=lambda v: v.lower() == "true", default=True, help="Center representations per scan before cosine L_inv")
    parser.add_argument("--scan_centered_vicreg", type=lambda v: v.lower() == "true", default=True, help="Center representations per scan before computing VICReg")
    parser.add_argument("--continuous_pos_embed", type=lambda v: v.lower() == "true", default=True, help="Continuous 2D coordinate MLP pos_mlp(dr, dc) instead of a discrete pos_embed table")
    parser.add_argument("--contiguous_masking", type=lambda v: v.lower() == "true", default=True, help="Contiguous non-enclosing wedge masking (avoids context inside target and vice versa)")
    parser.add_argument("--num_target_rays", type=int, default=3, choices=[3, 5], help="Number of contiguous rays masked (3=9 tgt/16 ctx [36%%], 5=15 tgt/10 ctx [60%%])")
    parser.add_argument("--pred_loss_type", type=str, default="l1", choices=["l1", "normalized_l1"], help="Prediction loss: 'l1' (raw) or 'normalized_l1' (probe-wise z-scored, scale-invariant)")
    parser.add_argument("--anti_hijack_ratio", type=float, default=0.0, help="Cap L_inv at (ratio * pred_loss); 0 disables. VICReg is never capped (FM22 death spiral)")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine unified experiment directory.
    # Resume mode: reuse the folder that CONTAINS the checkpoint instead of
    # creating a new one, so logs/metrics/plots keep appending to the old run.
    # e.g. --resume_checkpoint experiments/.../exp48_.../checkpoints/latest_....pt
    #   -> run_dir = experiments/.../exp48_...
    run_dir = None
    run_name = None
    if args.resume_checkpoint:
        ckpt_abs = os.path.abspath(args.resume_checkpoint)
        ckpt_dir = os.path.dirname(ckpt_abs)          # .../checkpoints
        parent = os.path.dirname(ckpt_dir)            # .../<run_name>
        if os.path.basename(ckpt_dir) == "checkpoints" and os.path.isdir(parent):
            run_dir = os.path.relpath(parent)
            run_name = os.path.basename(parent)
    if run_dir is None:
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
        diverge_patience=args.diverge_patience,
        diverge_factor=args.diverge_factor,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_delta=args.early_stopping_min_delta,
        best_reff_floor_ratio=args.best_reff_floor_ratio,
        eval_val_every=args.eval_val_every,
        split_mode=args.split_mode,
        batch_size=args.batch_size,
        grid_stride=args.grid_stride,
        star_radii=tuple(args.star_radii),
        num_partitions=args.num_partitions,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        var_gamma=args.var_gamma,
        continuous_pos_embed=args.continuous_pos_embed,
        contiguous_masking=args.contiguous_masking,
        scan_centered_inv=args.scan_centered_inv,
        scan_centered_vicreg=args.scan_centered_vicreg,
        global_bg_energy_anchor=args.global_bg_energy_anchor,
        tokenizer_arch=args.tokenizer_arch,
        spatial_pool_mode=args.spatial_pool_mode,
        radial_decay_init=args.radial_decay_init,
        relative_target=args.relative_target,
        use_condition_token=args.use_condition_token,
        condition_modulation=args.condition_modulation,
        inv_weight=args.inv_weight,
        inv_cosine=args.inv_cosine,
        inv_predictor=args.inv_predictor,
        inv_trim_q=args.inv_trim_q,
        inv_start_epoch=args.inv_start_epoch,
        inv_ramp_epochs=args.inv_ramp_epochs,
        num_target_rays=args.num_target_rays,
        pred_loss_type=args.pred_loss_type,
        anti_hijack_ratio=args.anti_hijack_ratio,
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
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger.info(f"Initialized Unified Experiment Directory: {run_dir}")
    logger.info(f"Configuration: {config}")

    # Pre-load canonical validation scans for real-time downstream health telemetry.
    # CRITICAL: these must be HELD-OUT canonical val-split scans (the same files
    # the downstream benchmark evaluates), NEVER train files. A previous version
    # hardcoded 2 TRAIN scans here, inflating every run's "val AUC" (exp52:
    # telemetry 0.935 vs honest downstream val 0.625) and biasing early
    # stopping, best-saving, the divergence guard, and the R_eff floor toward
    # train optimism. Telemetry now matches downstream val exactly.
    val_scans = []
    _gt_rel = {
        "Corrosion": os.path.join("ground_truth", "corrosion", "corrosion_gt_mask.npy"),
        "Rivet_v1": os.path.join("ground_truth", "rivet_v1", "rivet_v1_gt_mask.npy"),
        "Rivet_v2": os.path.join("ground_truth", "rivet_v2", "rivet_v2_gt_mask.npy"),
    }
    _gt_masks = {}
    for _spec, _rel in _gt_rel.items():
        _p = os.path.join(args.data_dir, _rel)
        if os.path.exists(_p):
            _gt_masks[_spec] = np.load(_p)
    _all_tdms = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    _all_tdms = [f for f in _all_tdms if not f.endswith("_index.tdms") and "checkpoint" not in f]
    _splits = get_canonical_train_val_test_splits(_all_tdms, split_mode=config.split_mode, seed=42)
    for tdms_p in _splits["val"]:
        _spec = extract_file_metadata(tdms_p)["specimen"]
        if _spec not in _gt_masks:
            logger.info(f"Skipping telemetry scan (no GT mask for specimen {_spec}): {tdms_p}")
            continue
        logger.info(f"Loading Validation Reference Scan: {os.path.basename(tdms_p)}...")
        cscan = load_canonical_cscan_from_tdms(file_path=tdms_p, apply_transforms=True)
        val_scans.append({
            "cscan": cscan,
            "gt_mask": _gt_masks[_spec],
            "name": os.path.basename(tdms_p),
        })
    if len(val_scans) < len(_splits["val"]):
        logger.info(f"[WARN] Only {len(val_scans)}/{len(_splits['val'])} telemetry scans loaded; "
                    f"early-stopping/divergence guards run on a reduced val set.")
    logger.info(f"Loaded {len(val_scans)} Validation Reference Scans for Real-time Downstream Telemetry.")

    # ---- Data roles: JEPA pretraining vs downstream (no scope flag) ----
    # JEPA pretraining is label-free and always uses EVERY TDMS file. The
    # train/val/test partition is a downstream-only concept: the frozen encoder
    # turns each split into latents for the probes. `split_mode` therefore only
    # selects which files provide telemetry / fit / score below.
    pretrain_files = sorted(_all_tdms)
    logger.info("Data roles [split_mode=%s, seed=42]:" % config.split_mode)
    logger.info(f"  JEPA pretrain (all files)        : {len(pretrain_files):3d} files")
    logger.info(f"  Downstream fit  (train)          : {len(_splits['train']):3d} files")
    logger.info(f"  Val telemetry   (val)            : {len(_splits['val']):3d} files")
    logger.info(f"  Test / OOD score(test)           : {len(_splits['test']):3d} files")
    logger.info("  note: telemetry/test files are inside the pretraining pool, so encoder-side "
                "OOD is closed by construction; downstream numbers measure probe transfer only.")
    with open(os.path.join(run_dir, "data_manifest.json"), "w", encoding="utf-8") as _f:
        json.dump({
            "split_mode": config.split_mode,
            "seed": 42,
            "pretrain": "all_files",
            "counts": {
                "pretrain": len(pretrain_files),
                "train": len(_splits["train"]),
                "val": len(_splits["val"]),
                "test": len(_splits["test"]),
            },
            "pretrain_files": [os.path.basename(p) for p in pretrain_files],
            "train": [os.path.basename(p) for p in _splits["train"]],
            "val": [os.path.basename(p) for p in _splits["val"]],
            "test": [os.path.basename(p) for p in _splits["test"]],
        }, _f, indent=1, ensure_ascii=False)

    # Build Paired Dataset across all directed pairs of every loaded file
    logger.info(f"Loading JEPA pretraining set (all files, grid stride {config.grid_stride}, "
                f"K={config.num_partitions} partitions)...")
    dataset = PairedPECTDataset(
        file_paths=pretrain_files,
        grid_stride=config.grid_stride,
        samples_per_epoch=config.samples_per_epoch,
        num_partitions=config.num_partitions,
        star_radii=config.star_radii,
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
