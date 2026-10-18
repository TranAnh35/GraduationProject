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
import glob
from datetime import datetime
import numpy as np
import torch

from .configs.config import CrossConditionConfig
from .data.paired_dataset import PairedPECTDataset
from .data.preprocessing import load_canonical_cscan_from_tdms
from .data.split import extract_file_metadata, get_canonical_train_val_test_splits
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
    parser.add_argument("--star_radii", type=int, nargs=3, default=[1, 3, 7], help="Concentric star ring radii in px (default 1 3 7; e.g. 2 4 7 for harder masking)")
    parser.add_argument("--samples_per_epoch", type=int, default=None, help="Explicit samples per epoch (default None: 100%% exhaustive dataset traversal)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate for Cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension D")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--var_weight", type=float, default=0.5, help="VICReg variance hinge weight (0.5 gentle scale anchor)")
    parser.add_argument("--cov_weight", type=float, default=0.0, help="VICReg covariance penalty weight (0.0 to disable)")
    parser.add_argument("--relative_target", type=lambda v: v.lower() == "true", default=False, help="Predict relative flaw perturbation Delta H (default False: full target H_tgt)")
    parser.add_argument("--use_condition_token", type=lambda v: v.lower() == "true", default=True, help="Use condition token c_b in predictor")
    parser.add_argument("--condition_modulation", type=str, default="query_projection", choices=["query_projection", "cross_attention", "adaln", "additive", "none"], help="Modulation type for condition token c_b")
    parser.add_argument("--tokenizer_type", type=str, default="dual_domain", choices=["energy_adaptive", "patch_energy_adaptive", "dual_domain", "waveform_invariant"], help="Tokenizer type ('waveform_invariant', 'dual_domain', 'energy_adaptive', 'patch_energy_adaptive')")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs (defaults to K partitions = 1 full macro-cycle)")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience in epochs for early stopping post-warmup (0 to disable)")
    parser.add_argument("--diverge_patience", type=int, default=3, help="Stop when val recon exceeds factor*best for N consecutive evals (0 to disable)")
    parser.add_argument("--diverge_factor", type=float, default=1.5, help="Val divergence trigger factor over running best")
    parser.add_argument("--early_stopping_metric", type=str, default="composite", choices=["composite", "val_auc", "latent_snr", "loss"], help="Metric to monitor for early stopping and best checkpoint")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Minimum loss reduction delta for early stopping")
    parser.add_argument("--best_reff_floor_ratio", type=float, default=0.5, help="Only save best when R_eff >= ratio * max-R_eff-so-far (0 to disable)")
    parser.add_argument("--eval_val_every", type=int, default=1, help="Evaluate validation telemetry every N epochs")
    parser.add_argument("--latent_disentanglement", type=lambda v: v.lower() == "true", default=False, help="Split latent space into [z_c, z_s] (Carrier vs Scattering)")
    parser.add_argument("--carrier_dim", type=int, default=32, help="Dimension of carrier subspace z_c")
    parser.add_argument("--scattering_dim", type=int, default=32, help="Dimension of scattering subspace z_s")
    parser.add_argument("--carrier_flat_weight", type=float, default=1.0, help="Spatial flatness penalty weight on carrier subspace z_c")
    parser.add_argument("--scattering_var_weight", type=float, default=1.0, help="VICReg variance weight on scattering subspace z_s")
    parser.add_argument("--scattering_cov_weight", type=float, default=0.2, help="VICReg covariance weight on scattering subspace z_s (0.2 holds rank vs inv 0.5; run 161630 collapsed at 0.05)")
    parser.add_argument("--inv_weight", type=float, default=0.0, help="EXP-49 cross-condition invariance MSE(z_s(A), stopgrad(z_s(B))) weight")
    parser.add_argument("--xcov_weight", type=float, default=0.0, help="EXP-49 cross-subspace decorrelation ||XCov(z_c,z_s)||^2 weight")
    parser.add_argument("--inv_cosine", type=lambda v: v.lower() == "true", default=False, help="P0-1: scale-invariant cosine L_inv instead of raw MSE (kills shrink-to-zero shortcut)")
    parser.add_argument("--inv_predictor", type=lambda v: v.lower() == "true", default=False, help="Asymmetric inv: route L_inv via inv_head MLP (BYOL-style, encoder never directly pulled)")
    parser.add_argument("--inv_oracle_mask", type=lambda v: v.lower() == "true", default=False, help="Stage-0 selective inv: L_inv over GT sound tokens only (supervised diagnostic, not deployable)")
    parser.add_argument("--inv_start_epoch", type=int, default=1, help="P0-1: first epoch where L_inv may be > 0 (e.g. 11 to skip warmup)")
    parser.add_argument("--inv_ramp_epochs", type=int, default=0, help="P0-1: cosine ramp length for L_inv (0 = constant from start)")
    parser.add_argument("--xcov_start_epoch", type=int, default=1, help="P0-1 schedule start for L_xcov")
    parser.add_argument("--xcov_ramp_epochs", type=int, default=0, help="P0-1 cosine ramp length for L_xcov")
    parser.add_argument("--zc_cond_weight", type=float, default=0.0, help="Grounding job for z_c: MSE(head(mean(z_c)), stopgrad(c_b))")
    parser.add_argument("--offset_weight", type=float, default=0.0, help="Fix 1: kill shared DC offset, (||mean(zc)||^2+||mean(zs)||^2)/D (try 1e-4)")
    parser.add_argument("--pool_var_weight", type=float, default=0.0, help="Fix 4: var-hinge on pooled readout vector (try 0.5)")
    parser.add_argument("--scan_centered_vicreg", type=lambda v: v.lower() == "true", default=True, help="Center representations per scan before computing VICReg")
    parser.add_argument("--continuous_pos_embed", type=lambda v: v.lower() == "true", default=False, help="Option A: continuous 2D coordinate MLP pos_mlp(dr, dc) instead of discrete pos_embed table")
    parser.add_argument("--contiguous_masking", type=lambda v: v.lower() == "true", default=False, help="Contiguous non-enclosing wedge masking (avoids context inside target and vice versa)")
    parser.add_argument("--predictor_scattering_only", type=lambda v: v.lower() == "true", default=False, help="Zero out H_ctx_c before predictor so predictor reads only z_s")
    parser.add_argument("--scan_centered_inv", type=lambda v: v.lower() == "true", default=False, help="Center z_s before cosine L_inv to eliminate LayerNorm DC bias needle")
    parser.add_argument("--patch_energy_norm", type=lambda v: v.lower() == "true", default=False, help="Normalize temporal amplitude by shared 25-probe patch std (dim=(-2,-1)) instead of per-probe (dim=-1)")
    parser.add_argument("--median_patch_energy", type=lambda v: v.lower() == "true", default=False, help="Keep per-probe unit-variance x_norm while normalizing amplitude/spectral anchors by 25-probe median std")
    parser.add_argument("--clean_phasor_token", type=lambda v: v.lower() == "true", default=False, help="Canonical peak-polarity alignment + 5 diffusion energy quantiles + root-saliency complex phasors (fixes FM16 & FM18)")
    parser.add_argument("--subspace_norm", type=lambda v: v.lower() == "true", default=False, help="Independent LayerNorm(D_c) and bias-free LayerNorm(D_s, bias=False) on [z_c, z_s] (fixes FM17 & FM8)")
    parser.add_argument("--relative_energy_anchor", type=lambda v: v.lower() == "true", default=False, help="Append spatial energy ratio log(1 + E_i) - log(1 + E_context) to time branch (fixes FM11 on Rivet_v2 without FM13/FM14)")
    parser.add_argument("--patch_polarity_align", type=lambda v: v.lower() == "true", default=False, help="Align patch polarity using center probe 0 dominant peak (fixes TMR polarity without FM18 spectral starvation)")
    parser.add_argument("--bi_directional_relative_energy", type=lambda v: v.lower() == "true", default=False, help="Append both signed Delta E and absolute magnitude |Delta E| to time branch (eliminates Rivet_v2 sign inversion)")
    parser.add_argument("--var_gamma", type=float, default=1.0, help="Target standard deviation threshold for VICReg variance hinge")
    parser.add_argument("--downstream_latent_type", type=str, default="scattering", choices=["scattering", "carrier", "full"], help="Subspace to extract for downstream tasks")
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
        continuous_pos_embed=args.continuous_pos_embed,
        contiguous_masking=args.contiguous_masking,
        predictor_scattering_only=args.predictor_scattering_only,
        scan_centered_inv=args.scan_centered_inv,
        patch_energy_norm=args.patch_energy_norm,
        median_patch_energy=args.median_patch_energy,
        clean_phasor_token=args.clean_phasor_token,
        subspace_norm=args.subspace_norm,
        relative_energy_anchor=args.relative_energy_anchor,
        patch_polarity_align=args.patch_polarity_align,
        bi_directional_relative_energy=args.bi_directional_relative_energy,
        relative_target=args.relative_target,
        use_condition_token=args.use_condition_token,
        condition_modulation=args.condition_modulation,
        tokenizer_type=args.tokenizer_type,
        latent_disentanglement=args.latent_disentanglement,
        carrier_dim=args.carrier_dim,
        scattering_dim=args.scattering_dim,
        carrier_flat_weight=args.carrier_flat_weight,
        scattering_var_weight=args.scattering_var_weight,
        scattering_cov_weight=args.scattering_cov_weight,
        inv_weight=args.inv_weight,
        xcov_weight=args.xcov_weight,
        inv_cosine=args.inv_cosine,
        inv_predictor=args.inv_predictor,
        inv_oracle_mask=args.inv_oracle_mask,
        inv_start_epoch=args.inv_start_epoch,
        inv_ramp_epochs=args.inv_ramp_epochs,
        xcov_start_epoch=args.xcov_start_epoch,
        xcov_ramp_epochs=args.xcov_ramp_epochs,
        zc_cond_weight=args.zc_cond_weight,
        offset_weight=args.offset_weight,
        pool_var_weight=args.pool_var_weight,
        scan_centered_vicreg=args.scan_centered_vicreg,
        var_gamma=args.var_gamma,
        downstream_latent_type=args.downstream_latent_type,
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
    # CRITICAL: these must be HELD-OUT canonical val-split scans (the same 4
    # files the downstream benchmark evaluates), NEVER train files. A previous
    # version hardcoded 2 TRAIN scans here, inflating every run's "val AUC"
    # (exp52: telemetry 0.935 vs honest downstream val 0.625) and biasing
    # early stopping, best-saving, the divergence guard, and the R_eff floor
    # toward train optimism. Telemetry now matches downstream val exactly.
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
    _splits = get_canonical_train_val_test_splits(_all_tdms, val_files_count=4, seed=42)
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

    # Build Paired Dataset across all directed pairs
    logger.info(f"Loading Base Domain Dataset (Exhaustive grid stride: {config.grid_stride}, K={config.num_partitions} partitions)...")
    dataset = PairedPECTDataset(
        data_dir=config.data_dir,
        grid_stride=config.grid_stride,
        samples_per_epoch=config.samples_per_epoch,
        num_partitions=config.num_partitions,
        star_radii=config.star_radii,
        transform_csv=config.transform_csv,
        include_sound_mask=args.inv_oracle_mask,
    )

    # Initialize Trainer with logger and validation reference scans
    trainer = CrossConditionTrainer(config=config, device=device, logger=logger, val_scans=val_scans)

    # Start training
    best_ckpt = trainer.train(dataset=dataset, epochs=config.epochs)
    logger.info(f"Training successfully finished! Checkpoint saved at: {best_ckpt}")
    print(f"\nExperiment Run Finished. All checkpoints and logs saved to:\n  {run_dir}\n")


if __name__ == "__main__":
    main()
