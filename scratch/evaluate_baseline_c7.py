#!/usr/bin/env python3
"""
Baseline C7: Predictor Loss Evaluation relative to:
1. Zero Prediction: H_pred = 0
2. Dataset Static Mean Prediction: H_pred = E[Delta H_tgt]
3. EXP-41 Trained JEPA Predictor
4. Random Untrained Encoder + Predictor
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.data.dataset import PECT5x5Dataset, collate_5x5_batch
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Baseline C7 on device: {device}")

    cfg_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    ckpt_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/best_model_5x5.pt"
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"

    config = Spatiotemporal5x5Config.from_json(cfg_path)

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_files = splits["val_files"]
    print(f"Validation set: {len(val_files)} files")

    # Build validation dataset and loader
    val_dataset = PECT5x5Dataset(
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
        spatial_topology=config.spatial_topology,
        star_radii=config.star_radii,
        apply_lowpass=config.apply_lowpass,
        lowpass_cutoff=config.lowpass_cutoff,
        lowpass_order=config.lowpass_order,
        use_memmap=config.use_memmap,
        preload_ram=False,
        return_meta=False,
        cache_dir=config.cache_dir,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_5x5_batch,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 1. Load Trained EXP-41 Model
    trained_model = PECT_JEPA_5x5(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(ckpt["model_state_dict"])
    trained_model.to(device).eval()

    # Pass 1: Collect targets and predictions for Trained Model & calculate Dataset Static Mean
    targets_all = []
    preds_all = []
    max_batches = 150

    print("\n--- Pass 1: Evaluating Trained EXP-41 & Collecting Targets ---")
    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(val_loader, desc="Pass 1 Val Batches")):
            if b_idx >= max_batches:
                break
            x = batch["data"].to(device, non_blocking=True)
            file_ids = batch.get("file_ids", None)
            if file_ids is not None:
                file_ids = file_ids.to(device, non_blocking=True)
            loss_dict = trained_model(x, file_ids=file_ids)
            H_tgt = loss_dict["H_target_for_loss"].detach().cpu()
            H_pred = loss_dict["H_pred_for_loss"].detach().cpu()
            targets_all.append(H_tgt)
            preds_all.append(H_pred)

    targets_tensor = torch.cat(targets_all, dim=0)  # [N, K, D]
    preds_tensor = torch.cat(preds_all, dim=0)      # [N, K, D]

    # Calculate losses
    loss_trained = F.smooth_l1_loss(preds_tensor, targets_tensor).item()
    loss_zero = F.smooth_l1_loss(torch.zeros_like(targets_tensor), targets_tensor).item()

    static_mean = targets_tensor.mean(dim=(0, 1), keepdim=True) # [1, 1, D]
    loss_static_mean = F.smooth_l1_loss(static_mean.expand_as(targets_tensor), targets_tensor).item()

    print("\n" + "=" * 80)
    print("BASELINE C7 PREDICTOR LOSS AUDIT (SMOOTH L1 ON EXP-41 VAL SET)")
    print("=" * 80)
    print(f"Zero Prediction Loss (H_pred = 0):              {loss_zero:.6f}")
    print(f"Static Mean Prediction Loss (H_pred = mean):     {loss_static_mean:.6f}")
    print(f"Trained EXP-41 JEPA Predictor Loss:             {loss_trained:.6f}")

    rel_improvement_vs_zero = (loss_zero - loss_trained) / loss_zero * 100.0
    rel_improvement_vs_mean = (loss_static_mean - loss_trained) / loss_static_mean * 100.0

    print(f"\nRelative Improvement vs Zero:                    {rel_improvement_vs_zero:+.2f}%")
    print(f"Relative Improvement vs Static Mean:             {rel_improvement_vs_mean:+.2f}%")

    # 2. Random Untrained Model Evaluation
    print("\n--- Pass 2: Evaluating Random Untrained Model (Seed 999) ---")
    torch.manual_seed(999)
    random_model = PECT_JEPA_5x5(config)
    random_model.to(device).eval()

    rnd_targets_all = []
    rnd_preds_all = []
    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(val_loader, desc="Pass 2 Random Model")):
            if b_idx >= max_batches:
                break
            x = batch["data"].to(device, non_blocking=True)
            file_ids = batch.get("file_ids", None)
            if file_ids is not None:
                file_ids = file_ids.to(device, non_blocking=True)
            loss_dict = random_model(x, file_ids=file_ids)
            rnd_targets_all.append(loss_dict["H_target_for_loss"].detach().cpu())
            rnd_preds_all.append(loss_dict["H_pred_for_loss"].detach().cpu())

    rnd_targets = torch.cat(rnd_targets_all, dim=0)
    rnd_preds = torch.cat(rnd_preds_all, dim=0)
    rnd_loss_pred = F.smooth_l1_loss(rnd_preds, rnd_targets).item()
    rnd_loss_zero = F.smooth_l1_loss(torch.zeros_like(rnd_targets), rnd_targets).item()
    rnd_static_mean = rnd_targets.mean(dim=(0, 1), keepdim=True)
    rnd_loss_mean = F.smooth_l1_loss(rnd_static_mean.expand_as(rnd_targets), rnd_targets).item()

    print(f"\nRandom Model - Untrained Predictor Loss:         {rnd_loss_pred:.6f}")
    print(f"Random Model - Zero Prediction Loss:             {rnd_loss_zero:.6f}")
    print(f"Random Model - Static Mean Loss:                 {rnd_loss_mean:.6f}")
    rnd_rel_vs_zero = (rnd_loss_zero - rnd_loss_pred) / rnd_loss_zero * 100.0
    rnd_rel_vs_mean = (rnd_loss_mean - rnd_loss_pred) / rnd_loss_mean * 100.0
    print(f"Random Model Relative vs Zero:                   {rnd_rel_vs_zero:+.2f}%")
    print(f"Random Model Relative vs Static Mean:            {rnd_rel_vs_mean:+.2f}%")

    results = {
        "exp41_trained": {
            "loss_pred": loss_trained,
            "loss_zero": loss_zero,
            "loss_static_mean": loss_static_mean,
            "rel_gain_vs_zero_pct": rel_improvement_vs_zero,
            "rel_gain_vs_static_mean_pct": rel_improvement_vs_mean,
        },
        "random_baseline": {
            "loss_pred": rnd_loss_pred,
            "loss_zero": rnd_loss_zero,
            "loss_static_mean": rnd_loss_mean,
            "rel_gain_vs_zero_pct": rnd_rel_vs_zero,
            "rel_gain_vs_static_mean_pct": rnd_rel_vs_mean,
        }
    }

    out_file = "experiments/5x5/exp41_autonomous_dual_domain/evaluation_results/baseline_c7_predictor.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved C7 results to {out_file}")


if __name__ == "__main__":
    main()
