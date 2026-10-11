"""
Official Trainer for Cross-Condition PECT-JEPA.
Incorporates:
- Real-time tqdm progress bars with streaming metrics.
- Pair-balanced Cross-Condition dataset across all 112 directed pairs.
- Automatic Mixed Precision (AMP FP16).
- Cosine Annealing Learning Rate Schedule.
- Exponential Moving Average (EMA) teacher update.
- Periodic Latent SVD & Separability (SNR) diagnostics.
"""

import os
import sys
import time
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.config import CrossConditionConfig
from ..models.cross_condition_jepa import CrossConditionJEPA
from ..data.paired_dataset import PairedPECTDataset
from .diagnostics import compute_svd_diagnostics, compute_latent_snr


class CrossConditionTrainer:
    """
    Manages end-to-end self-supervised pretraining of CrossConditionJEPA.
    """
    def __init__(
        self,
        config: CrossConditionConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.visualizations_dir, exist_ok=True)

        # 1. Initialize model
        self.model = CrossConditionJEPA(config).to(device)

        # 2. Optimizer & Cosine LR
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # 3. AMP Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")

        # 4. History tracking
        self.history: Dict[str, list] = {
            "total_loss": [],
            "pred_loss": [],
            "var_loss": [],
            "cov_loss": [],
            "effective_rank": [],
            "latent_snr": [],
            "lr": [],
        }

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_batches = len(loader)

        running_loss = 0.0
        running_pred = 0.0
        running_var = 0.0
        running_cov = 0.0

        # Cosine learning rate per epoch
        lr = self.config.min_lr + 0.5 * (self.config.lr - self.config.min_lr) * (
            1.0 + math.cos(math.pi * epoch / total_epochs)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # EMA momentum cosine schedule from ema_momentum_start to ema_momentum_end
        ema_momentum = self.config.ema_momentum_end - (self.config.ema_momentum_end - self.config.ema_momentum_start) * (
            0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
        )

        # tqdm progress bar with unbuffered live streaming
        pbar = tqdm(
            loader,
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )

        sample_reps = []

        for batch_idx, batch in enumerate(pbar):
            x_a = batch["x_a"].to(self.device, non_blocking=True)
            x_b = batch["x_b"].to(self.device, non_blocking=True)
            ref_b = batch["ref_b"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(x_a, x_b, ref_b)
                loss = outputs["total_loss"]

            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            self.model.update_target_encoder(momentum=ema_momentum)

            running_loss += loss.item()
            running_pred += outputs["pred_loss"].item()
            running_var += outputs["var_loss"].item()
            running_cov += outputs["cov_loss"].item()

            if batch_idx < 5:
                sample_reps.append(outputs["H_ctx_a"].detach().cpu())

            # Update tqdm progress bar description
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pred": f"{outputs['pred_loss'].item():.4f}",
                "var": f"{outputs['var_loss'].item():.4f}",
                "lr": f"{lr:.2e}",
            })

        avg_loss = running_loss / total_batches
        avg_pred = running_pred / total_batches
        avg_var = running_var / total_batches
        avg_cov = running_cov / total_batches

        # SVD Effective Rank on sampled context representations
        all_reps = torch.cat(sample_reps, dim=0)
        svd_res = compute_svd_diagnostics(all_reps)

        return {
            "total_loss": avg_loss,
            "pred_loss": avg_pred,
            "var_loss": avg_var,
            "cov_loss": avg_cov,
            "effective_rank": svd_res["effective_rank"],
            "lr": lr,
        }

    def train(
        self,
        dataset: PairedPECTDataset,
        epochs: Optional[int] = None,
    ) -> str:
        total_epochs = epochs or self.config.epochs
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        print(f"\n=======================================================")
        print(f"Starting Cross-Condition JEPA Training")
        print(f"Total Epochs: {total_epochs} | Batches/Epoch: {len(loader)} | Batch Size: {self.config.batch_size}")
        print(f"Total Samples/Epoch: {len(loader) * self.config.batch_size:,} across 112 Directed Pairs")
        print(f"Device: {self.device} | AMP Mixed Precision: {self.config.mixed_precision}")
        print(f"=======================================================\n", flush=True)

        best_loss = float("inf")
        best_ckpt_path = os.path.join(self.config.checkpoint_dir, "best_cross_condition_jepa.pt")

        for epoch in range(1, total_epochs + 1):
            epoch_metrics = self.train_epoch(loader, epoch, total_epochs)

            # Record history
            self.history["total_loss"].append(epoch_metrics["total_loss"])
            self.history["pred_loss"].append(epoch_metrics["pred_loss"])
            self.history["var_loss"].append(epoch_metrics["var_loss"])
            self.history["cov_loss"].append(epoch_metrics["cov_loss"])
            self.history["effective_rank"].append(epoch_metrics["effective_rank"])
            self.history["lr"].append(epoch_metrics["lr"])

            print(
                f"[Summary Epoch {epoch:02d}/{total_epochs:02d}] "
                f"Loss: {epoch_metrics['total_loss']:.4f} | "
                f"Pred L1: {epoch_metrics['pred_loss']:.4f} | "
                f"Var: {epoch_metrics['var_loss']:.4f} | "
                f"R_eff: {epoch_metrics['effective_rank']:.1f}D | "
                f"LR: {epoch_metrics['lr']:.2e}",
                flush=True
            )

            # Checkpoint best model
            if epoch_metrics["total_loss"] < best_loss:
                best_loss = epoch_metrics["total_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "metrics": epoch_metrics,
                }, best_ckpt_path)

            # Save latest checkpoint
            latest_path = os.path.join(self.config.checkpoint_dir, "latest_cross_condition_jepa.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "metrics": epoch_metrics,
                "history": self.history,
            }, latest_path)

        print(f"\nTraining Complete! Best model saved to: {best_ckpt_path}", flush=True)
        return best_ckpt_path
