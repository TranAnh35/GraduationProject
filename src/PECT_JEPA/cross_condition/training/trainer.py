"""
Official Trainer for Cross-Condition PECT-JEPA.
Incorporates:
- Real-time tqdm progress bars with streaming metrics.
- Multi-tier logging with dedicated run directory, timestamps, and metrics_epoch.csv.
- Resume training support from saved checkpoint.
- Automatic Mixed Precision (AMP FP16).
- Cosine Annealing Learning Rate Schedule.
- Exponential Moving Average (EMA) teacher update.
- Periodic Latent SVD & Separability (SNR) diagnostics.
- Automatic training curve visualization.
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
from ..utils.logger import CrossConditionLogger
from ..evaluation.visualizations import plot_training_curves
from .diagnostics import compute_svd_diagnostics, compute_latent_snr


class CrossConditionTrainer:
    """
    Manages end-to-end self-supervised pretraining of CrossConditionJEPA.
    """
    def __init__(
        self,
        config: CrossConditionConfig,
        device: torch.device,
        logger: Optional[CrossConditionLogger] = None,
    ):
        self.config = config
        self.device = device
        self.logger = logger

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
        self.start_epoch = 1
        self.history: Dict[str, list] = {
            "epoch": [],
            "total_loss": [],
            "pred_loss": [],
            "var_loss": [],
            "cov_loss": [],
            "effective_rank": [],
            "lr": [],
        }

        # 5. Resume training if checkpoint specified
        if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
            self._load_checkpoint(config.resume_checkpoint)

    def _load_checkpoint(self, ckpt_path: str):
        msg = f"Resuming training from checkpoint: {ckpt_path}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        if "history" in ckpt:
            self.history = ckpt["history"]

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

        # EMA momentum cosine schedule
        ema_momentum = self.config.ema_momentum_end - (self.config.ema_momentum_end - self.config.ema_momentum_start) * (
            0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
        )

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

        all_reps = torch.cat(sample_reps, dim=0)
        svd_res = compute_svd_diagnostics(all_reps)

        return {
            "epoch": epoch,
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

        header = (
            f"\n=======================================================\n"
            f"Starting Cross-Condition JEPA Training\n"
            f"Epochs: {self.start_epoch} -> {total_epochs} | Batches/Epoch: {len(loader)} | Batch Size: {self.config.batch_size}\n"
            f"Samples/Epoch: {len(loader) * self.config.batch_size:,} across 112 Directed Pairs\n"
            f"Device: {self.device} | AMP Mixed Precision: {self.config.mixed_precision}\n"
            f"=======================================================\n"
        )
        if self.logger:
            self.logger.info(header)
        else:
            print(header, flush=True)

        best_loss = float("inf")
        ckpt_dir = self.logger.checkpoint_dir if self.logger else self.config.checkpoint_dir
        viz_dir = self.logger.visualizations_dir if self.logger else self.config.visualizations_dir
        best_ckpt_path = os.path.join(ckpt_dir, "best_cross_condition_jepa.pt")

        for epoch in range(self.start_epoch, total_epochs + 1):
            epoch_metrics = self.train_epoch(loader, epoch, total_epochs)

            # Record history
            for k in self.history:
                if k in epoch_metrics:
                    self.history[k].append(epoch_metrics[k])

            summary_msg = (
                f"[Summary Epoch {epoch:02d}/{total_epochs:02d}] "
                f"Loss: {epoch_metrics['total_loss']:.4f} | "
                f"Pred L1: {epoch_metrics['pred_loss']:.4f} | "
                f"Var: {epoch_metrics['var_loss']:.4f} | "
                f"R_eff: {epoch_metrics['effective_rank']:.1f}D | "
                f"LR: {epoch_metrics['lr']:.2e}"
            )
            if self.logger:
                self.logger.info(summary_msg)
                self.logger.log_epoch(epoch_metrics)
            else:
                print(summary_msg, flush=True)

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
            latest_path = os.path.join(ckpt_dir, "latest_cross_condition_jepa.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "metrics": epoch_metrics,
                "history": self.history,
            }, latest_path)

            # Periodically plot training loss curves every 5 epochs
            if epoch % 5 == 0 or epoch == total_epochs:
                curve_path = os.path.join(viz_dir, "training_curves.png")
                plot_training_curves(self.history, curve_path)

        fin_msg = f"\nTraining Complete! Best model saved to: {best_ckpt_path}"
        if self.logger:
            self.logger.info(fin_msg)
        else:
            print(fin_msg, flush=True)

        return best_ckpt_path
