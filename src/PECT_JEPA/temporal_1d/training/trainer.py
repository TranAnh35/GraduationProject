"""
Trainer for 1D Temporal PECT-JEPA (TS-JEPA) with tqdm progress tracking.
Supports fast mini-batching over millions of 1D waveforms, mixed precision, and EMA updates.
"""

import os
import time
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import Temporal1DConfig
from .optimizer import build_optimizer_1d, WarmupCosineLRScheduler1D, EMAScheduler1D


class JEPATrainer1D:
    """
    Trainer class for 1D Transient Waveform JEPA.
    """
    def __init__(
        self,
        model: PECT_JEPA_1D,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Temporal1DConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or model.config

        # Device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = build_optimizer_1d(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Steps calculation
        self.total_epochs = self.config.epochs
        self.steps_per_epoch = max(1, len(train_loader))
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.warmup_steps = self.config.warmup_epochs * self.steps_per_epoch

        # Schedulers
        self.lr_scheduler = WarmupCosineLRScheduler1D(
            optimizer=self.optimizer,
            base_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps
        )
        self.ema_scheduler = EMAScheduler1D(
            base_momentum=self.config.ema_momentum,
            final_momentum=self.config.ema_momentum_end,
            total_steps=self.total_steps,
            use_schedule=self.config.use_momentum_schedule
        )

        # Mixed precision
        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch+1}/{self.total_epochs}]",
            dynamic_ncols=True,
            leave=True
        )

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                x = batch["data"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(x)
                loss = out["loss"]

            self.scaler.scale(loss).backward()

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Target Encoder EMA update
            momentum = self.ema_scheduler.get_momentum(self.global_step)
            self.model.update_target_encoder(momentum=momentum)

            # Step learning rate
            current_lr = self.lr_scheduler.step(self.global_step)

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.5f}",
                "avg_loss": f"{total_loss / n_batches:.5f}",
                "lr": f"{current_lr:.2e}",
                "ema": f"{momentum:.4f}"
            })

        avg_loss = total_loss / max(1, n_batches)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc="[Val Evaluation]", dynamic_ncols=True, leave=False)

        for batch in pbar:
            if isinstance(batch, dict):
                x = batch["data"]
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(x)
                loss = out["loss"]

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"val_loss": f"{total_loss / n_batches:.5f}"})

        avg_loss = total_loss / max(1, n_batches)
        return {"val_loss": avg_loss}

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "config": self.config.to_dict(),
            "metrics": metrics
        }
        torch.save(checkpoint, path)

    def train(self) -> Dict[str, Any]:
        print(f"--- Starting 1D Temporal PECT-JEPA Training ({self.total_epochs} epochs) ---")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.total_epochs):
            epoch_metrics = self.train_epoch(epoch)
            train_loss = epoch_metrics["train_loss"]
            history["train_loss"].append(train_loss)

            if self.val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.evaluate()
                val_loss = val_metrics.get("val_loss", float("inf"))
                history["val_loss"].append(val_loss)
                print(f"--> Epoch {epoch+1} Evaluation - Val Loss: {val_loss:.6f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.save_dir, "best_model_1d.pt")
                    self.save_checkpoint(best_path, epoch, val_metrics)
                    print(f"Saved new best 1D model to {best_path}")

            latest_path = os.path.join(self.save_dir, "latest_model_1d.pt")
            self.save_checkpoint(latest_path, epoch, epoch_metrics)

        print("--- 1D Temporal PECT-JEPA Training Completed ---")
        return history
