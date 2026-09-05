"""
Trainer for Unified 5x5 Spatiotemporal PECT-JEPA.
Features multi-tier logging: TensorBoard, structured CSVs, console/file logs, and WandB.
"""

import os
import time
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any

# Clean device-agnostic AMP helpers (eliminates FutureWarnings on PyTorch 2.4+)
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def create_grad_scaler(device_type: str, enabled: bool):
        return _GradScaler(device_type, enabled=enabled)
    def create_autocast(device_type: str, enabled: bool):
        return _autocast(device_type, enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def create_grad_scaler(device_type: str, enabled: bool):
        return _GradScaler(enabled=enabled and device_type == "cuda")
    def create_autocast(device_type: str, enabled: bool):
        return _autocast(enabled=enabled and device_type == "cuda")

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5
from ..utils.logger import PECTExperimentLogger5x5
from .optimizer import build_optimizer_5x5, WarmupCosineLRScheduler5x5, MomentumScheduler5x5


class Trainer5x5:
    """
    Self-Supervised Trainer for 5x5 PECT-JEPA.
    """

    def __init__(
        self,
        model: PECT_JEPA_5x5,
        config: Spatiotemporal5x5Config,
        train_loader,
        val_loader=None,
        logger: Optional[PECTExperimentLogger5x5] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or PECTExperimentLogger5x5(config)

        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self.model.to(self.device)

        self.optimizer = build_optimizer_5x5(
            model=self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.epochs
        warmup_steps = steps_per_epoch * config.warmup_epochs

        self.lr_scheduler = WarmupCosineLRScheduler5x5(
            optimizer=self.optimizer,
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.momentum_scheduler = MomentumScheduler5x5(
            base_momentum=config.ema_momentum,
            final_momentum=config.ema_momentum_end,
            total_steps=total_steps,
        )

        self.scaler = create_grad_scaler(self.device.type, enabled=config.mixed_precision and self.device.type == "cuda")
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pred = 0.0
        total_var = 0.0
        total_cov = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs} [Train 5x5]",
            dynamic_ncols=True,
            leave=False
        )

        for batch in pbar:
            x = batch["data"].to(self.device)

            lr = self.lr_scheduler.step(self.global_step)
            momentum = self.momentum_scheduler.get_momentum(self.global_step)

            self.optimizer.zero_grad()

            with create_autocast(self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                loss_dict = self.model(x)
                loss = loss_dict["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                if self.logger:
                    self.logger.warning(f"NaN/Inf loss at step {self.global_step}. Skipping batch.")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()

            grad_norm = 0.0
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA target update
            self.model.update_target_encoder(momentum=momentum)

            loss_val = float(loss.item())
            pred_val = float(loss_dict["loss_pred"].item())
            var_val = float(loss_dict["loss_var"].item())
            cov_val = float(loss_dict["loss_cov"].item())

            total_loss += loss_val
            total_pred += pred_val
            total_var += var_val
            total_cov += cov_val
            n_batches += 1

            if self.logger:
                self.logger.log_step(
                    step=self.global_step,
                    metrics={
                        "loss": loss_val,
                        "loss_pred": pred_val,
                        "loss_var": var_val,
                        "loss_cov": cov_val,
                        "lr": lr,
                        "momentum": momentum,
                        "grad_norm": grad_norm,
                    },
                    epoch=self.current_epoch + 1
                )

            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "pred": f"{pred_val:.4f}",
                "lr": f"{lr:.1e}"
            })

        metrics = {
            "loss": total_loss / max(1, n_batches),
            "loss_pred": total_pred / max(1, n_batches),
            "loss_var": total_var / max(1, n_batches),
            "loss_cov": total_cov / max(1, n_batches),
        }
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_pred = 0.0
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs} [Val 5x5]  ",
            dynamic_ncols=True,
            leave=False
        )

        for batch in pbar:
            x = batch["data"].to(self.device)
            with create_autocast(self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                loss_dict = self.model(x)
            loss_val = loss_dict["loss"].item()
            pred_val = loss_dict["loss_pred"].item()
            total_loss += loss_val
            total_pred += pred_val
            n_batches += 1
            pbar.set_postfix({"v_loss": f"{loss_val:.4f}", "v_pred": f"{pred_val:.4f}"})

        return {
            "val_loss": total_loss / max(1, n_batches),
            "val_loss_pred": total_pred / max(1, n_batches),
        }

    def save_checkpoint(self, path: str, val_loss: Optional[float] = None):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config.to_dict(),
        }
        torch.save(ckpt, path)

    def fit(self):
        msg_start = f"--- Starting Unified 5x5 Spatiotemporal PECT-JEPA Training ({self.config.epochs} epochs, device={self.device}) ---"
        if self.logger:
            self.logger.info(msg_start)
        else:
            print(msg_start)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            dt = time.time() - t0

            val_str = f" | Val Loss: {val_metrics['val_loss']:.4f}" if val_metrics else ""
            log_line = (
                f"[Epoch {epoch + 1:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Pred: {train_metrics['loss_pred']:.4f}, Var: {train_metrics['loss_var']:.4f}, Cov: {train_metrics['loss_cov']:.4f})"
                f"{val_str} [{dt:.1f}s]"
            )
            if self.logger:
                self.logger.info(log_line)
            else:
                print(log_line)

            # Log epoch metrics to TensorBoard & CSV
            if self.logger:
                epoch_data = {
                    "train_loss": train_metrics["loss"],
                    "lr": self.lr_scheduler.get_lr(self.global_step),
                    "time_sec": dt,
                }
                if val_metrics and "val_loss" in val_metrics:
                    epoch_data["val_loss"] = val_metrics["val_loss"]
                self.logger.log_epoch(epoch=epoch + 1, metrics=epoch_data, step=self.global_step)

            # Checkpoint saving
            latest_path = os.path.join(self.config.save_dir, "latest_model_5x5.pt")
            self.save_checkpoint(latest_path, val_metrics.get("val_loss"))

            current_loss = val_metrics.get("val_loss", train_metrics["loss"])
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                best_path = os.path.join(self.config.save_dir, "best_model_5x5.pt")
                self.save_checkpoint(best_path, current_loss)
                msg_best = f"  --> Saved new best checkpoint: {best_path}"
                if self.logger:
                    self.logger.info(msg_best)
                else:
                    print(msg_best)

        if self.logger:
            self.logger.info("--- 5x5 PECT-JEPA Training Complete ---")
            self.logger.close()
        else:
            print("--- 5x5 PECT-JEPA Training Complete ---")
