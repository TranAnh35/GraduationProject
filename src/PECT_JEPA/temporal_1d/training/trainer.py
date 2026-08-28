"""
Trainer for 1D Temporal PECT-JEPA (Stage A).

Reuses v1 optimizer/schedulers. Adds:
  - per-component loss logging (pred / var / cov),
  - effective-rank monitoring of val features (collapse early-warning),
  - 4 GB VRAM friendly defaults (batch 256, AMP).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.jepa_1d import PECT_JEPA_1D
from ..configs.config import Temporal1DConfig
from .optimizer import build_optimizer_1d, WarmupCosineLRScheduler1D, EMAScheduler1D
from ..probing.probe_matrix import extract_pooled_features, effective_rank


class JEPATrainer1D:
    """Trainer class for 1D Transient Waveform JEPA (Stage A)."""

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

        self.device = torch.device(
            self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = build_optimizer_1d(
            self.model, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        self.total_epochs = self.config.epochs
        self.steps_per_epoch = max(1, len(train_loader))
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.warmup_steps = self.config.warmup_epochs * self.steps_per_epoch

        self.lr_scheduler = WarmupCosineLRScheduler1D(
            optimizer=self.optimizer, base_lr=self.config.learning_rate,
            min_lr=self.config.min_lr, warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
        )
        self.ema_scheduler = EMAScheduler1D(
            base_momentum=self.config.ema_momentum,
            final_momentum=self.config.ema_momentum_end,
            total_steps=self.total_steps,
            use_schedule=self.config.use_momentum_schedule,
        )

        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _unpack(self, batch):
        if isinstance(batch, dict):
            return batch["data"]
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        sums = {"loss": 0.0, "loss_pred": 0.0, "loss_var": 0.0, "loss_cov": 0.0}
        n = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch + 1}/{self.total_epochs}]",
            dynamic_ncols=True, leave=True,
        )
        for batch in pbar:
            x = self._unpack(batch).to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(x)
                loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Warning] NaN/Inf loss detected at step {self.global_step}. Skipping batch update.", flush=True)
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            momentum = self.ema_scheduler.get_momentum(self.global_step)
            self.model.update_target_encoder(momentum=momentum)
            current_lr = self.lr_scheduler.step(self.global_step)
            self.global_step += 1

            loss_val = float(loss.item())
            sums["loss"] += loss_val
            for k in ("loss_pred", "loss_var", "loss_cov"):
                sums[k] += float(out[k].item())
            n += 1
            pbar.set_postfix({
                "loss": f"{sums['loss'] / n:.5f}",
                "pred": f"{sums['loss_pred'] / n:.5f}",
                "var": f"{sums['loss_var'] / n:.4f}",
                "cov": f"{sums['loss_cov'] / n:.5f}",
                "lr": f"{current_lr:.2e}",
            })

        return {k: v / max(1, n) for k, v in sums.items()}
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        total, n = 0.0, 0
        for batch in tqdm(self.val_loader, desc="[Val Evaluation]", dynamic_ncols=True, leave=False):
            x = self._unpack(batch).to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(x)
            total += float(out["loss"].item())
            n += 1
        metrics: Dict[str, float] = {"val_loss": total / max(1, n)}

        # Collapse early-warning: effective rank of pooled val features
        feats = extract_pooled_features(self.model, self.val_loader, self.device, max_batches=8)
        metrics["effective_rank"] = float(effective_rank(feats))
        return metrics

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
        print(f"--- Starting 1D Temporal PECT-JEPA (Stage A) Training "
              f"({self.total_epochs} epochs, device={self.device}) ---")
        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "effective_rank": []}

        for epoch in range(self.total_epochs):
            if hasattr(self.train_loader, "batch_sampler") and hasattr(
                self.train_loader.batch_sampler, "set_epoch"
            ):
                self.train_loader.batch_sampler.set_epoch(epoch)

            epoch_metrics = self.train_epoch(epoch)
            history["train_loss"].append(epoch_metrics["loss"])

            if self.val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self.evaluate()
                val_loss = val_metrics.get("val_loss", float("inf"))
                history["val_loss"].append(val_loss)
                history["effective_rank"].append(val_metrics.get("effective_rank", 0.0))
                print(f"--> Epoch {epoch + 1} Evaluation - Val Loss: {val_loss:.6f} | "
                      f"Effective Rank: {val_metrics.get('effective_rank', float('nan')):.1f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.save_dir, "best_model_1d.pt")
                    self.save_checkpoint(best_path, epoch, val_metrics)
                    print(f"Saved new best 1D model to {best_path}")

            latest_path = os.path.join(self.save_dir, "latest_model_1d.pt")
            self.save_checkpoint(latest_path, epoch, epoch_metrics)

        print("--- 1D Temporal PECT-JEPA Training Completed ---")
        return history

