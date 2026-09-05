"""
Trainer for Unified 5x5 Spatiotemporal PECT-JEPA.
Features multi-tier logging: TensorBoard, structured CSVs, console/file logs, and WandB.
"""

import os
import time
import glob
import numpy as np
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
from ..evaluation.liftoff_invariance import compute_effective_rank
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
        resume_checkpoint: Optional[str] = None,
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
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        if config.save_dir is None:
            base_dir = self.logger.run_dir if self.logger else os.path.join(config.log_dir, config.exp_name)
            config.save_dir = os.path.join(base_dir, "checkpoints")

        os.makedirs(config.save_dir, exist_ok=True)
        if config.log_dir:
            os.makedirs(config.log_dir, exist_ok=True)

        target_resume = resume_checkpoint or getattr(config, "resume", None)
        if target_resume:
            self.resume_from_checkpoint(target_resume)

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
        nan_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs} [Val 5x5]  ",
            dynamic_ncols=True,
            leave=False
        )

        val_features = []
        # Sample representations across the full validation dataset using a stride
        # to capture true representation geometry across defects, sound metal, and lift-offs
        val_stride = max(1, len(self.val_loader) // 16)

        for batch_idx, batch in enumerate(pbar):
            x = batch["data"].to(self.device)
            with create_autocast(self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                loss_dict = self.model(x)

            loss_tensor = loss_dict["loss"]
            if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
                nan_batches += 1
                pbar.set_postfix({"v_loss": "NaN (skip)", "nan_batches": nan_batches})
                continue

            loss_val = float(loss_tensor.item())
            pred_val = float(loss_dict["loss_pred"].item())
            total_loss += loss_val
            total_pred += pred_val
            n_batches += 1

            if batch_idx % val_stride == 0 and len(val_features) < 16:
                z_center = self.model.extract_center_feature(x)
                z_np = z_center.detach().cpu().numpy()
                if not np.isnan(z_np).any():
                    val_features.append(z_np)

            pbar.set_postfix({"v_loss": f"{loss_val:.4f}", "v_pred": f"{pred_val:.4f}"})

        if nan_batches > 0 and self.logger:
            self.logger.warning(
                f"[Val Epoch {self.current_epoch + 1}] Skipped {nan_batches}/{len(self.val_loader)} validation batches due to NaN/Inf loss."
            )

        eff_rank = 0.0
        if val_features:
            feats = np.concatenate(val_features, axis=0)
            feats = feats[~np.isnan(feats).any(axis=1)]
            if len(feats) >= 2:
                eff_rank = compute_effective_rank(feats)

        val_loss = (total_loss / n_batches) if n_batches > 0 else float("nan")
        val_loss_pred = (total_pred / n_batches) if n_batches > 0 else float("nan")

        return {
            "val_loss": val_loss,
            "val_loss_pred": val_loss_pred,
            "effective_rank": eff_rank,
        }

    def save_checkpoint(self, path: str, val_loss: Optional[float] = None, effective_rank: Optional[float] = None):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if hasattr(self, "scaler") else None,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "effective_rank": effective_rank,
            "config": self.config.to_dict(),
        }
        torch.save(ckpt, path)

    def resume_from_checkpoint(self, checkpoint_path_or_keyword: str) -> bool:
        """
        Resumes model, optimizer, scaler, step, epoch, and best metrics from a checkpoint file
        or keyword ('auto', 'latest', 'best').
        """
        keyword = str(checkpoint_path_or_keyword).strip()
        resolved_path = None

        # Build comprehensive list of search directories for checkpoints
        candidate_dirs = []
        if hasattr(self, "config") and getattr(self.config, "save_dir", None):
            candidate_dirs.append(self.config.save_dir)
        if hasattr(self, "logger") and getattr(self.logger, "run_dir", None):
            candidate_dirs.append(os.path.join(self.logger.run_dir, "checkpoints"))
            candidate_dirs.append(self.logger.run_dir)
        # Search existing runs for this exp_name (including timestamped ones)
        exp_name = getattr(self.config, "exp_name", "pect_jepa_5x5_base")
        log_base_dir = getattr(self.config, "log_dir", "experiments/5x5")
        for prev_dir in sorted(glob.glob(os.path.join(log_base_dir, f"{exp_name}*")), reverse=True):
            candidate_dirs.append(os.path.join(prev_dir, "checkpoints"))
            candidate_dirs.append(prev_dir)
        # Legacy fallback directory
        candidate_dirs.append("checkpoints/pect_jepa_5x5")

        seen = set()
        search_dirs = []
        for d in candidate_dirs:
            norm = os.path.normpath(d)
            if norm not in seen and os.path.isdir(norm):
                seen.add(norm)
                search_dirs.append(norm)

        if keyword.lower() in ("auto", "latest", "true", "1"):
            for d in search_dirs:
                c_latest = os.path.join(d, "latest_model_5x5.pt")
                if os.path.isfile(c_latest):
                    resolved_path = c_latest
                    break
            if not resolved_path:
                for d in search_dirs:
                    c_best = os.path.join(d, "best_model_5x5.pt")
                    if os.path.isfile(c_best):
                        resolved_path = c_best
                        break
        elif keyword.lower() == "best":
            for d in search_dirs:
                c_best = os.path.join(d, "best_model_5x5.pt")
                if os.path.isfile(c_best):
                    resolved_path = c_best
                    break
        else:
            if os.path.isfile(checkpoint_path_or_keyword):
                resolved_path = checkpoint_path_or_keyword
            else:
                for d in search_dirs:
                    cand = os.path.join(d, checkpoint_path_or_keyword)
                    if os.path.isfile(cand):
                        resolved_path = cand
                        break

        if not resolved_path or not os.path.isfile(resolved_path):
            msg = (
                f"[Resume] No checkpoint found matching '{checkpoint_path_or_keyword}'. "
                f"Searched in: {search_dirs}. Starting fresh training from epoch 1."
            )
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)
            return False

        msg = f"[Resume] Loading checkpoint from: {resolved_path}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

        checkpoint = torch.load(resolved_path, map_location=self.device)

        # 1. Model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        # 2. Optimizer state
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Resume] Could not restore optimizer state: {e}. Keeping fresh optimizer buffers.")

        # 3. Scaler state
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None and hasattr(self, "scaler"):
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Resume] Could not restore scaler state: {e}. Keeping clean scaler.")

        # 4. Step and Epoch
        self.global_step = checkpoint.get("global_step", 0)
        saved_epoch = checkpoint.get("epoch", -1)
        self.start_epoch = max(0, saved_epoch + 1)
        self.current_epoch = self.start_epoch

        # 5. Best Val Loss
        self.best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))

        # 6. Schedulers: synchronize LR to global_step
        current_lr = self.lr_scheduler.step(self.global_step)

        summary_msg = (
            f"[Resume] Successfully restored checkpoint! Resuming at Epoch {self.start_epoch + 1}/{self.config.epochs} "
            f"(Global Step: {self.global_step}, Best Val Loss: {self.best_val_loss:.4f}, LR: {current_lr:.2e})"
        )
        if self.logger:
            self.logger.info(summary_msg)
        else:
            print(summary_msg)

        return True

    def fit(self):
        msg_start = f"--- Starting Unified 5x5 Spatiotemporal PECT-JEPA Training ({self.config.epochs} epochs, device={self.device}) ---"
        if self.logger:
            self.logger.info(msg_start)
        else:
            print(msg_start)

        if self.start_epoch >= self.config.epochs:
            done_msg = (
                f"[Resume] Training already reached requested epoch {self.start_epoch}/{self.config.epochs}. "
                f"To train further, increase --epochs (e.g. --epochs {self.start_epoch + 20})."
            )
            if self.logger:
                self.logger.info(done_msg)
            else:
                print(done_msg)
            return

        for epoch in range(self.start_epoch, self.config.epochs):
            self.current_epoch = epoch
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            dt = time.time() - t0

            val_str = f" | Val Loss: {val_metrics['val_loss']:.4f}" if val_metrics else ""
            rank_str = f" | Rank: {val_metrics['effective_rank']:.1f}/{self.config.embed_dim}" if val_metrics and "effective_rank" in val_metrics and val_metrics["effective_rank"] > 0 else ""
            log_line = (
                f"[Epoch {epoch + 1:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Pred: {train_metrics['loss_pred']:.4f}, Var: {train_metrics['loss_var']:.4f}, Cov: {train_metrics['loss_cov']:.4f})"
                f"{val_str}{rank_str} [{dt:.1f}s]"
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
                if val_metrics and "effective_rank" in val_metrics:
                    epoch_data["effective_rank"] = val_metrics["effective_rank"]
                self.logger.log_epoch(epoch=epoch + 1, metrics=epoch_data, step=self.global_step)

            # Checkpoint saving
            eff_rank = val_metrics.get("effective_rank") if val_metrics else None
            latest_path = os.path.join(self.config.save_dir, "latest_model_5x5.pt")
            self.save_checkpoint(latest_path, val_metrics.get("val_loss"), eff_rank)

            current_loss = val_metrics.get("val_loss", train_metrics["loss"])
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                best_path = os.path.join(self.config.save_dir, "best_model_5x5.pt")
                self.save_checkpoint(best_path, current_loss, eff_rank)
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
