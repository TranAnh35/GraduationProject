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
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.config import CrossConditionConfig
from ..models.cross_condition_jepa import CrossConditionJEPA
from ..data.paired_dataset import PairedPECTDataset
from ..utils.logger import CrossConditionLogger
from ..evaluation.visualizations import plot_training_curves
from ..evaluation.cscan_extractor import CScanFeatureExtractor
from ..evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
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
        val_scans: Optional[List[Dict[str, Any]]] = None,
    ):
        self.config = config
        self.device = device
        self.logger = logger
        self.val_scans = val_scans or []

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
            "val_auc": [],
            "val_ap": [],
            "val_snr": [],
            "val_sound_var": [],
        }

    @torch.no_grad()
    def evaluate_validation_metrics(self) -> Dict[str, float]:
        """
        Computes fast downstream validation metrics (Unsupervised Mahalanobis AUC/AP, Latent SNR, Sound Var)
        across held-out validation C-scans in sub-second GPU time.
        """
        if not self.val_scans:
            return {}

        was_training = self.model.training
        self.model.eval()

        extractor = CScanFeatureExtractor(
            model=self.model,
            device=self.device,
            batch_size=4096,
            use_amp=self.config.mixed_precision,
        )
        suite = DownstreamBenchmarkSuite(n_splits=3, random_state=42)

        auc_list = []
        ap_list = []
        snr_list = []
        var_list = []

        for item in self.val_scans:
            cscan = item["cscan"]
            gt_mask = item["gt_mask"]
            latents = extractor.extract_full_cscan(cscan)  # [270, 270, 2*D]

            # Mahalanobis Anomaly Detection (pure representation distance without probe training)
            anom = suite.compute_anomaly_detection_metrics(latents, gt_mask)
            auc_list.append(anom["auc_roc"])
            ap_list.append(anom["average_precision"])

            # Latent SNR & Sound Metal Noise Variance
            flat_l = latents.reshape(-1, latents.shape[-1])
            flat_m = (gt_mask.reshape(-1) > 0.5).astype(int)
            sound_pts = flat_l[flat_m == 0]
            defect_pts = flat_l[flat_m == 1]

            if len(sound_pts) > 0 and len(defect_pts) > 0:
                mu_s = np.mean(sound_pts, axis=0)
                mu_d = np.mean(defect_pts, axis=0)
                var_s = float(np.mean(np.var(sound_pts, axis=0)))
                snr = float(np.linalg.norm(mu_d - mu_s) / (np.sqrt(var_s) + 1e-6))
            else:
                snr = 0.0
                var_s = 0.0

            snr_list.append(snr)
            var_list.append(var_s)

        self.model.train(was_training)

        return {
            "val_auc": float(np.mean(auc_list)) if auc_list else 0.0,
            "val_ap": float(np.mean(ap_list)) if ap_list else 0.0,
            "val_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            "val_sound_var": float(np.mean(var_list)) if var_list else 0.0,
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

        # Linear Warmup + Cosine Annealing learning rate schedule
        warmup_epochs = max(0, getattr(self.config, "warmup_epochs", 3))
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr = self.config.min_lr + (self.config.lr - self.config.min_lr) * (epoch / warmup_epochs)
        else:
            anneal_epochs = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / anneal_epochs
            lr = self.config.min_lr + 0.5 * (self.config.lr - self.config.min_lr) * (
                1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # EMA momentum cosine schedule
        progress_total = epoch / max(1, total_epochs)
        ema_momentum = self.config.ema_momentum_end - (self.config.ema_momentum_end - self.config.ema_momentum_start) * (
            0.5 * (1.0 + math.cos(math.pi * progress_total))
        )

        pbar = tqdm(
            loader,
            desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
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

        batches_per_epoch = len(dataset) // self.config.batch_size
        cycle_info = (
            f" [Macro-Cycle Partitioning: K={dataset.num_partitions} sub-epochs/cycle]"
            if getattr(dataset, "use_partitioning", False)
            else ""
        )

        header = (
            f"\n=======================================================\n"
            f"Starting Cross-Condition JEPA Training{cycle_info}\n"
            f"Epochs: {self.start_epoch} -> {total_epochs} | Batches/Epoch: {batches_per_epoch:,} | Batch Size: {self.config.batch_size}\n"
            f"Samples/Epoch: {len(dataset):,} across 112 Directed Pairs\n"
            f"Device: {self.device} | AMP Mixed Precision: {self.config.mixed_precision}\n"
            f"=======================================================\n"
        )
        if self.logger:
            self.logger.info(header)
        else:
            print(header, flush=True)

        warmup_epochs = max(0, getattr(self.config, "warmup_epochs", 10))
        patience = getattr(self.config, "early_stopping_patience", 15)
        min_delta = getattr(self.config, "early_stopping_min_delta", 1e-4)
        metric_mode = getattr(self.config, "early_stopping_metric", "composite")
        eval_val_every = max(1, getattr(self.config, "eval_val_every", 1))

        patience_counter = 0
        best_loss = float("inf")
        best_val_auc = 0.0
        best_val_snr = 0.0

        ckpt_dir = self.logger.checkpoint_dir if self.logger else self.config.checkpoint_dir
        viz_dir = self.logger.visualizations_dir if self.logger else self.config.visualizations_dir
        best_ckpt_path = os.path.join(ckpt_dir, "best_cross_condition_jepa.pt")

        for epoch in range(self.start_epoch, total_epochs + 1):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

            epoch_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
                drop_last=True,
            )

            epoch_metrics = self.train_epoch(epoch_loader, epoch, total_epochs)

            # Evaluate held-out validation telemetry (Mahalanobis AUC/AP, Latent SNR, Sound Var)
            if self.val_scans and (epoch % eval_val_every == 0 or epoch == total_epochs):
                val_res = self.evaluate_validation_metrics()
                epoch_metrics.update(val_res)

            # Record history
            for k in self.history:
                if k in epoch_metrics:
                    self.history[k].append(epoch_metrics[k])

            # Formatted summary logging
            val_info = ""
            if "val_auc" in epoch_metrics:
                val_info = (
                    f" | Val AUC: {epoch_metrics['val_auc']:.4f}"
                    f" | SNR: {epoch_metrics['val_snr']:.2f}"
                    f" | SndVar: {epoch_metrics['val_sound_var']:.4f}"
                )

            summary_msg = (
                f"[Summary Epoch {epoch:02d}/{total_epochs:02d}] "
                f"Loss: {epoch_metrics['total_loss']:.4f} | "
                f"Pred: {epoch_metrics['pred_loss']:.4f}{val_info} | "
                f"R_eff: {epoch_metrics['effective_rank']:.1f}D | "
                f"LR: {epoch_metrics['lr']:.2e}"
            )
            if self.logger:
                self.logger.info(summary_msg)
                self.logger.log_epoch(epoch_metrics)
            else:
                print(summary_msg, flush=True)

            # Checkpoint best model & Early Stopping Evaluation
            curr_loss = epoch_metrics["total_loss"]
            curr_auc = epoch_metrics.get("val_auc", 0.0)
            curr_snr = epoch_metrics.get("val_snr", 0.0)

            improved = False
            improve_reason = ""

            if metric_mode == "val_auc":
                if curr_auc > (best_val_auc + 0.002):
                    improved = True
                    improve_reason = f"Val AUC improved: {best_val_auc:.4f} -> {curr_auc:.4f}"
                    best_val_auc = curr_auc
            elif metric_mode == "latent_snr":
                if curr_snr > (best_val_snr * 1.02):
                    improved = True
                    improve_reason = f"Latent SNR improved: {best_val_snr:.2f} -> {curr_snr:.2f}"
                    best_val_snr = curr_snr
            elif metric_mode == "loss":
                if curr_loss < (best_loss - min_delta):
                    improved = True
                    improve_reason = f"Loss improved: {best_loss:.4f} -> {curr_loss:.4f}"
                    best_loss = curr_loss
            else:  # "composite"
                # Multi-criterion representation health:
                # 1. Defect AUC improved significantly
                # 2. OR Latent SNR improved by >= 2% while AUC stable
                # 3. OR Loss dropped by min_delta while AUC stable
                if curr_auc > (best_val_auc + 0.002):
                    improved = True
                    improve_reason = f"Val AUC improved: {best_val_auc:.4f} -> {curr_auc:.4f}"
                    best_val_auc = max(best_val_auc, curr_auc)
                    best_val_snr = max(best_val_snr, curr_snr)
                    best_loss = min(best_loss, curr_loss)
                elif curr_snr > (best_val_snr * 1.02) and curr_auc >= (best_val_auc - 0.005):
                    improved = True
                    improve_reason = f"Latent SNR improved: {best_val_snr:.2f} -> {curr_snr:.2f} (AUC stable: {curr_auc:.4f})"
                    best_val_snr = max(best_val_snr, curr_snr)
                    best_loss = min(best_loss, curr_loss)
                elif curr_loss < (best_loss - min_delta) and curr_auc >= (best_val_auc - 0.010):
                    improved = True
                    improve_reason = f"Prediction loss improved: {best_loss:.4f} -> {curr_loss:.4f} (AUC stable: {curr_auc:.4f})"
                    best_loss = min(best_loss, curr_loss)

            if improved:
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "metrics": epoch_metrics,
                }, best_ckpt_path)
                if epoch > warmup_epochs:
                    save_msg = f"  [*] New best model saved (Epoch {epoch:02d} | {improve_reason})"
                    if self.logger:
                        self.logger.info(save_msg)
                    else:
                        print(save_msg, flush=True)
            else:
                if epoch > warmup_epochs and patience > 0:
                    patience_counter += 1
                    status_msg = (
                        f"  [!] Early stopping patience: {patience_counter}/{patience} "
                        f"(Best Val AUC: {best_val_auc:.4f}, Best SNR: {best_val_snr:.2f}, Best Loss: {best_loss:.4f})"
                    )
                    if self.logger:
                        self.logger.info(status_msg)
                    else:
                        print(status_msg, flush=True)

                    if patience_counter >= patience:
                        stop_msg = (
                            f"\n[Early Stopping Triggered at Epoch {epoch:02d}] "
                            f"Neither Val AUC, Latent SNR, nor Loss improved for {patience_counter} consecutive epochs "
                            f"(Best Val AUC: {best_val_auc:.4f}, Best SNR: {best_val_snr:.2f}, Best Loss: {best_loss:.4f}). "
                            f"Terminating training early."
                        )
                        if self.logger:
                            self.logger.info(stop_msg)
                        else:
                            print(stop_msg, flush=True)
                        break

            # Warmup Phase Transition: Reset best metrics and counter for post-warmup regime
            if epoch == warmup_epochs:
                best_loss = float("inf")
                best_val_auc = 0.0
                best_val_snr = 0.0
                patience_counter = 0
                reset_msg = (
                    f"\n>>> [Warmup Completed: {warmup_epochs} Epochs / 1 Full Macro-Cycle] "
                    f"Resetting best metrics (Val AUC, Latent SNR, Loss) and patience counter for official post-warmup tracking. <<<\n"
                )
                if self.logger:
                    self.logger.info(reset_msg)
                else:
                    print(reset_msg, flush=True)

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

        # Always save final training curve at conclusion of training
        curve_path = os.path.join(viz_dir, "training_curves.png")
        plot_training_curves(self.history, curve_path)

        fin_msg = f"\nTraining Complete! Best model saved to: {best_ckpt_path}"
        if self.logger:
            self.logger.info(fin_msg)
        else:
            print(fin_msg, flush=True)

        return best_ckpt_path
