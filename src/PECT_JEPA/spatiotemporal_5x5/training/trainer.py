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
from ..evaluation.anomaly_detection import (
    AnomalyDetector5x5,
    compute_anomaly_metrics,
    plot_anomaly_heatmap_5x5,
    plot_latent_representation_quality,
    evaluate_anomaly_ground_truth,
)
from ..evaluation.manifold_dimension import estimate_twonn_dimension
from ..evaluation.cscan_extractor import extract_full_cscan_map, load_cscan_from_tdms
from .optimizer import build_optimizer_5x5, WarmupCosineLRScheduler5x5, MomentumScheduler5x5


def compute_hypersphere_uniformity(features: np.ndarray, t: float = 2.0) -> float:
    """
    Computes Hypersphere Uniformity (Wang & Isola, ICML 2020).
    Measures feature dispersion on the unit sphere:
        L_unif = log( E_{u,v} [ exp(-t * ||u - v||^2) ] )
    Collapse -> L_unif -> 0.0
    Healthy representation -> L_unif < -2.0 (more negative = better).
    """
    if len(features) < 2:
        return 0.0
    norms = np.linalg.norm(features, axis=-1, keepdims=True) + 1e-12
    u = features / norms
    sim = np.dot(u, u.T)
    dist_sq = np.clip(2.0 - 2.0 * sim, a_min=0.0, a_max=4.0)
    np.fill_diagonal(dist_sq, np.inf)
    valid = dist_sq < np.inf
    loss = float(np.log(np.mean(np.exp(-t * dist_sq[valid])) + 1e-12))
    return loss


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
        probe_file: Optional[str] = None,
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

        total_steps = config.epochs * max(1, len(train_loader))
        warmup_steps = config.warmup_epochs * max(1, len(train_loader))

        self.lr_scheduler = WarmupCosineLRScheduler5x5(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
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
        self.best_val_loss_pred = float("inf")
        self.best_probe_cnr = -float("inf")
        self.best_probe_gt_auc = -float("inf")
        self.patience_counter = 0

        if config.save_dir is None:
            base_dir = self.logger.run_dir if self.logger else os.path.join(config.log_dir, config.exp_name)
            config.save_dir = os.path.join(base_dir, "checkpoints")

        os.makedirs(config.save_dir, exist_ok=True)
        if config.log_dir:
            os.makedirs(config.log_dir, exist_ok=True)

        # Pre-load optional validation TDMS scan for epoch-by-epoch downstream defect probing
        self.probe_file = probe_file or getattr(config, "probe_file", None)
        self.probe_grid = None
        self.probe_fname = None
        self.probe_gt_mask = None
        if self.probe_file and os.path.isfile(self.probe_file):
            try:
                self.probe_grid = load_cscan_from_tdms(
                    file_path=self.probe_file,
                    time_samples=config.time_samples,
                    temporal_samples=config.temporal_samples,
                    resample_mode=config.resample_mode,
                    normalization=config.normalization,
                    raster_correction=config.raster_correction,
                    crop_border=config.crop_border,
                    apply_lowpass=getattr(config, "apply_lowpass", True),
                    lowpass_cutoff=getattr(config, "lowpass_cutoff", 2500.0),
                    lowpass_order=getattr(config, "lowpass_order", 4),
                )
                self.probe_fname = os.path.splitext(os.path.basename(self.probe_file))[0]
                if self.logger:
                    self.logger.info(
                        f"[Probe] Pre-loaded validation C-scan for downstream probing: {self.probe_fname} "
                        f"(grid: {self.probe_grid.shape[0]}x{self.probe_grid.shape[1]}, C={self.probe_grid.shape[2]})"
                    )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Probe] Could not pre-load probe file '{self.probe_file}': {e}")

            # Locate and load authoritative CAD Ground Truth mask
            if self.probe_file:
                try:
                    from ..data.ground_truth import get_ground_truth_manager
                    gt_mgr = get_ground_truth_manager(data_dir=config.data_dir)
                    self.probe_gt_mask = gt_mgr.get_ground_truth_mask_for_file(self.probe_file, aligned_scan=True)
                    if self.logger and self.probe_gt_mask is not None:
                        self.logger.info(
                            f"[Probe GT] Loaded authoritative CAD mask for '{self.probe_fname}': "
                            f"(shape: {self.probe_gt_mask.shape}, defects: {int(np.sum(self.probe_gt_mask == 1))})"
                        )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[Probe GT] Could not load authoritative mask via manager: {e}")

            if self.probe_gt_mask is None:
                fname_lower = os.path.basename(self.probe_file).lower()
                specimen_key = None
                if "corosion" in fname_lower or "corrosion" in fname_lower:
                    specimen_key = "corrosion"
                elif "rivet_v1" in fname_lower or "rivet1" in fname_lower:
                    specimen_key = "rivet_v1"
                elif "rivet_v2" in fname_lower or "rivet2" in fname_lower or "mixed" in fname_lower:
                    specimen_key = "rivet_v2"

                if specimen_key:
                    candidates = [
                        os.path.join(config.data_dir, "ground_truth", specimen_key, f"{specimen_key}_gt_mask.npy"),
                        os.path.join("data", "ground_truth", specimen_key, f"{specimen_key}_gt_mask.npy"),
                    ]
                    for c_gt in candidates:
                        if os.path.isfile(c_gt):
                            try:
                                self.probe_gt_mask = np.load(c_gt)
                                if self.logger:
                                    self.logger.info(
                                        f"[Probe GT] Loaded ground-truth mask fallback for '{specimen_key}': {c_gt} "
                                        f"(mask shape: {self.probe_gt_mask.shape}, defects: {int(np.sum(self.probe_gt_mask == 1))})"
                                    )
                                break
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"[Probe GT] Could not load mask '{c_gt}': {e}")

        target_resume = resume_checkpoint or getattr(config, "resume", None)
        if target_resume:
            self.resume_from_checkpoint(target_resume)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pred = 0.0
        total_var = 0.0
        total_cov = 0.0
        total_rank_barrier = 0.0
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
            rank_barrier_val = float(loss_dict.get("loss_rank_barrier", torch.tensor(0.0)).item())

            total_loss += loss_val
            total_pred += pred_val
            total_var += var_val
            total_cov += cov_val
            total_rank_barrier += rank_barrier_val
            n_batches += 1

            if self.logger:
                self.logger.log_step(
                    step=self.global_step,
                    metrics={
                        "loss": loss_val,
                        "loss_pred": pred_val,
                        "loss_var": var_val,
                        "loss_cov": cov_val,
                        "loss_rank_barrier": rank_barrier_val,
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
            "loss_rank_barrier": total_rank_barrier / max(1, n_batches),
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

        eff_rank = None
        uniformity = float("nan")
        twonn_dim = 0.0
        if val_features:
            feats = np.concatenate(val_features, axis=0)
            feats = feats[~np.isnan(feats).any(axis=1)]
            if len(feats) >= 2:
                uniformity = compute_hypersphere_uniformity(feats)
                try:
                    twonn_dim = float(estimate_twonn_dimension(feats, subsample=2000))
                except Exception:
                    twonn_dim = 0.0
                if getattr(self.config, "track_effective_rank", False):
                    eff_rank = compute_effective_rank(feats)

        val_loss = (total_loss / n_batches) if n_batches > 0 else float("nan")
        val_loss_pred = (total_pred / n_batches) if n_batches > 0 else float("nan")

        res = {
            "val_loss": val_loss,
            "val_loss_pred": val_loss_pred,
            "uniformity": uniformity,
            "twonn_dim": twonn_dim,
        }
        if eff_rank is not None:
            res["effective_rank"] = eff_rank
        return res

    @torch.no_grad()
    def run_downstream_probe(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        Runs full C-scan feature extraction and unsupervised anomaly detection on self.probe_grid,
        plots the 2D anomaly heatmap, logs to TensorBoard, and returns defect metrics (CNR, etc.).
        """
        if self.probe_grid is None:
            return None
        self.model.eval()
        try:
            # 1. Extract 1-to-1 C-Scan feature map
            feature_map = extract_full_cscan_map(
                model=self.model,
                full_cscan_3d=self.probe_grid,
                batch_size=512,
                device=self.device.type,
                show_pbar=False,
            )

            # 2. Fit unsupervised anomaly detector on representations
            detector = AnomalyDetector5x5(n_clusters=2, detrend=True)
            detector.fit(feature_map)
            score_map = detector.score_map(feature_map, detrend=True)
            raw_score_map = detector.score_map(feature_map, detrend=False)

            # 3. Compute quantitative defect contrast metrics & ground-truth AUC (True Label-based when mask present)
            metrics = compute_anomaly_metrics(score_map, gt_mask=self.probe_gt_mask)
            cnr = metrics["contrast_ratio_cnr"]
            raw_metrics = compute_anomaly_metrics(raw_score_map, gt_mask=self.probe_gt_mask)
            raw_cnr = float(raw_metrics["contrast_ratio_cnr"])
            metrics["raw_cnr"] = raw_cnr

            probe_gt_auc = metrics.get("auc_roc")
            probe_gt_ap = metrics.get("average_precision")
            probe_gt_f1 = metrics.get("best_f1")
            metrics["probe_gt_auc"] = probe_gt_auc
            metrics["probe_gt_ap"] = probe_gt_ap
            metrics["probe_gt_f1"] = probe_gt_f1

            # 4. Save heatmap images to disk
            probe_dir = os.path.join(self.logger.run_dir if self.logger else "experiments/5x5", "probe_heatmaps")
            os.makedirs(probe_dir, exist_ok=True)
            if probe_gt_auc is not None:
                heatmap_path = os.path.join(probe_dir, f"epoch_{epoch:02d}_auc_{probe_gt_auc:.4f}.png")
                title = f"Epoch {epoch:02d} | Probe: {self.probe_fname} | GT AUC: {probe_gt_auc:.4f} (AP: {probe_gt_ap:.4f}, CNR: {cnr:.2f})"
            else:
                heatmap_path = os.path.join(probe_dir, f"epoch_{epoch:02d}_cnr_{cnr:.2f}.png")
                title = f"Epoch {epoch:02d} | Probe: {self.probe_fname} | CNR: {cnr:.2f} (Raw CNR: {raw_cnr:.2f})"
            fig = plot_anomaly_heatmap_5x5(score_map, save_path=heatmap_path, title=title, close_fig=False)

            # Also save raw latent anomaly heatmap for transparent representation inspection
            raw_heatmap_path = os.path.join(probe_dir, f"epoch_{epoch:02d}_raw_latent_cnr_{raw_cnr:.2f}.png")
            raw_title = f"Epoch {epoch:02d} (Raw Latent) | Probe: {self.probe_fname} | Raw CNR: {raw_cnr:.2f}"
            fig_raw = plot_anomaly_heatmap_5x5(raw_score_map, save_path=raw_heatmap_path, title=raw_title, close_fig=False)
            if fig_raw is not None:
                import matplotlib.pyplot as plt
                plt.close(fig_raw)

            # 5. Generate Latent Representation Quality Visualizer (PCA-RGB + Hypersphere Angular Distance + CAD Overlay)
            lq_path = os.path.join(probe_dir, f"epoch_{epoch:02d}_latent_quality.png")
            lq_res = plot_latent_representation_quality(
                feature_map=feature_map,
                gt_mask=self.probe_gt_mask,
                save_path=lq_path,
                title_prefix=f"Epoch {epoch:02d} | Probe: {self.probe_fname}",
                close_fig=False,
            )
            if self.logger and lq_res.get("fig") is not None:
                self.logger.log_figure("representation/latent_quality", lq_res["fig"], global_step=epoch)
                import matplotlib.pyplot as plt
                plt.close(lq_res["fig"])

            if lq_res.get("angular_cnr") is not None and not np.isnan(lq_res["angular_cnr"]):
                metrics["latent_angular_cnr"] = float(lq_res["angular_cnr"])
            if lq_res.get("angular_auc") is not None:
                metrics["latent_angular_auc"] = float(lq_res["angular_auc"])
            if lq_res.get("total_3pc_variance") is not None:
                metrics["latent_total_3pc_variance"] = float(lq_res["total_3pc_variance"])

            # 6. Log figure to TensorBoard & WandB
            if self.logger and fig is not None:
                self.logger.log_figure("downstream_probe/anomaly_heatmap", fig, global_step=epoch)
                import matplotlib.pyplot as plt
                plt.close(fig)

            return metrics
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[Probe Epoch {epoch}] Downstream probing failed: {e}")
            return None

    def save_checkpoint(
        self,
        path: str,
        val_loss: Optional[float] = None,
        val_loss_pred: Optional[float] = None,
        effective_rank: Optional[float] = None,
        uniformity: Optional[float] = None,
        twonn_dim: Optional[float] = None,
        probe_gt_auc: Optional[float] = None,
    ):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if hasattr(self, "scaler") else None,
            "val_loss": val_loss,
            "val_loss_pred": val_loss_pred,
            "best_val_loss": self.best_val_loss,
            "best_val_loss_pred": getattr(self, "best_val_loss_pred", float("inf")),
            "best_probe_cnr": getattr(self, "best_probe_cnr", -float("inf")),
            "best_probe_gt_auc": getattr(self, "best_probe_gt_auc", -float("inf")),
            "effective_rank": effective_rank,
            "uniformity": uniformity,
            "twonn_dim": twonn_dim,
            "probe_gt_auc": probe_gt_auc,
            "config": self.config.to_dict(),
        }
        torch.save(ckpt, path)

    def resume_from_checkpoint(self, checkpoint_path_or_keyword: str) -> bool:
        """
        Resumes model, optimizer, scaler, step, epoch, and best metrics from a checkpoint file
        or keyword ('auto', 'latest', 'best', 'best_probe').
        """
        keyword = str(checkpoint_path_or_keyword).strip()
        resolved_path = None

        # Checkpoint directories for this unified experiment
        candidate_dirs = []
        if hasattr(self, "config") and getattr(self.config, "save_dir", None):
            candidate_dirs.append(self.config.save_dir)
        if hasattr(self, "logger") and getattr(self.logger, "run_dir", None):
            candidate_dirs.append(os.path.join(self.logger.run_dir, "checkpoints"))
            candidate_dirs.append(self.logger.run_dir)

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
        elif keyword.lower() in ("best_probe", "probe"):
            for d in search_dirs:
                c_probe = os.path.join(d, "best_probe_model_5x5.pt")
                if os.path.isfile(c_probe):
                    resolved_path = c_probe
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

        # 5. Best Val Loss, Best Val Pred Loss, and Best Probe CNR
        self.best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        self.best_val_loss_pred = checkpoint.get("best_val_loss_pred", checkpoint.get("val_loss_pred", float("inf")))
        self.best_probe_cnr = checkpoint.get("best_probe_cnr", -float("inf"))

        # 6. Schedulers: synchronize LR to global_step
        current_lr = self.lr_scheduler.step(self.global_step)

        summary_msg = (
            f"[Resume] Successfully restored checkpoint! Resuming at Epoch {self.start_epoch + 1}/{self.config.epochs} "
            f"(Global Step: {self.global_step}, Best Val Pred: {self.best_val_loss_pred:.4f}, Best Val Loss: {self.best_val_loss:.4f}, LR: {current_lr:.2e})"
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

            # Downstream defect probing on validation scan
            probe_metrics = None
            if self.probe_grid is not None and getattr(self.config, "probe_interval", 1) > 0:
                if (epoch + 1) % self.config.probe_interval == 0:
                    probe_metrics = self.run_downstream_probe(epoch=epoch + 1)
                    probe_improved = False
                    msg_probe = ""
                    if probe_metrics and "probe_gt_auc" in probe_metrics and probe_metrics["probe_gt_auc"] is not None:
                        auc_val = float(probe_metrics["probe_gt_auc"])
                        if not np.isnan(auc_val) and auc_val > self.best_probe_gt_auc:
                            self.best_probe_gt_auc = auc_val
                            probe_improved = True
                            msg_probe = f"  --> Saved new best downstream probe checkpoint (GT AUC: {auc_val:.4f}, AP: {probe_metrics['probe_gt_ap']:.4f})"
                    elif probe_metrics and "contrast_ratio_cnr" in probe_metrics:
                        cnr = float(probe_metrics["contrast_ratio_cnr"])
                        if not np.isnan(cnr) and cnr > self.best_probe_cnr:
                            self.best_probe_cnr = cnr
                            probe_improved = True
                            msg_probe = f"  --> Saved new best downstream probe checkpoint (CNR: {cnr:.2f})"

                    if probe_improved:
                        eff_rank = val_metrics.get("effective_rank") if val_metrics else None
                        unif_val = val_metrics.get("uniformity") if val_metrics else None
                        twonn_val = val_metrics.get("twonn_dim") if val_metrics else None
                        best_probe_path = os.path.join(self.config.save_dir, "best_probe_model_5x5.pt")
                        self.save_checkpoint(
                            best_probe_path,
                            val_loss=val_metrics.get("val_loss") if val_metrics else None,
                            val_loss_pred=val_metrics.get("val_loss_pred") if val_metrics else None,
                            effective_rank=eff_rank,
                            uniformity=unif_val,
                            twonn_dim=twonn_val,
                            probe_gt_auc=probe_metrics.get("probe_gt_auc") if probe_metrics else None,
                        )
                        msg_full = f"{msg_probe}: {best_probe_path}"
                        if self.logger:
                            self.logger.info(msg_full)
                        else:
                            print(msg_full)

            val_str = ""
            if val_metrics:
                v_pred = val_metrics.get("val_loss_pred")
                if v_pred is not None and not np.isnan(v_pred):
                    val_str = f" | Val Pred: {v_pred:.4f}"
                elif val_metrics.get("val_loss") is not None:
                    val_str = f" | Val Loss: {val_metrics['val_loss']:.4f}"

            unif_val = val_metrics.get("uniformity") if val_metrics else None
            unif_str = f" | Unif: {unif_val:.2f}" if unif_val is not None and not np.isnan(unif_val) else ""

            twonn_val = val_metrics.get("twonn_dim") if val_metrics else None
            twonn_str = f" | Two-NN: {twonn_val:.1f}D" if twonn_val is not None and twonn_val > 0 else ""

            probe_str = ""
            if probe_metrics:
                if "probe_gt_auc" in probe_metrics and probe_metrics["probe_gt_auc"] is not None:
                    probe_str = f" | Probe GT-AUC: {probe_metrics['probe_gt_auc']:.4f} (AP: {probe_metrics['probe_gt_ap']:.4f})"
                elif "contrast_ratio_cnr" in probe_metrics:
                    probe_str = f" | Probe CNR: {probe_metrics['contrast_ratio_cnr']:.2f}"

            pred_loss_str = f"(Pred: {train_metrics['loss_pred']:.4f})" if "loss_pred" in train_metrics else ""
            log_line = (
                f"[Epoch {epoch + 1:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} {pred_loss_str}"
                f"{val_str}{unif_str}{twonn_str}{probe_str} [{dt:.1f}s]"
            )
            if self.logger:
                self.logger.info(log_line)
                if probe_metrics:
                    if "probe_gt_auc" in probe_metrics and probe_metrics["probe_gt_auc"] is not None:
                        self.logger.info(f"  --> [Probe Heatmap] Saved: probe_heatmaps/epoch_{epoch + 1:02d}_auc_{probe_metrics['probe_gt_auc']:.4f}.png")
                    elif "contrast_ratio_cnr" in probe_metrics:
                        self.logger.info(f"  --> [Probe Heatmap] Saved: probe_heatmaps/epoch_{epoch + 1:02d}_cnr_{probe_metrics['contrast_ratio_cnr']:.2f}.png")
                    self.logger.info(f"  --> [Latent Quality Figure] Saved: probe_heatmaps/epoch_{epoch + 1:02d}_latent_quality.png")
            else:
                print(log_line)

            # Log epoch metrics to TensorBoard & CSV
            if self.logger:
                epoch_data = {
                    "train_loss": train_metrics["loss"],
                    "lr": self.lr_scheduler.get_lr(self.global_step),
                    "time_sec": dt,
                }
                if val_metrics:
                    if "val_loss" in val_metrics and not np.isnan(val_metrics["val_loss"]):
                        epoch_data["val_loss"] = val_metrics["val_loss"]
                    if "val_loss_pred" in val_metrics and not np.isnan(val_metrics["val_loss_pred"]):
                        epoch_data["val_loss_pred"] = val_metrics["val_loss_pred"]
                    if "uniformity" in val_metrics and not np.isnan(val_metrics["uniformity"]):
                        epoch_data["uniformity"] = val_metrics["uniformity"]
                    if "twonn_dim" in val_metrics and val_metrics["twonn_dim"] > 0:
                        epoch_data["twonn_dim"] = val_metrics["twonn_dim"]
                    if "effective_rank" in val_metrics:
                        epoch_data["effective_rank"] = val_metrics["effective_rank"]
                if probe_metrics:
                    if "probe_gt_auc" in probe_metrics and probe_metrics["probe_gt_auc"] is not None:
                        epoch_data["probe_gt_auc"] = probe_metrics["probe_gt_auc"]
                    if "probe_gt_ap" in probe_metrics and probe_metrics["probe_gt_ap"] is not None:
                        epoch_data["probe_gt_ap"] = probe_metrics["probe_gt_ap"]
                    if "contrast_ratio_cnr" in probe_metrics:
                        epoch_data["probe_cnr"] = probe_metrics["contrast_ratio_cnr"]
                    if "latent_angular_auc" in probe_metrics and probe_metrics["latent_angular_auc"] is not None:
                        epoch_data["latent_angular_auc"] = probe_metrics["latent_angular_auc"]
                    if "latent_angular_cnr" in probe_metrics and not np.isnan(probe_metrics["latent_angular_cnr"]):
                        epoch_data["latent_angular_cnr"] = probe_metrics["latent_angular_cnr"]
                    if "latent_total_3pc_variance" in probe_metrics:
                        epoch_data["latent_total_3pc_variance"] = probe_metrics["latent_total_3pc_variance"]
                self.logger.log_epoch(epoch=epoch + 1, metrics=epoch_data, step=self.global_step)

            # Checkpoint saving
            eff_rank = val_metrics.get("effective_rank") if val_metrics else None
            unif_val = val_metrics.get("uniformity") if val_metrics else None
            twonn_val = val_metrics.get("twonn_dim") if val_metrics else None
            val_loss = val_metrics.get("val_loss") if val_metrics else None
            val_loss_pred = val_metrics.get("val_loss_pred") if val_metrics else None
            probe_gt_auc_val = probe_metrics.get("probe_gt_auc") if probe_metrics else None
            latest_path = os.path.join(self.config.save_dir, "latest_model_5x5.pt")
            self.save_checkpoint(
                latest_path,
                val_loss=val_loss,
                val_loss_pred=val_loss_pred,
                effective_rank=eff_rank,
                uniformity=unif_val,
                twonn_dim=twonn_val,
                probe_gt_auc=probe_gt_auc_val,
            )

            # Early stopping & best checkpoint evaluation
            monitor_metric = getattr(self.config, "early_stopping_metric", "val_loss_pred")
            improved = False
            metric_info = ""

            if monitor_metric == "val_loss_pred":
                cur_metric = val_loss_pred if val_loss_pred is not None else train_metrics["loss_pred"]
                if not np.isnan(cur_metric) and cur_metric < self.best_val_loss_pred:
                    self.best_val_loss_pred = cur_metric
                    improved = True
                metric_info = f"val_loss_pred: {cur_metric:.4f} (best: {self.best_val_loss_pred:.4f})"
            elif monitor_metric in ("probe_gt_auc", "gt_auc"):
                cur_metric = float(probe_metrics["probe_gt_auc"]) if (probe_metrics and "probe_gt_auc" in probe_metrics and probe_metrics["probe_gt_auc"] is not None) else float("-inf")
                if not np.isnan(cur_metric) and cur_metric > getattr(self, "best_probe_gt_auc", -float("inf")):
                    self.best_probe_gt_auc = cur_metric
                    improved = True
                metric_info = f"probe_gt_auc: {cur_metric:.4f} (best: {self.best_probe_gt_auc:.4f})"
            elif monitor_metric == "val_loss":
                cur_metric = val_loss if val_loss is not None else train_metrics["loss"]
                if not np.isnan(cur_metric) and cur_metric < self.best_val_loss:
                    self.best_val_loss = cur_metric
                    improved = True
                metric_info = f"val_loss: {cur_metric:.4f} (best: {self.best_val_loss:.4f})"
            elif monitor_metric == "probe_cnr":
                cur_metric = float(probe_metrics["contrast_ratio_cnr"]) if (probe_metrics and "contrast_ratio_cnr" in probe_metrics) else float("nan")
                if not np.isnan(cur_metric) and cur_metric > self.best_probe_cnr:
                    improved = True
                metric_info = f"probe_cnr: {cur_metric:.2f} (best: {self.best_probe_cnr:.2f})"
            else:
                cur_metric = val_loss_pred if val_loss_pred is not None else train_metrics["loss_pred"]
                if not np.isnan(cur_metric) and cur_metric < self.best_val_loss_pred:
                    self.best_val_loss_pred = cur_metric
                    improved = True
                metric_info = f"val_loss_pred: {cur_metric:.4f} (best: {self.best_val_loss_pred:.4f})"

            # Keep best_val_loss and best_val_loss_pred updated tracking values
            if val_loss is not None and not np.isnan(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if val_loss_pred is not None and not np.isnan(val_loss_pred) and val_loss_pred < self.best_val_loss_pred:
                self.best_val_loss_pred = val_loss_pred

            if improved:
                self.patience_counter = 0
                best_path = os.path.join(self.config.save_dir, "best_model_5x5.pt")
                self.save_checkpoint(
                    best_path,
                    val_loss=val_loss,
                    val_loss_pred=val_loss_pred,
                    effective_rank=eff_rank,
                    uniformity=unif_val,
                    twonn_dim=twonn_val,
                    probe_gt_auc=probe_gt_auc_val,
                )
                msg_best = f"  --> Saved new best checkpoint (monitored {monitor_metric}): {best_path}"
                if self.logger:
                    self.logger.info(msg_best)
                else:
                    print(msg_best)
            else:
                patience = getattr(self.config, "early_stopping_patience", 0)
                if patience > 0:
                    self.patience_counter += 1
                    msg_patience = f"  --> Early stopping patience: {self.patience_counter}/{patience} ({metric_info})"
                    if self.logger:
                        self.logger.info(msg_patience)
                    else:
                        print(msg_patience)
                    if self.patience_counter >= patience:
                        msg_stop = f"[Early Stopping] Monitored metric '{monitor_metric}' did not improve for {patience} consecutive epochs ({metric_info}). Stopping training at Epoch {epoch + 1}."
                        if self.logger:
                            self.logger.info(msg_stop)
                        else:
                            print(msg_stop)
                        break

        if self.logger:
            self.logger.info("--- 5x5 PECT-JEPA Training Complete ---")
            self.logger.close()
        else:
            print("--- 5x5 PECT-JEPA Training Complete ---")
