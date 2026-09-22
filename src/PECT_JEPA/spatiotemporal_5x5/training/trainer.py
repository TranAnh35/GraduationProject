"""
Trainer for Unified 5x5 Spatiotemporal PECT-JEPA.
Features multi-tier logging: TensorBoard, structured CSVs, console/file logs, and WandB.
"""

import os
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
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
from ..evaluation.manifold_dimension import estimate_twonn_dimension
from .diagnostics import (
    compute_spatial_variogram,
    extract_attention_receptive_fields,
    compute_liftoff_invariance_diagnostic,
    compute_domain_similarity_matrix,
    plot_foundation_diagnostics_dashboard,
)
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

        # Foundation training diagnostics tracking
        self.diagnostics_dir = os.path.join(self.logger.run_dir if self.logger else config.experiment_dir, "training_diagnostics")
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        self.latest_val_pred = float("nan")
        self.latest_twonn_dim = 0.0
        self.latest_uniformity = float("nan")
        self.latest_val_batch = None

        target_resume = resume_checkpoint or getattr(config, "resume", None)
        if target_resume:
            self.resume_from_checkpoint(target_resume)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pred = 0.0
        total_liftoff = 0.0
        total_phase = 0.0
        total_var = 0.0
        total_cov = 0.0
        total_unif = 0.0
        total_norm = 0.0
        total_mean_norm = 0.0
        total_inter_cos = 0.0
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
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            momentum = self.momentum_scheduler.get_momentum(self.global_step)

            self.optimizer.zero_grad(set_to_none=True)
            with create_autocast(self.device.type, enabled=self.config.mixed_precision and self.device.type == "cuda"):
                loss_dict = self.model(x)
                loss = loss_dict["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                if self.logger:
                    self.logger.warning(f"NaN/Inf loss at global step {self.global_step}. Skipping batch.")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA target update
            self.model.update_target_encoder(momentum=momentum)

            loss_val = float(loss.item())
            pred_val = float(loss_dict["loss_pred"].item())
            liftoff_val = float(loss_dict.get("loss_liftoff", torch.tensor(0.0)).item())
            phase_val = float(loss_dict.get("loss_phase", torch.tensor(0.0)).item())
            var_val = float(loss_dict.get("loss_var", torch.tensor(0.0)).item())
            cov_val = float(loss_dict.get("loss_cov", torch.tensor(0.0)).item())
            unif_val = float(loss_dict.get("loss_unif", torch.tensor(0.0)).item())
            norm_val = float(loss_dict.get("loss_norm", torch.tensor(0.0)).item())
            mean_norm_val = float(loss_dict.get("mean_norm", torch.tensor(1.0)).item())

            # Inter-sample diversity monitoring (anti-collapse health metric)
            with torch.no_grad():
                H_tgt_step = loss_dict.get("H_tgt")
                if H_tgt_step is not None and H_tgt_step.shape[0] > 1:
                    H_tgt_p = F.normalize(H_tgt_step.detach().mean(dim=1).float(), p=2, dim=-1)
                    sim_b = torch.mm(H_tgt_p, H_tgt_p.t())
                    mask_b = ~torch.eye(sim_b.shape[0], dtype=torch.bool, device=sim_b.device)
                    inter_cos_val = float(sim_b[mask_b].mean().item())
                else:
                    inter_cos_val = 1.0

            total_loss += loss_val
            total_pred += pred_val
            total_liftoff += liftoff_val
            total_phase += phase_val
            total_var += var_val
            total_cov += cov_val
            total_unif += unif_val
            total_norm += norm_val
            total_mean_norm += mean_norm_val
            total_inter_cos += inter_cos_val
            n_batches += 1

            if self.logger:
                step_metrics = {
                    "loss": loss_val,
                    "loss_pred": pred_val,
                    "loss_liftoff": liftoff_val,
                    "loss_phase": phase_val,
                    "loss_unif": unif_val,
                    "loss_norm": norm_val,
                    "mean_norm": mean_norm_val,
                    "inter_cos": inter_cos_val,
                    "lr": lr,
                    "momentum": momentum,
                    "grad_norm": grad_norm,
                }
                if var_val > 0.0 or getattr(self.config, "var_weight", 0.0) > 0.0:
                    step_metrics["loss_var"] = var_val
                if cov_val > 0.0 or getattr(self.config, "cov_weight", 0.0) > 0.0:
                    step_metrics["loss_cov"] = cov_val
                self.logger.log_step(
                    step=self.global_step,
                    metrics=step_metrics,
                    epoch=self.current_epoch + 1
                )

            self.global_step += 1

            postfix = {
                "loss": f"{loss_val:.4f}",
                "pred": f"{pred_val:.4f}",
            }
            if var_val > 0.0:
                postfix["var"] = f"{var_val:.3f}"
            postfix.update({
                "norm": f"{mean_norm_val:.2f}",
                "cos": f"{inter_cos_val:.3f}",
                "lr": f"{lr:.1e}"
            })
            pbar.set_postfix(postfix)

        metrics = {
            "loss": total_loss / max(1, n_batches),
            "loss_pred": total_pred / max(1, n_batches),
            "loss_liftoff": total_liftoff / max(1, n_batches),
            "loss_phase": total_phase / max(1, n_batches),
            "loss_var": total_var / max(1, n_batches),
            "loss_cov": total_cov / max(1, n_batches),
            "loss_unif": total_unif / max(1, n_batches),
            "loss_norm": total_norm / max(1, n_batches),
            "mean_norm": total_mean_norm / max(1, n_batches),
            "inter_cos": total_inter_cos / max(1, n_batches),
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
    def run_training_diagnostics(self, epoch: int, sample_batch: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Runs physics-grounded self-supervised training diagnostics after each epoch:
        - Panel 1: Lift-Off Invariance Trajectory on sound metal (z1 vs z2 vs z3).
        - Panel 2: Empirical Latent Spatial Variogram gamma(r).
        - Panel 3: Multi-Physics Cross-Domain Similarity Matrix.
        - Panel 4: 2D Spatial Attention Receptive Fields (4 Heads on 5x5 grid).
        Saves dashboard to training_diagnostics/epoch_{epoch:02d}_diagnostics.png and logs to TensorBoard.
        """
        self.model.eval()
        diag_dir = os.path.join(self.logger.run_dir if self.logger else "experiments/5x5", "training_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        save_path = os.path.join(diag_dir, f"epoch_{epoch:02d}_diagnostics.png")

        try:
            # 1. Spatial Variogram & Attention Receptive Fields on sample grids
            if sample_batch is not None and "data" in sample_batch:
                sample_grids = sample_batch["data"].to(self.device)
            elif self.val_loader is not None and len(self.val_loader) > 0:
                sample_grids = next(iter(self.val_loader))["data"].to(self.device)
            elif self.train_loader is not None and len(self.train_loader) > 0:
                sample_grids = next(iter(self.train_loader))["data"].to(self.device)
            else:
                sample_grids = torch.randn(8, 5, 5, self.config.in_channels, device=self.device)

            unique_lags, gamma_r = compute_spatial_variogram(self.model, sample_grids, self.device)
            attn_maps = extract_attention_receptive_fields(self.model, sample_grids, self.device)

            # 2. Lift-off Invariance Diagnostic (grouped slices of batch)
            B_sub = sample_grids.shape[0]
            liftoff_feats = {}
            if B_sub >= 3:
                chunk = B_sub // 3
                z_all = self.model.extract_center_feature(sample_grids).detach().cpu().numpy()
                liftoff_feats["z1"] = z_all[:chunk]
                liftoff_feats["z2"] = z_all[chunk:2*chunk]
                liftoff_feats["z3"] = z_all[2*chunk:]
            liftoff_sims = compute_liftoff_invariance_diagnostic(liftoff_feats)

            # 3. Domain Cross-Similarity Matrix across available representation subsets
            half_b = max(1, B_sub // 2)
            z_c = self.model.extract_center_feature(sample_grids).detach().cpu().numpy()
            domain_feats = {
                "Domain_A": z_c[:half_b],
                "Domain_B": z_c[half_b:],
            }
            domain_names, domain_sim_matrix = compute_domain_similarity_matrix(domain_feats)

            # 4. Render and save dashboard figure
            val_pred = self.latest_val_pred if not np.isnan(self.latest_val_pred) else 0.0
            twonn_dim = self.latest_twonn_dim
            uniformity = self.latest_uniformity

            fig = plot_foundation_diagnostics_dashboard(
                epoch=epoch,
                val_loss_pred=val_pred,
                unique_lags=unique_lags,
                gamma_r=gamma_r,
                attn_maps=attn_maps,
                liftoff_sims=liftoff_sims,
                domain_names=domain_names,
                domain_sim_matrix=domain_sim_matrix,
                twonn_dim=twonn_dim,
                uniformity=uniformity,
                save_path=save_path,
                close_fig=False,
            )

            if self.logger and fig is not None:
                self.logger.log_figure("diagnostics/foundation_dashboard", fig, global_step=epoch)
                import matplotlib.pyplot as plt
                plt.close(fig)

            return {
                "dashboard_path": save_path,
                "sim_z1_z2": liftoff_sims.get("sim_z1_z2", 0.0),
                "sim_z1_z3": liftoff_sims.get("sim_z1_z3", 0.0),
            }
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[Diagnostics Epoch {epoch}] Failed: {e}")
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

            # Run Foundation Training Diagnostics (Physics-Grounded Dashboard)
            diag_metrics = None
            if getattr(self.config, "diagnostics_interval", 1) > 0:
                if (epoch + 1) % self.config.diagnostics_interval == 0:
                    diag_metrics = self.run_training_diagnostics(epoch=epoch + 1)

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

            diag_str = ""
            if diag_metrics:
                diag_str = f" | LiftOff-Sim: {diag_metrics.get('sim_z1_z2', 0.0):.2f}"

            pred_loss_str = f"(Pred: {train_metrics['loss_pred']:.4f})" if "loss_pred" in train_metrics else ""
            log_line = (
                f"[Epoch {epoch + 1:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} {pred_loss_str}"
                f"{val_str}{unif_str}{twonn_str}{diag_str} [{dt:.1f}s]"
            )
            if self.logger:
                self.logger.info(log_line)
                if diag_metrics and "dashboard_path" in diag_metrics:
                    self.logger.info(f"  --> [Foundation Diagnostics] Saved: {diag_metrics['dashboard_path']}")
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
                if diag_metrics:
                    if "sim_z1_z2" in diag_metrics:
                        epoch_data["sim_z1_z2"] = diag_metrics["sim_z1_z2"]
                self.logger.log_epoch(epoch=epoch + 1, metrics=epoch_data, step=self.global_step)

            # Checkpoint saving
            eff_rank = val_metrics.get("effective_rank") if val_metrics else None
            unif_val = val_metrics.get("uniformity") if val_metrics else None
            twonn_val = val_metrics.get("twonn_dim") if val_metrics else None
            val_loss = val_metrics.get("val_loss") if val_metrics else None
            val_loss_pred = val_metrics.get("val_loss_pred") if val_metrics else None
            latest_path = os.path.join(self.config.save_dir, "latest_model_5x5.pt")
            self.save_checkpoint(
                latest_path,
                val_loss=val_loss,
                val_loss_pred=val_loss_pred,
                effective_rank=eff_rank,
                uniformity=unif_val,
                twonn_dim=twonn_val,
            )

            # Early stopping & best checkpoint evaluation
            monitor_metric = getattr(self.config, "early_stopping_metric", "val_loss_pred")
            improved = False
            cur_metric = None

            if monitor_metric == "val_loss":
                cur_metric = val_loss
                if cur_metric is not None and not np.isnan(cur_metric) and cur_metric < self.best_val_loss:
                    self.best_val_loss = cur_metric
                    improved = True
                metric_info = f"val_loss: {cur_metric:.4f} (best: {self.best_val_loss:.4f})" if cur_metric is not None else "val_loss: N/A"

            else:  # default 'val_loss_pred'
                cur_metric = val_loss_pred
                if cur_metric is not None and not np.isnan(cur_metric) and cur_metric < self.best_val_loss_pred:
                    self.best_val_loss_pred = cur_metric
                    improved = True
                metric_info = f"val_loss_pred: {cur_metric:.4f} (best: {self.best_val_loss_pred:.4f})" if cur_metric is not None else "val_loss_pred: N/A"

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
                )
                msg_best = f"  --> Saved new best checkpoint (monitored {monitor_metric}): {best_path}"
                if self.logger:
                    self.logger.info(msg_best)
                else:
                    print(msg_best)
            else:
                patience = getattr(self.config, "early_stopping_patience", 0)
                early_stop_warmup = getattr(self.config, "early_stopping_warmup", 0)
                in_warmup = (epoch + 1) <= early_stop_warmup
                if patience > 0:
                    if in_warmup:
                        msg_patience = f"  --> Warmup phase (Epoch {epoch + 1}/{early_stop_warmup}): early stopping patience paused ({metric_info})"
                    else:
                        self.patience_counter += 1
                        msg_patience = f"  --> Early stopping patience: {self.patience_counter}/{patience} ({metric_info})"
                    if self.logger:
                        self.logger.info(msg_patience)
                    else:
                        print(msg_patience)
                    if not in_warmup and self.patience_counter >= patience:
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
