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
from ..data.topologies import get_spatial_topology_offsets as _get_offsets
from .diagnostics import compute_svd_diagnostics, compute_latent_snr


def _ramp_scale(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    """P0-1 curriculum multiplier in [0, 1] for auxiliary representation
    losses (L_inv / L_xcov). Stays 0 before start_epoch, then cosine-ramps to
    1 over ramp_epochs (ramp <= 0 means legacy constant-on from start_epoch).
    Lets the predictor path build features first; the direct-on-encoder terms
    only polish invariance afterwards instead of hijacking early training.
    """
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    p = min(1.0, max(0.0, (epoch - start_epoch) / ramp_epochs))
    return 0.5 * (1.0 - math.cos(math.pi * p))


def _val_divergence_step(val_loss: float, best_min: float, counter: int,
                         factor: float) -> tuple:
    """One update of the train/val-divergence guard.

    Tracks the running best val recon loss; counts consecutive evals where val
    exceeds best * factor (default 1.5x). A rising val while train stays
    flat/falling = shortcut or overfit (exp49: val 0.0275 -> 0.06 while train
    pred sat at ~0.03). Returns (new_best_min, new_counter); the caller stops
    when counter reaches diverge_patience.
    NOTE: read R_eff only jointly with SndVar/SNR/AUC — a high R_eff with
    collapsed SndVar is an isotropic noise floor, not learned rank (exp49:
    R_eff 15 with SndVar -> 0, AUC ~= chance).
    """
    if val_loss > best_min * factor:
        return best_min, counter + 1
    return min(best_min, val_loss), 0


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

        # Runtime: autotune cudnn kernels once + allow TF32/high-precision
        # matmuls. Numerics of the AMP fp16 training path are unchanged.
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Hoisted validation helpers (built once, not re-allocated every epoch)
        self._val_extractor: Optional[CScanFeatureExtractor] = None
        self._val_suite: Optional[DownstreamBenchmarkSuite] = None

        # 3. AMP Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")

        # 4. History tracking
        self.start_epoch = 1
        self._resumed_patience = 0
        self.history: Dict[str, list] = {
            "epoch": [],
            "total_loss": [],
            "pred_loss": [],
            "var_loss": [],
            "cov_loss": [],
            "inv_loss": [],
            "xcov_loss": [],
            "zc_cond_loss": [],
            "effective_rank": [],
            "lr": [],
            "val_loss": [],
            "val_auc": [],
            "val_ap": [],
            "val_snr": [],
            "val_sound_var": [],
        }

        # 5. Resume training if checkpoint specified
        if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
            self._load_checkpoint(config.resume_checkpoint)

    @torch.no_grad()
    def _compute_val_reconstruction_loss(self, num_patches: int = 512) -> float:
        """Self-conditioned JEPA reconstruction loss on held-out val scans.

        No cross-condition pairs exist at eval time, so each val scan
        predicts itself: context probes -> core target probes within the
        same scan (A == B self-pair). Uses the same scattering L1 objective
        as training in disentangled mode. Returns mean over val scans.
        """
        if not self.val_scans:
            return 0.0
        was_training = self.model.training
        self.model.eval()
        rng = np.random.RandomState(12345)
        losses = []
        for item in self.val_scans:
            cscan = item["cscan"]  # [H, W, C]
            H, W, C = cscan.shape
            _radii = tuple(getattr(self.config, "star_radii", (1, 3, 7)))
            # Topology offsets: idx 0..8 core target, 9..24 outer context
            offsets = _get_offsets("concentric_star", _radii, 5)
            pad = int(max(1, np.max(np.abs(offsets))))
            padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            rr = rng.randint(0, H, size=num_patches) + pad
            cc = rng.randint(0, W, size=num_patches) + pad
            # Gather 25 star probes per patch: [N, 25, C]
            r_idx = rr[:, None] + offsets[None, :, 0]
            c_idx = cc[:, None] + offsets[None, :, 1]
            patch25 = padded[r_idx, c_idx]  # [N, 25, C]
            patch55 = patch25.reshape(num_patches, 5, 5, C)
            x = torch.from_numpy(patch55).to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                out = self.model(x_a=x, x_b=x, ref_b=x[:1].expand(x.shape[0], -1, -1, -1) if x.shape[0] > 0 else x)
                pl = out.get("pred_loss", None)
                losses.append(float(pl.item()) if pl is not None else 0.0)
        self.model.train(was_training)
        return float(np.mean(losses)) if losses else 0.0

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

        if self._val_extractor is None:
            self._val_extractor = CScanFeatureExtractor(
                model=self.model,
                device=self.device,
                batch_size=8192,
                use_amp=self.config.mixed_precision,
            )
            self._val_suite = DownstreamBenchmarkSuite(n_splits=3, random_state=42, device=self.device)
        extractor = self._val_extractor
        suite = self._val_suite

        auc_list = []
        ap_list = []
        snr_list = []
        var_list = []
        vloss_list = []

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

        # Self-conditioned JEPA reconstruction loss on held-out val scans
        try:
            vloss_list.append(self._compute_val_reconstruction_loss())
        except Exception as e:
            print(f"  [WARN] val_loss computation skipped: {type(e).__name__}: {e}", flush=True)

        self.model.train(was_training)

        return {
            "val_auc": float(np.mean(auc_list)) if auc_list else 0.0,
            "val_ap": float(np.mean(ap_list)) if ap_list else 0.0,
            "val_snr": float(np.mean(snr_list)) if snr_list else 0.0,
            "val_sound_var": float(np.mean(var_list)) if var_list else 0.0,
            "val_loss": float(np.mean(vloss_list)) if vloss_list else 0.0,
        }

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
        # AMP scaler state (loss scale / growth tracker); absent in old ckpts
        if "scaler_state_dict" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self._resumed_patience = int(ckpt.get("patience_counter", 0))
        if "history" in ckpt:
            self.history = ckpt["history"]
            # Backward compat: pad any missing/short key with zeros so resumed
            # history lists stay aligned (old ckpts lack val_loss, inv_loss,
            # xcov_loss, val_ap, ...). Padding uses total_loss length as ref.
            n = len(self.history.get("total_loss", []))
            for k, v in self.history.items():
                if isinstance(v, list) and len(v) < n:
                    v += [0.0] * (n - len(v))
        else:
            self._resumed_patience = 0

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_batches = len(loader)

        # Runtime: accumulate loss sums as GPU-side float64 scalars and sync to
        # CPU only once per LOG_EVERY batches for the progress bar (plus once
        # at epoch end). The old code called .item() 7x per batch, forcing a
        # full CUDA pipeline stall on every one of the ~3.2k steps/epoch.
        LOG_EVERY = 10
        acc_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_pred = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_carrier = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_var = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_cov = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_inv = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_xcov = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_zc = torch.zeros((), device=self.device, dtype=torch.float64)
        disp = {"loss": float("nan"), "pred": float("nan"), "car": float("nan"),
                "var": float("nan"), "inv": float("nan")}

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

        # P0-1 curriculum scales for this epoch (1.0 = legacy unscheduled path)
        inv_scale = _ramp_scale(epoch, getattr(self.config, "inv_start_epoch", 1),
                                getattr(self.config, "inv_ramp_epochs", 0))
        xcov_scale = _ramp_scale(epoch, getattr(self.config, "xcov_start_epoch", 1),
                                 getattr(self.config, "xcov_ramp_epochs", 0))

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
            scan_id = batch.get("scan_id")
            if scan_id is not None:
                scan_id = scan_id.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(x_a, x_b, ref_b, scan_id=scan_id,
                                     inv_scale=inv_scale, xcov_scale=xcov_scale)
                loss = outputs["total_loss"]

            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            self.model.update_target_encoder(momentum=ema_momentum)

            # Sync-free accumulation (detach keeps the graph released)
            acc_loss += loss.detach().double()
            acc_pred += outputs["pred_loss"].detach().double()
            _car = outputs.get("carrier_loss", None)
            acc_carrier += _car.detach().double() if _car is not None else 0.0
            acc_var += outputs["var_loss"].detach().double()
            acc_cov += outputs["cov_loss"].detach().double()
            _inv = outputs.get("inv_loss", None)
            acc_inv += _inv.detach().double() if _inv is not None else 0.0
            _xc = outputs.get("xcov_loss", None)
            acc_xcov += _xc.detach().double() if _xc is not None else 0.0
            _zc = outputs.get("zc_cond_loss", None)
            acc_zc += _zc.detach().double() if _zc is not None else 0.0

            if batch_idx < 5:
                sample_reps.append(outputs["H_ctx_a"].detach().cpu())

            # Refresh displayed values at most every LOG_EVERY batches
            if batch_idx % LOG_EVERY == 0:
                snap = torch.stack([acc_loss, acc_pred, acc_carrier, acc_var, acc_inv]).cpu() / (batch_idx + 1)
                snap_list = snap.tolist()
                disp = {"loss": snap_list[0], "pred": snap_list[1], "car": snap_list[2],
                        "var": snap_list[3], "inv": snap_list[4]}
            pbar.set_postfix({
                "loss": f"{disp['loss']:.4f}",
                "pred": f"{disp['pred']:.4f}",
                "car": f"{disp['car']:.4f}",
                "var": f"{disp['var']:.4f}",
                "inv": f"{disp['inv']:.4f}",
                "lr": f"{lr:.2e}",
            })

        avg_loss = (acc_loss / total_batches).item()
        avg_pred = (acc_pred / total_batches).item()
        avg_carrier = (acc_carrier / total_batches).item()
        avg_var = (acc_var / total_batches).item()
        avg_cov = (acc_cov / total_batches).item()
        avg_inv = (acc_inv / total_batches).item()
        avg_xcov = (acc_xcov / total_batches).item()
        avg_zc = (acc_zc / total_batches).item()

        all_reps = torch.cat(sample_reps, dim=0)
        svd_res = compute_svd_diagnostics(all_reps)

        return {
            "epoch": epoch,
            "total_loss": avg_loss,
            "pred_loss": avg_pred,
            "carrier_loss": avg_carrier,
            "var_loss": avg_var,
            "cov_loss": avg_cov,
            "inv_loss": avg_inv,
            "xcov_loss": avg_xcov,
            "zc_cond_loss": avg_zc,
            "effective_rank": svd_res["effective_rank"],
            "lr": lr,
        }

    def _recover_best_state(self, warmup_epochs: int):
        """Recomputes early-stopping best state from resumed history.

        Warmup epochs are EXCLUDED: the warmup reset (epoch == warmup_epochs)
        discards warmup-era bests in a fresh run, so including them here would
        let a warmup-epoch value permanently block the post-warmup improvement
        paths (especially the val_loss branch). Filter by stored epoch numbers;
        fall back to the full history when resuming mid-warmup.
        Returns (best_loss, best_val_auc, best_val_snr, patience_counter).
        """
        best_loss, best_val_auc, best_val_snr = float("inf"), 0.0, 0.0
        if hasattr(self, "history") and self.history:
            ep_list = self.history.get("epoch", [])
            post_idx = [i for i, e in enumerate(ep_list) if isinstance(e, int) and e > warmup_epochs]
            if not post_idx:
                post_idx = list(range(len(ep_list)))

            def _col(key: str):
                vals = self.history.get(key, [])
                return [vals[i] for i in post_idx if i < len(vals) and isinstance(vals[i], (int, float))]

            post_val = [v for v in _col("val_loss") if v and v > 0]
            if post_val:
                best_loss = min(post_val)
            elif _col("total_loss"):
                best_loss = min(_col("total_loss"))
            post_auc = _col("val_auc")
            if post_auc:
                best_val_auc = max(post_auc)
            post_snr = _col("val_snr")
            if post_snr:
                best_val_snr = max(post_snr)
        return best_loss, best_val_auc, best_val_snr, getattr(self, "_resumed_patience", 0)

    def train(
        self,
        dataset: PairedPECTDataset,
        epochs: Optional[int] = None,
    ) -> str:
        total_epochs = epochs or self.config.epochs

        # Resume guard: requesting fewer total epochs than already trained is
        # almost always a CLI mistake (e.g. forgetting --epochs). Fail loud
        # instead of printing a fake "Training Complete".
        ckpt_dir = self.logger.checkpoint_dir if self.logger else self.config.checkpoint_dir
        viz_dir = self.logger.visualizations_dir if self.logger else self.config.visualizations_dir
        best_ckpt_path = os.path.join(ckpt_dir, "best_cross_condition_jepa.pt") if ckpt_dir else "best_cross_condition_jepa.pt"
        if self.start_epoch > total_epochs:
            msg = (f"[Resume] start_epoch={self.start_epoch} > total_epochs={total_epochs}: "
                   f"nothing to train. Re-run with --epochs >= {self.start_epoch}.")
            if self.logger:
                self.logger.info(msg)
            if os.path.exists(best_ckpt_path):
                return best_ckpt_path
            raise ValueError(msg)

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

        # If resumed, recover best metric state from history (post-warmup only;
        # loss-based state uses VAL loss, consistent with early stopping)
        if hasattr(self, "history") and self.history:
            best_loss, best_val_auc, best_val_snr, patience_counter = \
                self._recover_best_state(warmup_epochs)

        # Train/val-divergence guard state (val recon rising far above its
        # running best while train stays flat = shortcut/overfit, exp49)
        div_patience = max(0, getattr(self.config, "diverge_patience", 3))
        div_factor = getattr(self.config, "diverge_factor", 1.5)
        div_min_val = float("inf")
        div_counter = 0
        sndvar_max = 0.0
        sndvar_warned = False

        ckpt_dir = self.logger.checkpoint_dir if self.logger else self.config.checkpoint_dir
        viz_dir = self.logger.visualizations_dir if self.logger else self.config.visualizations_dir
        best_ckpt_path = os.path.join(ckpt_dir, "best_cross_condition_jepa.pt") if ckpt_dir else "best_cross_condition_jepa.pt"

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
                _vl = epoch_metrics.get("val_loss", 0.0) or 0.0
                _gap = _vl - epoch_metrics["pred_loss"] if _vl > 0 else 0.0
                val_info = (
                    f" | ValLoss: {_vl:.4f}"
                    f" | Val AUC: {epoch_metrics['val_auc']:.4f}"
                    f" | SNR: {epoch_metrics['val_snr']:.2f}"
                    f" | SndVar: {epoch_metrics['val_sound_var']:.4f}"
                    f" | Gap: {_gap:+.4f}"
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
            # NOTE: all loss-based comparisons use VAL loss (self-conditioned
            # JEPA recon on held-out scans), never train loss. Train L1 drops
            # monotonically even while the model overfits condition drift, so
            # rewarding train-loss decrease rewards overfitting.
            curr_loss = epoch_metrics.get("val_loss", 0.0) or 0.0
            if curr_loss == 0.0:
                # val telemetry unavailable (no val_scans): fall back to train
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
                    improve_reason = f"Val loss improved: {best_loss:.4f} -> {curr_loss:.4f}"
                    best_loss = curr_loss
            else:  # "composite"
                # Multi-criterion representation health:
                # 1. Defect AUC improved significantly
                # 2. OR Latent SNR improved by >= 2% while AUC stable
                # 3. OR Val loss dropped by min_delta while AUC stable
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
                    improve_reason = f"Val loss improved: {best_loss:.4f} -> {curr_loss:.4f} (AUC stable: {curr_auc:.4f})"
                    best_loss = min(best_loss, curr_loss)

            if improved:
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "patience_counter": patience_counter,
                    "config": self.config,
                    "metrics": epoch_metrics,
                    "history": self.history,
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
                        f"(Best Val AUC: {best_val_auc:.4f}, Best SNR: {best_val_snr:.2f}, Best ValLoss: {best_loss:.4f})"
                    )
                    if self.logger:
                        self.logger.info(status_msg)
                    else:
                        print(status_msg, flush=True)

                    if patience_counter >= patience:
                        stop_msg = (
                            f"\n[Early Stopping Triggered at Epoch {epoch:02d}] "
                            f"Neither Val AUC, Latent SNR, nor Val Loss improved for {patience_counter} consecutive epochs "
                            f"(Best Val AUC: {best_val_auc:.4f}, Best SNR: {best_val_snr:.2f}, Best ValLoss: {best_loss:.4f}). "
                            f"Terminating training early."
                        )
                        if self.logger:
                            self.logger.info(stop_msg)
                        else:
                            print(stop_msg, flush=True)
                        break

            # Train/val-divergence guard: val recon far above its running best
            # for div_patience consecutive evals -> stop. Catches shortcut /
            # overfit runs (exp49) long before early-stopping patience runs
            # out. Only active when val telemetry is present.
            vl = epoch_metrics.get("val_loss", 0.0) or 0.0
            if vl > 0 and div_patience > 0:
                div_min_val, div_counter = _val_divergence_step(vl, div_min_val, div_counter, div_factor)
                if div_counter >= div_patience:
                    div_msg = (
                        f"\n[Divergence Stop at Epoch {epoch:02d}] "
                        f"Val recon {vl:.4f} exceeds {div_factor}x its running best "
                        f"({div_min_val:.4f}) for {div_counter} consecutive evals while "
                        f"train pred is {epoch_metrics['total_loss']:.4f}. "
                        f"Representation is shortcutting/overfitting, not learning."
                    )
                    if self.logger:
                        self.logger.info(div_msg)
                    else:
                        print(div_msg, flush=True)
                    break

            # Sound-variance collapse watch (warn once): SndVar falling below
            # 10% of its running max while training continues = amplitude
            # shrinkage toward a noise floor. Read R_eff only jointly with
            # SndVar/SNR/AUC from here on.
            sv = epoch_metrics.get("val_sound_var", 0.0) or 0.0
            if sv > 0:
                sndvar_max = max(sndvar_max, sv)
            if (not sndvar_warned and sndvar_max > 0 and sv > 0
                    and sv < 0.1 * sndvar_max and epoch > warmup_epochs):
                sndvar_warned = True
                col_msg = (f"  [WARN] Sound-metal variance collapsed "
                           f"({sndvar_max:.4f} -> {sv:.4f}); R_eff={epoch_metrics.get('effective_rank', 0.0):.1f}D "
                           f"with collapsed amplitude is a noise floor, not learned rank.")
                if self.logger:
                    self.logger.info(col_msg)
                else:
                    print(col_msg, flush=True)

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
                "scaler_state_dict": self.scaler.state_dict(),
                "patience_counter": patience_counter,
                "config": self.config,
                "metrics": epoch_metrics,
                "history": self.history,
            }, latest_path)

            # Plot training curves periodically (cheap, but a full matplotlib
            # 4-panel figure every epoch wastes ~0.7 s x epochs for an image
            # that gets overwritten anyway) + always at the end of training.
            # (non-fatal: a plotting hiccup must never kill a long training run)
            if epoch % 5 == 0 or epoch == total_epochs:
                curve_path = os.path.join(viz_dir, "training_curves.png")
                try:
                    plot_training_curves(self.history, curve_path)
                except Exception as e:
                    warn_msg = f"  [WARN] Skipping training-curve plot at epoch {epoch}: {type(e).__name__}: {e}"
                    if self.logger:
                        self.logger.info(warn_msg)
                    else:
                        print(warn_msg, flush=True)

        # Always save final training curve at conclusion of training
        curve_path = os.path.join(viz_dir, "training_curves.png")
        try:
            plot_training_curves(self.history, curve_path)
        except Exception as e:
            warn_msg = f"  [WARN] Skipping final training-curve plot: {type(e).__name__}: {e}"
            if self.logger:
                self.logger.info(warn_msg)
            else:
                print(warn_msg, flush=True)

        fin_msg = f"\nTraining Complete! Best model saved to: {best_ckpt_path}"
        if self.logger:
            self.logger.info(fin_msg)
        else:
            print(fin_msg, flush=True)

        return best_ckpt_path
