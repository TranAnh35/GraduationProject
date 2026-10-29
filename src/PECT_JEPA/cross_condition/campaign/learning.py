"""Whole-waveform SSL and resumable optimization for the campaign.

CampaignModel subclasses the canonical model without altering legacy defaults.
Masking happens BEFORE any spatially coupled tokenizer operation (H37).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..configs.config import CrossConditionConfig
from ..data.paired_dataset import PairedPECTDataset
from ..models.cross_condition_jepa import CrossConditionJEPA, normalized_probe_l1, compute_vicreg_loss
from .common import ProgressReporter, atomic_json, atomic_torch, digest, seed_all, rng_state, restore_rng


class CampaignModel(CrossConditionJEPA):
    def __init__(self, config, objective="paired_jepa", adapter=False):
        super().__init__(config)
        self.objective = objective
        self.encoder_adapter = None
        if adapter:
            d = config.embed_dim
            self.encoder_adapter = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, d))
            nn.init.zeros_(self.encoder_adapter[-1].weight)
            nn.init.zeros_(self.encoder_adapter[-1].bias)
        self.reconstruction_head = nn.Linear(config.embed_dim, 128) if objective == "spatial_mae" else None

    def encode(self, tokens, pos):
        h = self.context_encoder(tokens, pos)
        return h if self.encoder_adapter is None else h + self.encoder_adapter(h)

    def extract_features(self, x, bg_ref_energy=None):
        tokens, pos = self.tokenizer(x, bg_ref_energy=bg_ref_energy)
        h = self.encode(tokens, pos)
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)

    def visible_tokens(self, x, ctx):
        """Hidden probes cannot affect normalization, sign anchoring or energy."""
        b, _, _, t = x.shape
        ar = torch.arange(b, device=x.device)[:, None]
        flat = x.reshape(b, 25, t)
        visible = flat[ar, ctx]
        safe = visible.mean(1, keepdim=True).expand(-1, 25, -1).clone()
        safe[ar, ctx] = visible
        tokens, pos = self.tokenizer(safe.reshape(b, 5, 5, t))
        return tokens[ar, ctx], pos[ar, ctx]

    def ssl_loss(self, batch, teacher=None):
        xa = batch["x_a"]
        same = self.objective in ("self_jepa", "spatial_mae") or teacher is not None
        xb = xa if same else batch["x_b"]
        ref = batch["ref_a"] if same else batch["ref_b"]
        b = len(xa)
        ar = torch.arange(b, device=xa.device)[:, None]
        masks = torch.randint(len(self.target_idx_bank), (b,), device=xa.device)
        ctx, tgt = self.context_idx_bank[masks], self.target_idx_bank[masks]
        tokens, pos = self.visible_tokens(xa, ctx)
        hc = self.encode(tokens, pos)
        target_model = teacher if teacher is not None else self
        with torch.no_grad():
            tb, pb = target_model.tokenizer(xb)
            ht = target_model.target_encoder(tb[ar, tgt], pb[ar, tgt])
        cond = self.extract_condition_token(ref) if self.config.use_condition_token else None
        pred = self.predictor(hc, pb[ar, tgt],
                              condition_token=F.normalize(cond, dim=-1) if cond is not None else None)
        if self.reconstruction_head is not None:
            truth = xb.reshape(b, 25, 128)[ar, tgt].float()
            recon = self.reconstruction_head(pred).float()
            # Complete trace and full complex spectrum; no fixed time slices.
            pred_loss = F.smooth_l1_loss(recon, truth) + 0.1 * (
                torch.fft.rfft(recon, norm="ortho") - torch.fft.rfft(truth, norm="ortho")).abs().mean()
        else:
            pred_loss = normalized_probe_l1(pred.float(), ht.float())
        if teacher is None:
            # Uniform pool only. No positional mask-dependent weighted shortcut.
            vic, var, cov = compute_vicreg_loss(hc.float().mean(1),
                var_weight=self.config.var_weight, cov_weight=self.config.cov_weight,
                gamma=self.config.var_gamma, scan_id=batch.get("scan_id"))
            total = pred_loss + vic
        else:
            # Fixed source teacher prevents a moving target during adaptation.
            with torch.no_grad():
                hc0 = teacher.context_encoder(tokens, pos)
            var = cov = pred_loss.new_zeros(())
            total = pred_loss + 0.1 * F.mse_loss(hc.float(), hc0.float())
        return total, {"pred_loss": pred_loss, "var_loss": var, "cov_loss": cov,
                       "latent_std": hc.float().mean(1).std(0).mean()}


def load_model(path, device="cpu", adapter=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = CrossConditionConfig(**ckpt["config"])
    model = CampaignModel(config, ckpt["objective"], adapter=ckpt.get("adapter", False))
    model.load_state_dict(ckpt["model"], strict=True)
    if adapter and model.encoder_adapter is None:
        wrapped = CampaignModel(config, ckpt["objective"], adapter=True)
        missing, unexpected = wrapped.load_state_dict(model.state_dict(), strict=False)
        if unexpected or any(not k.startswith("encoder_adapter.") for k in missing):
            raise ValueError(f"Unexpected adapter load mismatch: {missing}/{unexpected}")
        model = wrapped
    return model.to(device)


def train(job, protocol, output, settings, device, fingerprint, prepare_dir=None):
    """Atomic batch checkpoints with exact RNG restoration and epoch metrics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(job["seed"])
    parent = job.get("parent_checkpoint")
    teacher = None
    if parent:
        model = load_model(parent, device, adapter=True)
        teacher = load_model(parent, device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("encoder_adapter."))
        config = model.config
    else:
        config = CrossConditionConfig(embed_dim=settings["embed_dim"],
            encoder_depth=settings["encoder_depth"], predictor_depth=2,
            condition_modulation="adaln", pred_loss_type="normalized_l1",
            inv_weight=0.0, mixed_precision=False,
            use_condition_token=job.get("use_condition_token", True))
        model = CampaignModel(config, job["objective"]).to(device)
        atomic_json(output / "config.json", {**vars(config), "campaign_settings": settings,
            "objective": job["objective"], "scope": job["scope"], "seed": job["seed"]})
        if job["objective"] == "random":
            atomic_torch(output / "best.pt", {"model": model.state_dict(), "config": vars(config),
                "objective": job["objective"], "adapter": False, "fingerprint": fingerprint,
                "epoch": 0, "rng": rng_state()})
            atomic_json(output / "result.json", {"kind": "random", "epochs": 0})
            return
    files = protocol["source"]
    if job["scope"] == "transductive":
        files = files + protocol["target"]
    elif job["scope"] == "adapt":
        files = protocol["target"]
    if parent:
        atomic_json(output / "config.json", {**vars(config), "campaign_settings": settings,
            "objective": model.objective, "scope": "adapt", "seed": job["seed"],
            "parent_checkpoint": parent})
    dataset = PairedPECTDataset(file_paths=files, grid_stride=settings["grid_stride"],
        samples_per_epoch=settings["samples_per_epoch"], num_partitions=1,
        star_radii=config.star_radii, transform_csv=settings["transform_csv"], seed=job["seed"],
        canonical_cache={p: str(Path(prepare_dir) / (digest(str(Path(p).resolve()))[:20] + ".npy"))
                         for p in files} if prepare_dir else None,
        self_pairs=parent is not None or job["objective"] in ("self_jepa", "spatial_mae"))
    if protocol.get("spatial_holdout"):
        from .protocol import spatial_masks
        keep, _ = spatial_masks((270, 270))
        dataset.coords = dataset.coords[keep[dataset.coords[:, 0], dataset.coords[:, 1]]]
        dataset.coords_per_pair = len(dataset.coords)
        dataset.total_exhaustive_samples = len(dataset.directed_pairs) * len(dataset.coords)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=settings["adapt_lr"] if parent else settings["lr"], weight_decay=1e-4)
    epochs = settings["adapt_epochs"] if parent else settings["epochs"]
    batch_size = settings["batch_size"]
    steps = math.ceil(settings["samples_per_epoch"] / batch_size)
    epoch, step, history, sums, count = 1, 0, [], {}, 0
    latest = output / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch; use a new campaign directory")
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        epoch, step, history = ck["epoch"], ck["step"], ck["history"]
        sums, count = ck["sums"], ck["count"]
        restore_rng(ck["rng"])

    total_steps = epochs * steps
    completed_steps = min(total_steps, max(0, (epoch - 1) * steps + step))
    progress_reporter = ProgressReporter(output)
    progress_reporter.update("train", completed_steps, total_steps, "batch",
        detail=f"epoch {epoch}/{epochs}, batch {step}/{steps}", force=True,
        next_epoch=epoch, next_batch=step)

    def checkpoint(next_epoch, next_step):
        atomic_torch(latest, {"schema": 1, "fingerprint": fingerprint,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": vars(config), "objective": model.objective,
            "adapter": model.encoder_adapter is not None, "rng": rng_state(),
            "epoch": next_epoch, "step": next_step, "history": history,
            "sums": sums, "count": count})
        current_step = min(total_steps, max(0, (next_epoch - 1) * steps + next_step))
        progress_reporter.update("train", current_step, total_steps, "batch",
            detail=f"checkpoint epoch {next_epoch}/{epochs}, batch {next_step}/{steps}",
            next_epoch=next_epoch, next_batch=next_step,
            total_epochs=epochs, batches_per_epoch=steps)

    while epoch <= epochs:
        model.train()
        for step_idx in range(step, steps):
            # Absolute step determines samples: explicit cap does not repeatedly
            # train the first directed pair / first 256 spatial coordinates.
            n = min(batch_size, settings["samples_per_epoch"] - step_idx * batch_size)
            rng = np.random.RandomState(job["seed"] + epoch * 100003 + step_idx)
            indices = rng.randint(0, dataset.total_exhaustive_samples, size=n)
            batch = torch.utils.data.default_collate([dataset[int(i)] for i in indices])
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            if teacher is not None or model.objective in ("self_jepa", "spatial_mae"):
                batch["x_b"], batch["ref_b"] = batch["x_a"], batch["ref_a"]
            progress = ((epoch - 1) * steps + step_idx) / max(1, epochs * steps)
            base = settings["adapt_lr"] if parent else settings["lr"]
            lr = base * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.ssl_loss(batch, teacher)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {epoch}/{step_idx}; previous checkpoint retained")
            loss.backward()
            norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient; optimizer step refused")
            optimizer.step()
            if teacher is None:
                model.update_target_encoder(momentum=0.996)
            metrics = {k: float(v.detach()) for k, v in metrics.items()}
            metrics.update(total_loss=float(loss.detach()), lr=lr, inv_loss=0.0)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v * n
            count += n
            progress_reporter.update("train", (epoch - 1) * steps + step_idx + 1,
                total_steps, "batch",
                detail=f"epoch {epoch}/{epochs}, batch {step_idx + 1}/{steps}, loss {float(loss.detach()):.4f}",
                next_epoch=epoch, next_batch=step_idx + 1)
            if (step_idx + 1) % settings["save_every_steps"] == 0:
                checkpoint(epoch, step_idx + 1)
        row = {"epoch": epoch, **{k: v / count for k, v in sums.items()}}
        if prepare_dir and protocol["val"]:
            from .telemetry import source_telemetry
            telemetry_steps = len(protocol["val"]) * 8
            progress_reporter.update("source_diagnostics", 0, telemetry_steps, "batch",
                                     detail="source validation telemetry", force=True)
            row.update(source_telemetry(model, protocol["val"], prepare_dir,
                settings["data_dir"], device, teacher,
                progress_callback=lambda current, total, detail: progress_reporter.update(
                    "source_diagnostics", current, total, "batch", detail=detail)))
        else:
            row.update({k: None for k in ["val_loss", "val_auc", "val_ap", "val_snr", "val_sound_var", "effective_rank"]})
        for name, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite epoch diagnostic {name}; checkpoint not promoted")
        history.append(row)
        print(f"epoch {epoch}/{epochs}: loss={row['total_loss']:.6f}", flush=True)
        # Fixed-budget final checkpoint: no target labels and no loss-only
        # checkpoint selection masquerading as improved representation quality.
        sums, count = {}, 0
        checkpoint(epoch + 1, 0)
        with (output / "metrics_epoch.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        epoch, step = epoch + 1, 0
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    atomic_torch(output / "best.pt", ck)
    atomic_json(output / "result.json", {"epochs": epochs, "selection": "fixed_budget_final",
        "objective": model.objective, "adapter": parent is not None,
        "metrics": history[-1], "pretrain_files": files})
