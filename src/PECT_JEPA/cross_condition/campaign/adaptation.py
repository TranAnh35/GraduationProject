"""Matched neural controls and PECT time/frequency consistency adaptation.

tf_consistency_pect is an independently implemented PECT adaptation inspired by
TFDA (Information Sciences 2025), NOT a reproduction of the original algorithm:
full-waveform spatial views replace temporal splitting; no class-uniform IM
objective is imposed on the rare-defect target distribution.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..downstream.domain_adaptation.da_bridge import grad_reverse
from .common import ProgressReporter, atomic_torch, seed_all, rng_state, restore_rng


class NeuralReadout(nn.Module):
    def __init__(self, dim, dual=False):
        super().__init__()
        self.dual = dual
        if dual:
            self.time = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
            self.frequency = nn.Sequential(nn.Linear(dim - 256, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
            self.time_head = nn.Linear(64, 2)
            self.frequency_head = nn.Linear(64, 2)
            hidden = 128
        else:
            self.adapter = nn.Sequential(nn.Linear(dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, dim))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
            hidden = dim
        self.head = nn.Linear(hidden, 2)
        self.domain = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 2))

    def embedding(self, x):
        if self.dual:
            return torch.cat([self.time(x[:, :256]), self.frequency(x[:, 256:])], -1)
        return x + self.adapter(x)

    def forward(self, x):
        h = self.embedding(x)
        return self.head(h), h


def fit_readout(output, x, y, xs, xt, seed, kind, cfg, device="cpu"):
    """Resume source warmup and adaptation at optimizer-step granularity."""
    output = Path(output)
    seed_all(seed)
    dual = kind in ("tf_source", "tf_consistency_pect")
    model = NeuralReadout(x.shape[1], dual=dual).to(device)
    source_steps, adapt_steps = cfg["neural_source_steps"], cfg["neural_adapt_steps"]
    total = source_steps if kind == "tf_source" else source_steps + adapt_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = torch.tensor([len(y) / max(1, (y == k).sum()) for k in [0, 1]], device=device).float()
    weights /= weights.mean()
    checkpoint = output / "readout_latest.pt"
    progress = ProgressReporter(output)
    start = 0
    teacher = None
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck["kind"] != kind:
            raise ValueError("Readout checkpoint method mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start = ck["step"]
        if ck["teacher"] is not None:
            teacher = copy.deepcopy(model).eval()
            teacher.load_state_dict(ck["teacher"])
        restore_rng(ck["rng"])
    progress.update("neural_readout", start, total, "step", detail=f"{kind} resumed", force=True)
    if dual and start >= source_steps:
        for name, p in model.named_parameters():
            if "head" in name:
                p.requires_grad_(False)
    for step in range(start, total):
        model.train()
        rng = np.random.RandomState(seed + step * 1009)
        n = cfg["batch_size"]
        if step < source_steps or kind in ("neural_control", "dann", "tf_source"):
            ids = rng.randint(len(x), size=n)
            bx = torch.from_numpy(x[ids]).to(device)
            by = torch.from_numpy(y[ids]).to(device)
            logits, h = model(bx)
            loss = F.cross_entropy(logits, by, weight=weights)
            if dual:
                loss = loss + 0.5 * (F.cross_entropy(model.time_head(h[:, :64]), by, weight=weights)
                    + F.cross_entropy(model.frequency_head(h[:, 64:]), by, weight=weights))
            if kind == "dann" and step >= source_steps:
                sx = torch.from_numpy(xs[rng.randint(len(xs), size=n)]).to(device)
                tx = torch.from_numpy(xt[rng.randint(len(xt), size=n)]).to(device)
                joined = torch.cat([model.embedding(sx), model.embedding(tx)])
                domain = torch.cat([torch.zeros(n), torch.ones(n)]).long().to(device)
                loss = loss + 0.1 * F.cross_entropy(model.domain(grad_reverse(joined, 1.0)), domain)
        else:
            if teacher is None:
                teacher = copy.deepcopy(model).eval()
                for p in teacher.parameters():
                    p.requires_grad_(False)
                for name, p in model.named_parameters():
                    if "head" in name:
                        p.requires_grad_(False)
            tx = torch.from_numpy(xt[rng.randint(len(xt), size=n)]).to(device)
            logits, h = model(tx)
            with torch.no_grad():
                # Teacher-neighbor pseudo labels; bank is unlabelled and fixed
                # by absolute step, with no access to target CAD or class prior.
                bank = torch.from_numpy(xt[rng.choice(len(xt), min(1024, len(xt)), replace=False)]).to(device)
                bl, bh = teacher(bank)
                tl, th = teacher(tx)
                nearest = torch.cdist(F.normalize(th, dim=-1), F.normalize(bh, dim=-1)).topk(5, largest=False).indices
                soft = 0.5 * (tl.softmax(-1) + bl.softmax(-1)[nearest].mean(1))
                confidence = soft.max(-1).values
                weight = ((confidence - 0.5) / 0.5).clamp(0, 1)
            time_logits, freq_logits = model.time_head(h[:, :64]), model.frequency_head(h[:, 64:])
            losses = [-(soft * z.log_softmax(-1)).sum(-1) for z in (logits, time_logits, freq_logits)]
            loss = sum((v * weight).mean() for v in losses) / 3 + 0.1 * F.mse_loss(h, th)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite neural loss at step {step}")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite neural gradient")
        optimizer.step()
        if teacher is not None:
            with torch.no_grad():
                for a, b in zip(teacher.parameters(), model.parameters()):
                    a.lerp_(b, 0.01)
        if (step + 1) % cfg["save_every_steps"] == 0 or step + 1 == total:
            atomic_torch(checkpoint, {"kind": kind, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step + 1, "rng": rng_state(),
                "teacher": teacher.state_dict() if teacher is not None else None})
            progress.update("neural_readout", step + 1, total, "step",
                detail=f"{kind} checkpoint", next_step=step + 1,
                total_steps=total, kind=kind)
        else:
            phase = "source" if step < source_steps else "adaptation"
            progress.update("neural_readout", step + 1, total, "step",
                            detail=f"{kind} {phase}")
    model.eval()
    return model


def predict(model, x, batch_size=4096):
    device = next(model.parameters()).device
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[i:i + batch_size].astype(np.float32)).to(device))[0].softmax(-1)[:, 1].cpu().numpy()
                               for i in range(0, len(x), batch_size)])
