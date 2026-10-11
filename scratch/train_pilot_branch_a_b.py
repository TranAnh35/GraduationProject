# -*- coding: utf-8 -*-
"""
R1-2: Pilot Training with Reformed Objectives (Branch A & Branch B)
Protocol V2:
Branch A (main, compliance-focused):
- Tokenizer T1 (autonomous, no median subtraction).
- Target Delta H WITHOUT pos_embed (position only enters predictor query).
- Channel-normalize Delta H using unlabeled train statistics (mu_c, sigma_c).
- Loss: Smooth L1 with beta=1.0 on normalized Delta H in FP32.
- Fixed: 20 train files, 4 val files, 3 epochs across 5 seeds: {42, 100, 2024, 777, 999}.
- One convergence run: Branch A, seed 42, 20 epochs.

Branch B (exploratory, scan-median centered target):
- Target: Delta H minus scan-median of the same probe.
- 3 epochs, 5 seeds.
"""

import os
import sys
import io
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets


class CScanPatchDataset(Dataset):
    """Dataset extracting 5x5 concentric star patches from C-scans with RAM caching."""
    def __init__(self, file_paths, config, patches_per_file=500, seed=42):
        self.config = config
        self.offsets = get_spatial_topology_offsets(config.spatial_topology, tuple(config.star_radii), config.grid_size)
        max_off = int(np.max(np.abs(self.offsets)))
        self.pad = max(config.grid_size // 2, max_off)
        self.rng = np.random.RandomState(seed)
        
        self.scans = []
        for fp in file_paths:
            cscan = load_cscan_from_tdms(fp)
            sY, sX, C = cscan.shape
            padded = np.pad(cscan, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")
            self.scans.append({
                "padded": padded,
                "sY": sY,
                "sX": sX,
                "file_path": fp
            })
        self.patches_per_file = patches_per_file
        self.total_len = len(self.scans) * patches_per_file

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        scan_idx = idx // self.patches_per_file
        scan = self.scans[scan_idx]
        sY, sX = scan["sY"], scan["sX"]
        
        r = self.rng.randint(0, sY)
        c = self.rng.randint(0, sX)
        sr = r + self.pad + self.offsets[:, 0]
        sc = c + self.pad + self.offsets[:, 1]
        patch = scan["padded"][sr, sc, :].reshape(self.config.grid_size, self.config.grid_size, -1)
        return torch.from_numpy(patch).float()


def compute_unlabeled_delta_h_stats(model, loader, device, branch="A"):
    """Compute empirical channel-wise mean and std of Delta H on train set."""
    model.eval()
    all_dh = []
    print("Computing unlabeled channel statistics (mu_c, sigma_c) on train set...", flush=True)
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            B = x.shape[0]
            tokens, pos = model.tokenizer(x)
            
            # Sample context and target masks
            context_indices, target_indices, _ = model.masker.sample_mask(B, device=device)
            
            batch_arange = torch.arange(B, device=device).unsqueeze(1)
            context_tokens = tokens[batch_arange, context_indices]
            context_pos = pos[batch_arange, context_indices]
            target_tokens = tokens[batch_arange, target_indices]
            
            # In Branch A: target has NO pos_embed
            target_pos = torch.zeros_like(context_pos[:, :target_tokens.shape[1], :])
            
            H_ctx = model.context_encoder(context_tokens, context_pos, context_indices=context_indices)
            H_tgt = model.context_encoder(target_tokens, target_pos, context_indices=target_indices)
            
            H_base = H_ctx.mean(dim=1, keepdim=True)
            dh = (H_tgt - H_base).reshape(-1, H_tgt.shape[-1])
            all_dh.append(dh.cpu().numpy())
            if len(all_dh) * B >= 5000:
                break
    all_dh = np.vstack(all_dh)
    mu_c = torch.from_numpy(np.mean(all_dh, axis=0)).float().to(device)
    sigma_c = torch.from_numpy(np.std(all_dh, axis=0) + 1e-6).float().to(device)
    print(f"  mu norm: {mu_c.norm().item():.4f}, sigma mean: {sigma_c.mean().item():.4f}", flush=True)
    return mu_c, sigma_c


def train_pilot(branch="A", seed=42, epochs=3, mu_c=None, sigma_c=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cfg_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/config_5x5.json"
    split_path = "experiments/5x5/exp41_autonomous_dual_domain/checkpoints/exp41_autonomous_dual_domain_split_summary.json"
    
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_files = splits["train_files"]
    val_files = splits["val_files"]
    
    config = Spatiotemporal5x5Config.from_json(cfg_path)
    config.epochs = epochs
    
    # Initialize fresh model
    model = PECT_JEPA_5x5(config).to(device)
    
    train_ds = CScanPatchDataset(train_files, config, patches_per_file=600, seed=seed)
    val_ds = CScanPatchDataset(val_files, config, patches_per_file=300, seed=seed + 1)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    if mu_c is None or sigma_c is None:
        mu_c, sigma_c = compute_unlabeled_delta_h_stats(model, train_loader, device, branch=branch)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    
    print(f"\n--- Training Branch {branch} | Seed {seed} | Epochs {epochs} ---", flush=True)
    
    history = []
    for ep in range(epochs):
        model.train()
        train_loss, train_pred = [], []
        for x in train_loader:
            x = x.to(device)
            B = x.shape[0]
            tokens, pos = model.tokenizer(x)
            
            context_indices, target_indices, _ = model.masker.sample_mask(B, device=device)
            batch_arange = torch.arange(B, device=device).unsqueeze(1)
            
            context_tokens = tokens[batch_arange, context_indices]
            context_pos = pos[batch_arange, context_indices]
            target_tokens = tokens[batch_arange, target_indices]
            
            # Predictor receives true target pos queries
            pred_query_pos = pos[batch_arange, target_indices]
            
            # Target representation:
            if branch == "A":
                # Branch A: ZERO positional embedding in target!
                target_pos = torch.zeros_like(pred_query_pos)
            else:
                # Branch B: Standard pos, but median scan centering
                target_pos = pred_query_pos
                
            H_ctx = model.context_encoder(context_tokens, context_pos, context_indices=context_indices)
            H_tgt_full = model.context_encoder(target_tokens, target_pos, context_indices=target_indices)
            H_tgt = H_tgt_full.detach()
            
            # Predictor prediction
            H_pred = model.predictor(H_context=H_ctx, target_pos=pred_query_pos)
            
            H_base = H_ctx.mean(dim=1, keepdim=True)
            delta_h_target = H_tgt - H_base
            delta_h_pred = H_pred  # Predictor predicts Delta H directly
            
            if branch == "B":
                # Subtract batch/scan median per probe
                delta_h_target = delta_h_target - delta_h_target.median(dim=0, keepdim=True).values
                delta_h_pred = delta_h_pred - delta_h_pred.median(dim=0, keepdim=True).values
                
            # Channel normalization by train statistics
            delta_h_tgt_norm = (delta_h_target - mu_c) / sigma_c
            delta_h_pred_norm = (delta_h_pred - mu_c) / sigma_c
            
            # Smooth L1 Loss with beta=1.0 in FP32
            loss_pred = F.smooth_l1_loss(delta_h_pred_norm.float(), delta_h_tgt_norm.float(), beta=1.0)
            
            # VICReg regularization on active target tokens to maintain manifold rank
            loss_dict_vic = model.loss_fn(
                H_pred=H_pred,
                H_target=H_tgt,
                target_indices=target_indices,
                H_ctx=H_ctx,
            )
            loss_var = loss_dict_vic.get("loss_var", torch.tensor(0.0, device=device))
            loss_cov = loss_dict_vic.get("loss_cov", torch.tensor(0.0, device=device))
            
            loss = loss_pred + 0.1 * loss_var + 0.05 * loss_cov
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            train_loss.append(loss.item())
            train_pred.append(loss_pred.item())
            
        # Validation evaluation
        model.eval()
        val_pred = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                B = x.shape[0]
                tokens, pos = model.tokenizer(x)
                context_indices, target_indices, _ = model.masker.sample_mask(B, device=device)
                batch_arange = torch.arange(B, device=device).unsqueeze(1)
                
                context_tokens = tokens[batch_arange, context_indices]
                context_pos = pos[batch_arange, context_indices]
                target_tokens = tokens[batch_arange, target_indices]
                pred_query_pos = pos[batch_arange, target_indices]
                target_pos = torch.zeros_like(pred_query_pos) if branch == "A" else pred_query_pos
                
                H_ctx = model.context_encoder(context_tokens, context_pos, context_indices=context_indices)
                H_tgt = model.context_encoder(target_tokens, target_pos, context_indices=target_indices)
                H_pred = model.predictor(H_context=H_ctx, target_pos=pred_query_pos)
                
                H_base = H_ctx.mean(dim=1, keepdim=True)
                delta_h_target = H_tgt - H_base
                delta_h_pred = H_pred
                if branch == "B":
                    delta_h_target = delta_h_target - delta_h_target.median(dim=0, keepdim=True).values
                    delta_h_pred = delta_h_pred - delta_h_pred.median(dim=0, keepdim=True).values
                    
                delta_h_tgt_norm = (delta_h_target - mu_c) / sigma_c
                delta_h_pred_norm = (delta_h_pred - mu_c) / sigma_c
                l_p = F.smooth_l1_loss(delta_h_pred_norm.float(), delta_h_tgt_norm.float(), beta=1.0)
                val_pred.append(l_p.item())
                
        tr_l = np.mean(train_loss)
        tr_p = np.mean(train_pred)
        vl_p = np.mean(val_pred)
        print(f"  Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {tr_l:.4f} | Pred Loss: {tr_p:.4f} | Val Pred Loss: {vl_p:.4f}", flush=True)
        history.append({"epoch": ep + 1, "train_loss": tr_l, "train_pred": tr_p, "val_pred": vl_p})

    # Save Checkpoint
    save_dir = f"experiments/5x5/pilot_reformed/checkpoints/branch_{branch}_seed_{seed}_ep_{epochs}"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "branch": branch,
        "seed": seed,
        "epochs": epochs,
        "mu_c": mu_c.cpu(),
        "sigma_c": sigma_c.cpu(),
        "history": history
    }, ckpt_path)
    print(f"  [Saved Checkpoint to {ckpt_path}]", flush=True)
    return ckpt_path, vl_p, mu_c, sigma_c


def main():
    seeds = [42, 100, 2024, 777, 999]
    
    # 1. Branch A: 5 seeds x 3 epochs
    print("\n" + "=" * 80)
    print("STARTING PILOT BRANCH A: 5 SEEDS x 3 EPOCHS")
    print("=" * 80)
    mu_c, sigma_c = None, None
    branch_a_3ep_ckpts = []
    for s in seeds:
        ckpt, val_loss, mu_c, sigma_c = train_pilot(branch="A", seed=s, epochs=3, mu_c=mu_c, sigma_c=sigma_c)
        branch_a_3ep_ckpts.append((s, ckpt, val_loss))
        
    # 2. Branch A: 1 convergence run (seed 42, 20 epochs)
    print("\n" + "=" * 80)
    print("STARTING PILOT BRANCH A: SEED 42 x 20 EPOCHS (CONVERGENCE RUN)")
    print("=" * 80)
    train_pilot(branch="A", seed=42, epochs=20, mu_c=mu_c, sigma_c=sigma_c)
    
    # 3. Branch B: 5 seeds x 3 epochs
    print("\n" + "=" * 80)
    print("STARTING PILOT BRANCH B: 5 SEEDS x 3 EPOCHS")
    print("=" * 80)
    mu_b, sigma_b = None, None
    for s in seeds:
        ckpt, val_loss, mu_b, sigma_b = train_pilot(branch="B", seed=s, epochs=3, mu_c=mu_b, sigma_c=sigma_b)
        
    print("\n[ALL PILOT TRAINING COMPLETE!]")


if __name__ == "__main__":
    main()
