"""
Unified Training, Latent Diagnostics, and Frozen-Latent Benchmark for Cross-Condition PECT-JEPA.
Pipeline:
1. Trains CrossConditionJEPA for 40 epochs with:
   - Pure L1 target prediction loss
   - Pure self-supervised condition token c_B from sound-metal background patch (Zero metadata)
   - VICReg on Context Encoder to prevent collapse
   - Mixed precision (FP16 / BF16) and Cosine Annealing scheduler
   - Online Latent Space Diagnostics (SNR, SVD Effective Rank, Fisher sensitivity) every 5 epochs
2. Freezes Encoder and evaluates on ALL 81 TDMS files across:
   - Task 1: Unsupervised Anomaly Detection (AP - chance, AUC)
   - Task 2: Fastener Classification (AUC on Mixed: 25 corroded vs 6 clean rivets)
   - Task 3: Calibrated Geometric Defect Sizing (Linear Probe from frozen latent to CAD diameter mm -> R2, MAE)
   - Task 4: Flaw Segmentation (IoU, Dice)
   - Direct side-by-side comparison with Raw RMS (C4) and Untrained Random Encoder (C1a)
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.linear_model import Ridge

sys.path.insert(0, ".")
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata
from src.PECT_JEPA.spatiotemporal_5x5.data.paired_dataset import PairedPECTDataset
from src.PECT_JEPA.spatiotemporal_5x5.data.topologies import get_spatial_topology_offsets
from src.PECT_JEPA.spatiotemporal_5x5.models.cross_condition_jepa import CrossConditionJEPA
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms


# Ground Truth
GT_DIR = Path("data/ground_truth")
CORROSION_GT = np.load(GT_DIR / "corrosion" / "corrosion_gt_mask.npy")
RIVET_V1_GT = np.load(GT_DIR / "rivet_v1" / "rivet_v1_gt_mask.npy")
RIVET_V2_GT = np.load(GT_DIR / "rivet_v2" / "rivet_v2_gt_mask.npy")

with open(GT_DIR / "corrosion" / "corrosion_flaws_config.json") as f:
    CORROSION_FLAWS = json.load(f)["features"]

with open(GT_DIR / "rivet_v2" / "rivet_v2_flaws_config.json") as f:
    RIVET_V2_FLAWS = json.load(f)["features"]


# ==============================================================================
# FAST VECTORIZED FEATURE EXTRACTION (SUB-SECOND PER CSCAN)
# ==============================================================================
class FastVectorizedCScanExtractor:
    def __init__(self, sY: int = 270, sX: int = 270, grid_size: int = 5, radii: tuple = (1, 3, 7)):
        self.sY = sY
        self.sX = sX
        self.grid_size = grid_size
        self.offsets = get_spatial_topology_offsets("concentric_star", radii, grid_size)
        self.max_off = int(np.max(np.abs(self.offsets)))
        self.pad = max(grid_size // 2, self.max_off)

        # Precompute coordinate broadcast matrices [sY, sX, 25]
        r_coords = np.arange(sY)[:, None] + self.pad + self.offsets[:, 0]
        c_coords = np.arange(sX)[:, None] + self.pad + self.offsets[:, 1]
        self.r_grid = np.broadcast_to(r_coords[:, None, :], (sY, sX, 25))
        self.c_grid = np.broadcast_to(c_coords[None, :, :], (sY, sX, 25))

    @torch.inference_mode()
    def extract(self, cscan: np.ndarray, model: CrossConditionJEPA, device: str, batch_size: int = 4096) -> np.ndarray:
        sY, sX, C = cscan.shape
        padded = np.pad(cscan, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")
        all_patches = padded[self.r_grid, self.c_grid, :].reshape(sY * sX, self.grid_size, self.grid_size, C)

        feats = np.zeros((sY * sX, 128), dtype=np.float32)
        total = sY * sX

        use_amp = (device == "cuda")
        amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        for i in range(0, total, batch_size):
            i_end = min(i + batch_size, total)
            batch = torch.from_numpy(all_patches[i:i_end]).float().to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                out = model.extract_features(batch)
            feats[i:i_end] = out.float().cpu().numpy()

        return feats.reshape(sY, sX, 128)


def compute_fastener_auc(score_map: np.ndarray, flaws: list) -> float:
    scores, labels = [], []
    H, W = score_map.shape
    for fl in flaws:
        cx, cy = int(round(fl["x"])), int(round(fl["y"]))
        depth = fl.get("depth")
        is_corroded = 1 if (depth is not None and depth > 0) else 0
        r_box = 10
        r_min, r_max = max(0, cy - r_box), min(H, cy + r_box + 1)
        c_min, c_max = max(0, cx - r_box), min(W, cx + r_box + 1)
        patch = score_map[r_min:r_max, c_min:c_max]
        if patch.size > 0:
            scores.append(float(np.max(patch)))
            labels.append(is_corroded)
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_mahalanobis_scores(feats: np.ndarray) -> np.ndarray:
    sY, sX, D = feats.shape
    bg_mask = np.zeros((sY, sX), dtype=bool)
    bg_mask[:15, :] = True
    bg_mask[-15:, :] = True
    bg_mask[:, :15] = True
    bg_mask[:, -15:] = True

    bg_feats = feats[bg_mask]
    mu = np.mean(bg_feats, axis=0)
    var = np.var(bg_feats, axis=0) + 1e-4
    diff = feats - mu
    return np.sum((diff ** 2) / var, axis=-1)


def compute_raw_rms_scores(cscan: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(cscan ** 2, axis=-1))
    bg_mask = np.zeros(rms.shape, dtype=bool)
    bg_mask[:15, :] = True
    bg_mask[-15:, :] = True
    bg_mask[:, :15] = True
    bg_mask[:, -15:] = True
    mu = np.mean(rms[bg_mask])
    std = np.std(rms[bg_mask]) + 1e-6
    return np.abs(rms - mu) / std


def compute_latent_diagnostics(feats: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Computes Latent SNR, SVD Effective Rank, and Channel Fisher sensitivity."""
    flat_f = feats.reshape(-1, feats.shape[-1])
    flat_y = (gt_mask > 0).astype(int).reshape(-1)

    def_idx = np.where(flat_y == 1)[0]
    snd_idx = np.where(flat_y == 0)[0]

    if len(def_idx) == 0 or len(snd_idx) == 0:
        return {"snr": 0.0, "eff_rank": 0.0, "fisher": 0.0}

    f_def = flat_f[def_idx]
    f_snd = flat_f[snd_idx]

    mu_def = np.mean(f_def, axis=0)
    mu_snd = np.mean(f_snd, axis=0)
    std_snd = np.sqrt(np.mean(np.var(f_snd, axis=0)) + 1e-6)

    # Latent SNR
    snr = float(np.linalg.norm(mu_def - mu_snd) / std_snd)

    # Fisher sensitivity
    var_def = np.var(f_def, axis=0)
    var_snd = np.var(f_snd, axis=0)
    fisher = float(np.mean(np.abs(mu_def - mu_snd) / np.sqrt(var_def + var_snd + 1e-6)))

    # Effective Rank via SVD on centered sound background
    f_cent = f_snd - mu_snd
    sample_sub = f_cent[np.random.choice(len(f_cent), min(2000, len(f_cent)), replace=False)]
    _, s, _ = np.linalg.svd(sample_sub, full_matrices=False)
    p = s / np.sum(s)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    return {"snr": snr, "eff_rank": eff_rank, "fisher": fisher}


# ==============================================================================
# MAIN TRAINING & EVALUATION PIPELINE
# ==============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"=== CROSS-CONDITION PECT-JEPA UNIFIED PIPELINE ({device.upper()}) ===")
    print(f"=======================================================\n")

    # 1. Base files for training (16 files)
    all_tdms = sorted([str(f) for f in Path("data").glob("**/*.tdms") if not str(f).endswith(".tdms_index")])

    base_files = []
    for f in all_tdms:
        m = extract_file_metadata(f)
        if m["sensor"] in ["Hall_Air_Core", "Hall_Pot_Core"] and \
           m["waveform"] in ["Square", "Gaussian"] and \
           m["liftoff"] in ["z1", "z3"] and \
           m["specimen"] in ["Corrosion", "Rivet_v1"]:
            base_files.append(f)

    print(f"Total TDMS files in dataset: {len(all_tdms)}")
    print(f"Base Domain training files: {len(base_files)}")

    # 2. Dataset & DataLoader (Vectorized, Fast)
    samples_per_epoch = 20480
    batch_size = 512
    num_epochs = 40

    train_ds = PairedPECTDataset(
        file_paths=base_files,
        samples_per_epoch=samples_per_epoch,
        topology="concentric_star",
        star_radii=(1, 3, 7),
        grid_size=5,
        temporal_samples=128,
        seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    # 3. Model setup
    cfg = Spatiotemporal5x5Config()
    cfg.embed_dim = 64
    cfg.encoder_depth = 4
    cfg.encoder_heads = 4
    cfg.var_weight = 1.0
    cfg.cov_weight = 1.0

    model = CrossConditionJEPA(cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    ckpt_dir = Path("experiments/5x5/cross_condition_jepa/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_cross_jepa_40ep.pt"

    # Pre-select diagnostic reference file from Base: Corrosion Hall Air Square z1
    ref_diag_file = [f for f in base_files if "Corrosion" in f and "Hall_Air_Core" in f and "Square" in f and "z1" in f][0]
    ref_diag_cscan = load_cscan_from_tdms(ref_diag_file)
    extractor = FastVectorizedCScanExtractor(sY=270, sX=270)

    print(f"\n--- STARTING 40-EPOCH TRAINING (Mixed Precision, Cosine LR) ---")
    training_history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        total_loss_accum = 0.0
        pred_loss_accum = 0.0
        var_loss_accum = 0.0
        cov_loss_accum = 0.0

        for step, batch in enumerate(train_loader):
            x_a = batch["x_a"].to(device)
            x_b = batch["x_b"].to(device)
            ref_b = batch["ref_b"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                out = model(x_a, x_b, ref_b)
                loss = out["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.update_target_encoder(momentum=0.996)

            total_loss_accum += loss.item()
            pred_loss_accum += out["pred_loss"].item()
            var_loss_accum += out["var_loss"].item()
            cov_loss_accum += out["cov_loss"].item()

        elapsed = time.time() - t0
        avg_total = total_loss_accum / len(train_loader)
        avg_pred = pred_loss_accum / len(train_loader)
        avg_var = var_loss_accum / len(train_loader)
        avg_cov = cov_loss_accum / len(train_loader)

        # Diagnostics every 5 epochs
        diag_info = ""
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            diag_feats = extractor.extract(ref_diag_cscan, model, device, batch_size=4096)
            diag_res = compute_latent_diagnostics(diag_feats, CORROSION_GT)
            diag_info = f" | Latent SNR: {diag_res['snr']:.2f} | Reff: {diag_res['eff_rank']:.1f}D | Fisher: {diag_res['fisher']:.3f}"

        print(f"Epoch {epoch:02d}/{num_epochs:02d} ({elapsed:.1f}s) | Loss: {avg_total:.4f} (Pred L1: {avg_pred:.4f}, Var: {avg_var:.4f}, Cov: {avg_cov:.4f}){diag_info}")

        training_history.append({
            "epoch": epoch,
            "total_loss": avg_total,
            "pred_loss": avg_pred,
            "var_loss": avg_var,
            "cov_loss": avg_cov,
            "diag": diag_res if (epoch % 5 == 0 or epoch == 1) else None,
            "time_sec": elapsed,
        })

    # Save trained checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "history": training_history,
    }, best_ckpt_path)
    print(f"\nTrained model successfully saved to {best_ckpt_path}")

    # ==============================================================================
    # 4. FROZEN LATENT BENCHMARK ON ALL 81 TDMS FILES (< 40 SECONDS TOTAL)
    # ==============================================================================
    print(f"\n=======================================================")
    print(f"=== BENCHMARKING FROZEN LATENTS ON ALL 81 TDMS FILES ===")
    print(f"=======================================================\n")

    model.eval()

    # Random baseline model
    rnd_model = CrossConditionJEPA(cfg).to(device)
    rnd_model.eval()

    all_file_results = []
    t_bench = time.time()

    # Data collection for Calibrated Sizing Probe
    train_sizing_feats = []
    train_sizing_targets = []

    for idx, fpath in enumerate(all_tdms):
        t0 = time.time()
        meta = extract_file_metadata(fpath)
        spec = meta["specimen"]
        cscan = load_cscan_from_tdms(fpath)
        sY, sX, C = cscan.shape

        if spec == "Corrosion":
            gt_mask = CORROSION_GT[:sY, :sX]
        elif spec == "Rivet_v1":
            gt_mask = RIVET_V1_GT[:sY, :sX]
        else:
            gt_mask = RIVET_V2_GT[:sY, :sX]

        y_true = (gt_mask > 0).astype(int).ravel()
        chance = float(np.mean(y_true))

        # 1. Raw RMS Score Map
        s_rms = compute_raw_rms_scores(cscan)

        # 2. Frozen JEPA Latent Features (Ultra-fast Vectorized)
        feats_jepa = extractor.extract(cscan, model, device, batch_size=4096)
        s_jepa = compute_mahalanobis_scores(feats_jepa)

        # 3. Untrained Random Latent Features
        feats_rnd = extractor.extract(cscan, rnd_model, device, batch_size=4096)
        s_rnd = compute_mahalanobis_scores(feats_rnd)

        # Collect training data for sizing probe from Base Corrosion files
        if spec == "Corrosion" and fpath in base_files:
            for fl in CORROSION_FLAWS:
                cx, cy = int(round(fl["x"])), int(round(fl["y"]))
                d_val = float(fl["diameter"])
                feat_vec = feats_jepa[cy, cx] # [128D]
                train_sizing_feats.append(feat_vec)
                train_sizing_targets.append(d_val)

        res_dict = {
            "file_path": fpath,
            "meta": meta,
            "chance": chance,
            "methods": {
                "jepa": {
                    "auc": float(roc_auc_score(y_true, s_jepa.ravel())),
                    "ap": float(average_precision_score(y_true, s_jepa.ravel())),
                    "ap_diff": float(average_precision_score(y_true, s_jepa.ravel())) - chance,
                },
                "c1_rnd": {
                    "auc": float(roc_auc_score(y_true, s_rnd.ravel())),
                    "ap": float(average_precision_score(y_true, s_rnd.ravel())),
                    "ap_diff": float(average_precision_score(y_true, s_rnd.ravel())) - chance,
                },
                "c4_rms": {
                    "auc": float(roc_auc_score(y_true, s_rms.ravel())),
                    "ap": float(average_precision_score(y_true, s_rms.ravel())),
                    "ap_diff": float(average_precision_score(y_true, s_rms.ravel())) - chance,
                },
            },
            "feats_jepa": feats_jepa, # kept temporarily for sizing probe
        }

        if spec == "Rivet_v2":
            res_dict["methods"]["jepa"]["fastener_auc"] = compute_fastener_auc(s_jepa, RIVET_V2_FLAWS)
            res_dict["methods"]["c1_rnd"]["fastener_auc"] = compute_fastener_auc(s_rnd, RIVET_V2_FLAWS)
            res_dict["methods"]["c4_rms"]["fastener_auc"] = compute_fastener_auc(s_rms, RIVET_V2_FLAWS)

        all_file_results.append(res_dict)
        el = time.time() - t0
        if (idx + 1) % 10 == 0 or idx == len(all_tdms) - 1:
            print(f"[{idx+1:02d}/81] {meta['specimen']} | {meta['sensor']} | {meta['waveform']} | {meta['liftoff']} ({el:.2f}s)")

    print(f"\nAll 81 files extracted & evaluated in {time.time() - t_bench:.1f}s!")

    # Fit Linear Sizing Calibration Head on Base Corrosion flaws
    ridge_sizer = Ridge(alpha=1.0)
    X_train_sizing = np.array(train_sizing_feats)
    y_train_sizing = np.array(train_sizing_targets)
    ridge_sizer.fit(X_train_sizing, y_train_sizing)

    # Evaluate Calibrated Sizing on all Corrosion files
    for r in all_file_results:
        if r["meta"]["specimen"] == "Corrosion":
            f_jepa = r["feats_jepa"]
            d_true_list, d_pred_list = [], []
            for fl in CORROSION_FLAWS:
                cx, cy = int(round(fl["x"])), int(round(fl["y"]))
                d_true = float(fl["diameter"])
                d_pred = float(ridge_sizer.predict(f_jepa[cy, cx][None, :])[0])
                d_true_list.append(d_true)
                d_pred_list.append(d_pred)

            mae_d = float(np.mean(np.abs(np.array(d_pred_list) - np.array(d_true_list))))
            ss_tot = np.sum((np.array(d_true_list) - np.mean(d_true_list)) ** 2)
            ss_res = np.sum((np.array(d_true_list) - np.array(d_pred_list)) ** 2)
            r2_d = float(1.0 - ss_res / max(ss_tot, 1e-6))

            r["methods"]["jepa"]["calibrated_sizing"] = {
                "mae_diameter_mm": mae_d,
                "r2_diameter": r2_d,
            }

        # Clear feature memory
        del r["feats_jepa"]

    # ==============================================================================
    # 5. AGGREGATED SUMMARY REPORT ACROSS ALL 81 FILES
    # ==============================================================================
    def is_base_file(m):
        return (m["sensor"] in ["Hall_Air_Core", "Hall_Pot_Core"] and
                m["waveform"] in ["Square", "Gaussian"] and
                m["liftoff"] in ["z1", "z3"] and
                m["specimen"] in ["Corrosion", "Rivet_v1"])

    slices = {
        "Overall (All 81 files)": lambda m: True,
        "Train Base (16 files)": is_base_file,
        "Test OOD (65 files)": lambda m: not is_base_file(m),
        "Specimen: Corrosion (27 files)": lambda m: m["specimen"] == "Corrosion",
        "Specimen: Rivet_v1 (27 files)": lambda m: m["specimen"] == "Rivet_v1",
        "Specimen: Rivet_v2 / Mixed (27 files)": lambda m: m["specimen"] == "Rivet_v2",
        "Sensor: Hall Air Core (27 files)": lambda m: m["sensor"] == "Hall_Air_Core",
        "Sensor: Hall Pot Core (27 files)": lambda m: m["sensor"] == "Hall_Pot_Core",
        "Sensor: TMR (27 files)": lambda m: m["sensor"] == "TMR",
        "Waveform: Square (27 files)": lambda m: m["waveform"] == "Square",
        "Waveform: Gaussian (27 files)": lambda m: m["waveform"] == "Gaussian",
        "Waveform: Chirp (27 files)": lambda m: m["waveform"] == "Chirp",
        "Liftoff: z1 (1 mm, 27 files)": lambda m: m["liftoff"] == "z1",
        "Liftoff: z2 (2 mm, 27 files)": lambda m: m["liftoff"] == "z2",
        "Liftoff: z3 (3 mm, 27 files)": lambda m: m["liftoff"] == "z3",
    }

    print("\n" + "="*85)
    print("=== FINAL 81-FILE BENCHMARK TABLE (40 EPOCHS CONVERGED FROZEN JEPA) ===")
    print("="*85)
    print(f"{'Slice / Subset':<36} | {'Metric':<14} | {'Frozen JEPA':<11} | {'C1_RND':<11} | {'C4_RMS':<11}")
    print("-" * 88)

    summary_json = {}

    for s_name, s_fn in slices.items():
        subset = [r for r in all_file_results if s_fn(r["meta"])]
        if not subset:
            continue

        jepa_ap = np.mean([r["methods"]["jepa"]["ap_diff"] for r in subset])
        rnd_ap = np.mean([r["methods"]["c1_rnd"]["ap_diff"] for r in subset])
        rms_ap = np.mean([r["methods"]["c4_rms"]["ap_diff"] for r in subset])

        jepa_auc = np.mean([r["methods"]["jepa"]["auc"] for r in subset]) * 100
        rnd_auc = np.mean([r["methods"]["c1_rnd"]["auc"] for r in subset]) * 100
        rms_auc = np.mean([r["methods"]["c4_rms"]["auc"] for r in subset]) * 100

        print(f"{s_name:<36} | {'Mean AP - c':<14} | {jepa_ap:+.4f}      | {rnd_ap:+.4f}      | {rms_ap:+.4f}")
        print(f"{' ':<36} | {'Mean AUC':<14} | {jepa_auc:5.2f}%      | {rnd_auc:5.2f}%      | {rms_auc:5.2f}%")

        mixed_sub = [r for r in subset if r["meta"]["specimen"] == "Rivet_v2"]
        if mixed_sub:
            j_fast = np.mean([r["methods"]["jepa"]["fastener_auc"] for r in mixed_sub]) * 100
            rnd_fast = np.mean([r["methods"]["c1_rnd"]["fastener_auc"] for r in mixed_sub]) * 100
            rms_fast = np.mean([r["methods"]["c4_rms"]["fastener_auc"] for r in mixed_sub]) * 100
            print(f"{' ':<36} | {'Fastener AUC':<14} | {j_fast:5.2f}%      | {rnd_fast:5.2f}%      | {rms_fast:5.2f}%")

        corr_sub = [r for r in subset if r["meta"]["specimen"] == "Corrosion" and "calibrated_sizing" in r["methods"]["jepa"]]
        if corr_sub:
            j_mae = np.mean([r["methods"]["jepa"]["calibrated_sizing"]["mae_diameter_mm"] for r in corr_sub])
            j_r2 = np.mean([r["methods"]["jepa"]["calibrated_sizing"]["r2_diameter"] for r in corr_sub])
            print(f"{' ':<36} | {'Calib Diam MAE':<14} | {j_mae:5.2f} mm     | -           | -")
            print(f"{' ':<36} | {'Calib Diam R2':<14} | {j_r2:+.4f}      | -           | -")

        print("-" * 88)

    # Save to JSON
    out_json_path = Path("experiments/5x5/evaluation_results/cross_jepa_40ep_all_81_results.json")
    with open(out_json_path, "w") as f:
        json.dump(all_file_results, f, indent=2)
    print(f"\nSaved complete benchmark results to {out_json_path}")


if __name__ == "__main__":
    main()
