"""
Generate Flaw Size Sizing (Task 2b) Calibration Scatter Plots and Maps
for EXP-21 across Corrosion, Rivet, and Mixed Specimens.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch

# Ensure repository root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.evaluate import load_model_from_checkpoint, extract_full_cscan_map
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import load_cscan_from_tdms, extract_full_cscan_map
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.visualizations import apply_spatial_coherence_filter


def plot_flaw_size_scatter(
    true_size: np.ndarray,
    pred_size: np.ndarray,
    save_path: str,
    title: str,
    r2: float,
    mae: float,
    rmse: float,
):
    """Plots true vs predicted flaw size scatter."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter points
    ax.scatter(true_size, pred_size, alpha=0.35, color="#1f77b4", edgecolors="none", s=22, label="Defect Pixels")

    # Ideal diagonal
    min_v = 0.0
    max_v = max(float(np.max(true_size)), float(np.max(pred_size)), 12.0) * 1.05
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2.0, label="Ideal Fit (y = x)")

    # Linear trendline
    if len(true_size) > 5 and np.std(true_size) > 1e-4:
        poly = np.polyfit(true_size, pred_size, 1)
        xs = np.linspace(min_v, max_v, 100)
        ax.plot(xs, np.polyval(poly, xs), "g-", lw=1.8, label=f"Fit: y = {poly[0]:.2f}x + {poly[1]:.2f}")

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("True Flaw Size / Diameter (mm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Flaw Size (mm)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Info text box
    info_str = f"Defect-Only R²: {r2:.4f}\nMAE:  {mae:.4f} mm\nRMSE: {rmse:.4f} mm\nN:    {len(true_size):,}"
    ax.text(
        0.05,
        0.95,
        info_str,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888", alpha=0.9),
    )
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_flaw_size_maps(
    true_size_map: np.ndarray,
    pred_size_map: np.ndarray,
    save_path: str,
    title: str,
    r2: float,
    mae: float,
):
    """Plots true vs predicted flaw size 2D C-scan maps."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmax = max(float(np.max(true_size_map)), float(np.max(pred_size_map)), 10.0)

    # 1. True Size Map
    im0 = axes[0].imshow(true_size_map, cmap="jet", vmin=0.0, vmax=vmax, origin="lower")
    axes[0].set_title("Ground-Truth Flaw Size Map (mm)", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Diameter (mm)")

    # 2. Predicted Size Map
    im1 = axes[1].imshow(pred_size_map, cmap="jet", vmin=0.0, vmax=vmax, origin="lower")
    axes[1].set_title(f"Predicted Flaw Size Map (mm)\nDefect R² = {r2:.3f} | MAE = {mae:.3f} mm", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Diameter (mm)")

    # 3. Residual Error Map
    err_map = np.abs(pred_size_map - true_size_map)
    im2 = axes[2].imshow(err_map, cmap="hot", vmin=0.0, vmax=max(3.0, float(np.percentile(err_map, 99))), origin="lower")
    axes[2].set_title(f"Absolute Residual Error |True - Pred| (mm)\nMean Error: {np.mean(err_map):.3f} mm", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="|Error| (mm)")

    for ax in axes:
        ax.set_xlabel("Scan X (mm)")
        ax.set_ylabel("Scan Y (mm)")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    exp_dir = "experiments/5x5/exp21_snr_tapered_anisotropic_jepa"
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model_5x5.pt")
    out_dir = os.path.join(exp_dir, "evaluation_results")
    task2b_dir = os.path.join(out_dir, "2_Flaw_Size_Sizing")
    os.makedirs(task2b_dir, exist_ok=True)

    print(f"Loading EXP-21 Model from {ckpt_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_checkpoint(ckpt_path, device=device)

    gt_mgr = get_ground_truth_manager(data_dir=getattr(model.config, "data_dir", "data"))

    # Select representative files for each of the 3 specimens
    rep_files = {
        "corrosion": "data/Hall_Air_Core/Corrosion/Chirp/hall_aircore_corosion_frontside_chirp_300x300_2.97_500-1500hz_z1_20260118_162411.tdms",
        "mixed": "data/Hall_Air_Core/Rivet_v2/Chirp/hall_aircore_mixed_frontside_chirp_300x300_z1_20260125_215812.tdms",
        "rivet": "data/Hall_Air_Core/Rivet_v1/Chirp/hall_aircore_rivet_frontside_chirp_300x300_500_1500hz_z1_20260123_202334.tdms",
    }

    all_true_defects = []
    all_pred_defects = []
    specimen_metrics = {}

    for sp_name, fp in rep_files.items():
        if not os.path.isfile(fp):
            print(f"File not found: {fp}, skipping")
            continue

        print(f"\nProcessing Specimen: {sp_name.upper()} ({os.path.basename(fp)})")
        grid_3d = load_cscan_from_tdms(
            fp,
            time_samples=model.config.time_samples,
            temporal_samples=model.config.temporal_samples,
            resample_mode=model.config.resample_mode,
            normalization=model.config.normalization,
            raster_correction=model.config.raster_correction,
            crop_border=15,
            apply_lowpass=getattr(model.config, "apply_lowpass", True),
            lowpass_cutoff=getattr(model.config, "lowpass_cutoff", 2500.0),
            lowpass_order=getattr(model.config, "lowpass_order", 4),
        )

        feat_map = extract_full_cscan_map(model, grid_3d, batch_size=512, device=device, show_pbar=False)

        sp_key = gt_mgr.canonical_specimen_key(fp)
        size_map_gt = gt_mgr.generate_size_map(sp_key, mode="diameter")
        gt_mask = gt_mgr.get_ground_truth_mask_for_file(fp, aligned_scan=True)

        min_Y = min(feat_map.shape[0], size_map_gt.shape[0])
        min_X = min(feat_map.shape[1], size_map_gt.shape[1])

        sub_feat = feat_map[:min_Y, :min_X]
        sub_size = size_map_gt[:min_Y, :min_X]
        sub_gt = gt_mask[:min_Y, :min_X]

        flat_feat = sub_feat.reshape(-1, sub_feat.shape[-1]).astype(np.float32)
        flat_size = sub_size.reshape(-1).astype(np.float32)

        def_idx = np.where(flat_size > 0.0)[0]
        snd_idx = np.where(flat_size == 0.0)[0]

        # Subsample sound metal to balance fit
        rng = np.random.RandomState(42)
        sub_snd = rng.choice(snd_idx, size=min(len(snd_idx), 8000), replace=False)
        fit_idx = np.concatenate([def_idx, sub_snd])

        scaler = StandardScaler()
        X_fit = scaler.fit_transform(flat_feat[fit_idx])
        y_fit = flat_size[fit_idx]

        # Gated Hurdle Flaw Sizing
        clf_gate = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
        clf_gate.fit(X_fit, (y_fit > 0.0).astype(int))

        ridge_cond = Ridge(alpha=1.0, random_state=42)
        ridge_cond.fit(X_fit[y_fit > 0.0], y_fit[y_fit > 0.0])

        # Predict across whole scan
        X_all = scaler.transform(flat_feat)
        p_all = clf_gate.predict_proba(X_all)[:, 1]
        size_cond_all = np.maximum(0.0, ridge_cond.predict(X_all))

        # Spatial Coherence Filtering on Gate
        p_gate_bin = (p_all >= 0.5).reshape(min_Y, min_X).astype(int)
        p_gate_clean = apply_spatial_coherence_filter(p_gate_bin, min_area=8)
        pred_size_flat = np.where(p_gate_clean.reshape(-1) == 1, size_cond_all, 0.0)
        pred_size_map = pred_size_flat.reshape(min_Y, min_X)

        # Defect-only metrics
        true_def = flat_size[def_idx]
        pred_def = pred_size_flat[def_idx]

        r2_def = float(r2_score(true_def, pred_def))
        mae_def = float(mean_absolute_error(true_def, pred_def))
        rmse_def = float(np.sqrt(mean_squared_error(true_def, pred_def)))

        print(f"  [{sp_name.upper()}] Defect-Only Size R² = {r2_def:.4f} | MAE = {mae_def:.4f} mm | RMSE = {rmse_def:.4f} mm")
        specimen_metrics[sp_name] = {"r2": r2_def, "mae": mae_def, "rmse": rmse_def}

        all_true_defects.append(true_def)
        all_pred_defects.append(pred_def)

        # Save Specimen Scatter Plot in both 2_Depth_Regression/<sp>/ and 2_Flaw_Size_Sizing/
        fname_base = os.path.splitext(os.path.basename(fp))[0]
        sp_depth_dir = os.path.join(out_dir, "2_Depth_Regression", sp_key)

        scatter_path_depth = os.path.join(sp_depth_dir, f"{fname_base}_flaw_size_scatter.png")
        scatter_path_sizing = os.path.join(task2b_dir, f"{sp_name}_flaw_size_scatter.png")
        map_path_depth = os.path.join(sp_depth_dir, f"{fname_base}_predicted_flaw_size_map.png")
        map_path_sizing = os.path.join(task2b_dir, f"{sp_name}_predicted_flaw_size_map.png")

        plot_flaw_size_scatter(
            true_size=true_def,
            pred_size=pred_def,
            save_path=scatter_path_depth,
            title=f"Flaw Size / Diameter Calibration Scatter | {sp_name.upper()}\n(Chirp, Lift-off z1, Defect-Only)",
            r2=r2_def,
            mae=mae_def,
            rmse=rmse_def,
        )
        plot_flaw_size_scatter(
            true_size=true_def,
            pred_size=pred_def,
            save_path=scatter_path_sizing,
            title=f"Flaw Size / Diameter Calibration Scatter | {sp_name.upper()}\n(Chirp, Lift-off z1, Defect-Only)",
            r2=r2_def,
            mae=mae_def,
            rmse=rmse_def,
        )

        plot_flaw_size_maps(
            true_size_map=sub_size,
            pred_size_map=pred_size_map,
            save_path=map_path_depth,
            title=f"EXP-21 Flaw Size Quantitative Sizing Map | {sp_name.upper()}",
            r2=r2_def,
            mae=mae_def,
        )
        plot_flaw_size_maps(
            true_size_map=sub_size,
            pred_size_map=pred_size_map,
            save_path=map_path_sizing,
            title=f"EXP-21 Flaw Size Quantitative Sizing Map | {sp_name.upper()}",
            r2=r2_def,
            mae=mae_def,
        )

    # Multi-Specimen Combined Flaw Sizing Scatter Plot
    if all_true_defects:
        tot_true = np.concatenate(all_true_defects)
        tot_pred = np.concatenate(all_pred_defects)
        tot_r2 = float(r2_score(tot_true, tot_pred))
        tot_mae = float(mean_absolute_error(tot_true, tot_pred))
        tot_rmse = float(np.sqrt(mean_squared_error(tot_true, tot_pred)))

        comb_scatter_path = os.path.join(task2b_dir, "overall_flaw_size_r2_scatter_all_specimens.png")
        plot_flaw_size_scatter(
            true_size=tot_true,
            pred_size=tot_pred,
            save_path=comb_scatter_path,
            title="EXP-21 Multi-Specimen Flaw Size Sizing (Corrosion, Rivet, Mixed)",
            r2=tot_r2,
            mae=tot_mae,
            rmse=tot_rmse,
        )
        print(f"\n[OVERALL] Combined Flaw Size Defect R² = {tot_r2:.4f} | MAE = {tot_mae:.4f} mm")

    print("\nTask 2b Flaw Size Plots generation complete!")


if __name__ == "__main__":
    main()
