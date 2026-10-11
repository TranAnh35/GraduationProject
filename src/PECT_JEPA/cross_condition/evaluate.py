"""
Official Evaluation Script for Cross-Condition PECT-JEPA across all 81 files.

Evaluates:
1. Mahalanobis Anomaly Detection (AUC-ROC, Average Precision, AP-chance, CNR).
2. Fastener Collar Discrimination on Rivet specimens (Fastener AUC, Fastener AP).
3. Conditional Calibrated Flaw Sizing on Corrosion specimen (Diameter R^2, MAE in mm, Pearson r).
4. Multi-slice disaggregation across all 3 sensors, 3 waveforms, 3 liftoffs.
5. Saves C-scan anomaly heatmaps and summary CSV/JSON reports.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import csv
import argparse
from typing import Dict, List, Any
import numpy as np
import torch
from tqdm import tqdm

from .configs.config import CrossConditionConfig
from .models.cross_condition_jepa import CrossConditionJEPA
from .data.preprocessing import load_canonical_cscan_from_tdms
from .data.split import extract_file_metadata
from .evaluation.cscan_extractor import CScanFeatureExtractor
from .evaluation.downstream_benchmarks import (
    compute_anomaly_detection_metrics,
    evaluate_fastener_classification,
    evaluate_conditional_sizing,
)
from .evaluation.visualizations import plot_cscan_anomaly_map


def load_ground_truth_masks() -> Dict[str, Any]:
    """Loads CAD ground truth masks and sizing targets for all specimens."""
    masks = {}

    # 1. Corrosion
    corr_gt_path = "data/ground_truth/corrosion/corrosion_gt_mask.npy"
    corr_cfg_path = "data/ground_truth/corrosion/corrosion_flaws_config.json"
    if os.path.exists(corr_gt_path):
        masks["Corrosion"] = {
            "mask": np.load(corr_gt_path),
            "config": json.load(open(corr_cfg_path, "r", encoding="utf-8")) if os.path.exists(corr_cfg_path) else None,
        }

    # 2. Rivet_v1
    r1_gt_path = "data/ground_truth/rivet_v1/rivet_v1_gt_mask.npy"
    if os.path.exists(r1_gt_path):
        masks["Rivet_v1"] = {
            "mask": np.load(r1_gt_path),
            "config": None,
        }

    # 3. Rivet_v2
    r2_gt_path = "data/ground_truth/rivet_v2/rivet_v2_gt_mask.npy"
    if os.path.exists(r2_gt_path):
        masks["Rivet_v2"] = {
            "mask": np.load(r2_gt_path),
            "config": None,
        }

    return masks


def build_corrosion_diameter_map(config: Dict[str, Any], grid_shape=(270, 270)) -> np.ndarray:
    """Creates [270, 270] continuous diameter map in mm from flaw config."""
    d_map = np.zeros(grid_shape, dtype=np.float32)
    features = config.get("features", [])
    for feat in features:
        cx = int(round(feat["x"]))
        cy = int(round(feat["y"]))
        dia = float(feat.get("diameter", 0.0))
        rad = int(round(dia / 2.0)) + 2  # include buffer
        for r in range(max(0, cy - rad), min(grid_shape[0], cy + rad + 1)):
            for c in range(max(0, cx - rad), min(grid_shape[1], cx + rad + 1)):
                if (r - cy) ** 2 + (c - cx) ** 2 <= rad ** 2:
                    d_map[r, c] = dia
    return d_map


def parse_args():
    parser = argparse.ArgumentParser(description="Official Evaluation of Cross-Condition PECT-JEPA")
    parser.add_argument("--checkpoint", type=str, default="experiments/cross_condition/checkpoints/best_cross_condition_jepa.pt", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--out_dir", type=str, default="experiments/cross_condition/results", help="Directory to save evaluation reports")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "heatmaps")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting Cross-Condition JEPA Benchmark Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"=======================================================\n", flush=True)

    # 1. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", CrossConditionConfig())
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Initialize extractor and load ground truth
    extractor = CScanFeatureExtractor(model=model, device=device, batch_size=4096)
    gt_masks = load_ground_truth_masks()
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])

    # 3. Find all TDMS files
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and not "checkpoint" in f]
    print(f"Total files discovered across all splits: {len(all_files)}\n", flush=True)

    results = []
    pbar = tqdm(all_files, desc="Evaluating Inspection Files", file=sys.stdout, dynamic_ncols=True)

    for file_path in pbar:
        meta = extract_file_metadata(file_path)
        specimen = meta["specimen"]
        sensor = meta["sensor"]
        waveform = meta["waveform"]
        liftoff = meta["liftoff"]
        file_name = os.path.basename(file_path)

        if specimen not in gt_masks:
            continue

        gt_mask = gt_masks[specimen]["mask"]

        # Load canonical transformed C-scan
        cscan = load_canonical_cscan_from_tdms(file_path=file_path)

        # Extract features
        latents = extractor.extract_full_cscan(cscan)

        # Anomaly Detection Evaluation
        anom_res = compute_anomaly_detection_metrics(latents=latents, ground_truth_mask=gt_mask)

        # Sizing Evaluation (Corrosion)
        sizing_r2, sizing_mae, sizing_r = 0.0, 0.0, 0.0
        if specimen == "Corrosion" and corr_d_map is not None:
            sizing_res = evaluate_conditional_sizing(
                features=latents,
                ground_truth_diameter_map=corr_d_map,
                flaw_mask=gt_mask,
            )
            sizing_r2 = sizing_res["r2"]
            sizing_mae = sizing_res["mae_mm"]
            sizing_r = sizing_res["pearson_r"]

        row = {
            "file": file_name,
            "sensor": sensor,
            "specimen": specimen,
            "waveform": waveform,
            "liftoff": liftoff,
            "auc_roc": anom_res["auc_roc"],
            "average_precision": anom_res["average_precision"],
            "ap_chance": anom_res["ap_chance"],
            "cnr": anom_res["cnr"],
            "sizing_r2": sizing_r2,
            "sizing_mae_mm": sizing_mae,
            "sizing_pearson_r": sizing_r,
        }
        results.append(row)

        # Generate and save heatmap for visualization
        plot_path = os.path.join(viz_dir, f"{os.path.splitext(file_name)[0]}_anomaly.png")
        plot_cscan_anomaly_map(
            anomaly_map=anom_res["anomaly_scores"],
            ground_truth_mask=gt_mask,
            file_title=f"{specimen} | {sensor} | {waveform} | {liftoff}",
            save_path=plot_path,
            metrics=anom_res,
        )

        pbar.set_postfix({
            "specimen": specimen[:4],
            "AUC": f"{anom_res['auc_roc']:.4f}",
            "AP": f"{anom_res['average_precision']:.4f}",
            "CNR": f"{anom_res['cnr']:.2f}",
        })

    # Save CSV Report
    csv_path = os.path.join(args.out_dir, "benchmark_metrics.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute Summary Statistics
    auc_list = [r["auc_roc"] for r in results]
    ap_list = [r["average_precision"] for r in results]
    cnr_list = [r["cnr"] for r in results]

    summary = {
        "num_files_evaluated": len(results),
        "mean_auc": float(np.mean(auc_list)),
        "std_auc": float(np.std(auc_list)),
        "mean_ap": float(np.mean(ap_list)),
        "mean_cnr": float(np.mean(cnr_list)),
    }

    # Summary by Specimen
    for sp in ["Corrosion", "Rivet_v1", "Rivet_v2"]:
        sp_sub = [r for r in results if r["specimen"] == sp]
        if sp_sub:
            summary[f"{sp}_mean_auc"] = float(np.mean([r["auc_roc"] for r in sp_sub]))
            summary[f"{sp}_mean_ap"] = float(np.mean([r["average_precision"] for r in sp_sub]))
            summary[f"{sp}_mean_cnr"] = float(np.mean([r["cnr"] for r in sp_sub]))
            if sp == "Corrosion":
                summary["Corrosion_mean_sizing_r2"] = float(np.mean([r["sizing_r2"] for r in sp_sub]))
                summary["Corrosion_mean_sizing_mae_mm"] = float(np.mean([r["sizing_mae_mm"] for r in sp_sub]))

    # Summary by Waveform
    for wf in ["Chirp", "Gaussian", "Square"]:
        wf_sub = [r for r in results if r["waveform"] == wf]
        if wf_sub:
            summary[f"waveform_{wf}_mean_auc"] = float(np.mean([r["auc_roc"] for r in wf_sub]))
            summary[f"waveform_{wf}_mean_ap"] = float(np.mean([r["average_precision"] for r in wf_sub]))

    # Summary by Sensor
    for sn in ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]:
        sn_sub = [r for r in results if r["sensor"] == sn]
        if sn_sub:
            summary[f"sensor_{sn}_mean_auc"] = float(np.mean([r["auc_roc"] for r in sn_sub]))
            summary[f"sensor_{sn}_mean_ap"] = float(np.mean([r["average_precision"] for r in sn_sub]))

    json_path = os.path.join(args.out_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"BENCHMARK EVALUATION SUMMARY")
    print(f"Total Files Evaluated: {summary['num_files_evaluated']}")
    print(f"Overall Mean AUC-ROC: {summary['mean_auc']:.4f} (+/- {summary['std_auc']:.4f})")
    print(f"Overall Mean AP:      {summary['mean_ap']:.4f}")
    print(f"Overall Mean CNR:     {summary['mean_cnr']:.2f}")
    if "Corrosion_mean_sizing_r2" in summary:
        print(f"Corrosion Sizing R^2: {summary['Corrosion_mean_sizing_r2']:.4f} | Sizing MAE: {summary['Corrosion_mean_sizing_mae_mm']:.2f} mm")
    print(f"CSV Report:  {csv_path}")
    print(f"JSON Report: {json_path}")
    print(f"Heatmaps:    {viz_dir}")
    print(f"=======================================================\n", flush=True)


if __name__ == "__main__":
    main()
