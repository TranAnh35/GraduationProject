"""
Official Evaluation Script for Cross-Condition PECT-JEPA across all 81 files.

Evaluates all Downstream NDT Benchmark Tasks on Frozen Foundation Latents:
1. Task 1: Binary Defect Detection Probe (Linear Probe & MLP 2-layer): AUC-ROC, Average Precision, F1, Accuracy.
2. Task 1b: Boundary Contours & IoU (Jaccard Index), Dice coefficient.
3. Task 2: Physical Flaw Sizing Hurdle Protocol (Defect Diameter R^2, MAE in mm, RMSE in mm).
4. Task 3: Unsupervised Mahalanobis Anomaly Distance & Contrast-to-Noise Ratio (CNR).
5. Fastener Collar Discrimination on Rivet specimens.

Outputs:
- Dedicated timestamped results folder: experiments/cross_condition/evaluations/<eval_name>_<YYYYMMDD_HHMMSS>/
- High-resolution C-scan anomaly heatmaps.
- Sizing calibration scatter plots.
- Structured metrics CSV & Summary JSON.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import torch
from tqdm import tqdm

from .configs.config import CrossConditionConfig
from .models.cross_condition_jepa import CrossConditionJEPA
from .data.preprocessing import load_canonical_cscan_from_tdms
from .data.split import extract_file_metadata
from .evaluation.cscan_extractor import CScanFeatureExtractor
from .evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from .evaluation.visualizations import (
    plot_cscan_anomaly_map,
    plot_sizing_calibration,
    plot_probability_heatmap,
)


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
    parser = argparse.ArgumentParser(description="Official Comprehensive Evaluation of Cross-Condition PECT-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--eval_name", type=str, default="benchmark_all_81", help="Evaluation experiment tag")
    parser.add_argument("--add_timestamp", type=lambda v: v.lower() == "true", default=True, help="Append timestamp to evaluation directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Explicit output directory (overrides automatic runs directory)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Resolve output directory with timestamp
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{args.eval_name}_{timestamp}" if args.add_timestamp else args.eval_name
        out_dir = os.path.join("experiments", "cross_condition", "evaluations", run_tag)
    else:
        out_dir = args.out_dir

    viz_dir = os.path.join(out_dir, "heatmaps")
    calib_dir = os.path.join(out_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Cross-Condition JEPA Full Multi-Task Benchmark Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {out_dir}")
    print(f"Device: {device}")
    print(f"=======================================================\n", flush=True)

    # 1. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", CrossConditionConfig())
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Initialize extractor, ground truth, and benchmark suite
    extractor = CScanFeatureExtractor(model=model, device=device, batch_size=4096)
    suite = DownstreamBenchmarkSuite(n_splits=5, random_state=42)
    gt_masks = load_ground_truth_masks()
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])

    # 3. Find all TDMS files
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and not "checkpoint" in f]
    print(f"Total files discovered across all splits: {len(all_files)}\n", flush=True)

    results = []
    pbar = tqdm(all_files, desc="Evaluating All Inspection Files", file=sys.stdout, dynamic_ncols=True)

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

        # Extract frozen representations
        latents = extractor.extract_full_cscan(cscan)

        # Task 1 & 4: Anomaly Detection (Mahalanobis distance)
        anom_res = suite.compute_anomaly_detection_metrics(latents=latents, ground_truth_mask=gt_mask)

        # Task 1: Linear Probe & MLP 2-layer Probing
        probe_res = suite.benchmark_binary_detection(features=latents, labels=gt_mask)
        lp_auc = probe_res.get("linear_probe", {}).get("auc_roc", anom_res["auc_roc"])
        lp_ap = probe_res.get("linear_probe", {}).get("average_precision", anom_res["average_precision"])
        mlp_auc = probe_res.get("mlp_2layer", {}).get("auc_roc", 0.0)
        mlp_ap = probe_res.get("mlp_2layer", {}).get("average_precision", 0.0)

        # Task 1b: Segmentation IoU
        # Normalize anomaly scores to [0, 1] probability proxy
        norm_scores = (anom_res["anomaly_scores"] - np.min(anom_res["anomaly_scores"])) / (
            np.ptp(anom_res["anomaly_scores"]) + 1e-6
        )
        iou_res = suite.benchmark_segmentation_iou(prob_map=norm_scores, gt_mask=gt_mask)

        # Task 2: Physical Flaw Sizing Hurdle Protocol (Corrosion)
        sizing_r2, sizing_mae, sizing_rmse = 0.0, 0.0, 0.0
        mlp_sizing_r2, mlp_sizing_mae = 0.0, 0.0
        if specimen == "Corrosion" and corr_d_map is not None:
            sizing_res = suite.benchmark_sizing_regression(
                features=latents,
                size_map=corr_d_map,
                focus_defects_only=True,
            )
            if "linear_probe" in sizing_res:
                sizing_r2 = sizing_res["linear_probe"]["r2_score"]
                sizing_mae = sizing_res["linear_probe"]["mae_mm"]
                sizing_rmse = sizing_res["linear_probe"]["rmse_mm"]
            if "mlp_2layer" in sizing_res:
                mlp_sizing_r2 = sizing_res["mlp_2layer"]["r2_score"]
                mlp_sizing_mae = sizing_res["mlp_2layer"]["mae_mm"]

        row = {
            "file": file_name,
            "sensor": sensor,
            "specimen": specimen,
            "waveform": waveform,
            "liftoff": liftoff,
            "mahalanobis_auc": anom_res["auc_roc"],
            "mahalanobis_ap": anom_res["average_precision"],
            "cnr": anom_res["cnr"],
            "linear_probe_auc": lp_auc,
            "linear_probe_ap": lp_ap,
            "mlp_probe_auc": mlp_auc,
            "mlp_probe_ap": mlp_ap,
            "segmentation_iou": iou_res["iou"],
            "segmentation_dice": iou_res["dice"],
            "sizing_r2": sizing_r2,
            "sizing_mae_mm": sizing_mae,
            "sizing_rmse_mm": sizing_rmse,
            "mlp_sizing_r2": mlp_sizing_r2,
            "mlp_sizing_mae_mm": mlp_sizing_mae,
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
            "LP_AUC": f"{lp_auc:.4f}",
            "Mahal_AUC": f"{anom_res['auc_roc']:.4f}",
            "CNR": f"{anom_res['cnr']:.2f}",
        })

    # Save CSV Report
    csv_path = os.path.join(out_dir, "benchmark_metrics.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute Summary Statistics
    lp_auc_list = [r["linear_probe_auc"] for r in results]
    lp_ap_list = [r["linear_probe_ap"] for r in results]
    mahal_auc_list = [r["mahalanobis_auc"] for r in results]
    cnr_list = [r["cnr"] for r in results]
    iou_list = [r["segmentation_iou"] for r in results]

    summary = {
        "num_files_evaluated": len(results),
        "mean_linear_probe_auc": float(np.mean(lp_auc_list)),
        "std_linear_probe_auc": float(np.std(lp_auc_list)),
        "mean_linear_probe_ap": float(np.mean(lp_ap_list)),
        "mean_mahalanobis_auc": float(np.mean(mahal_auc_list)),
        "mean_cnr": float(np.mean(cnr_list)),
        "mean_segmentation_iou": float(np.mean(iou_list)),
    }

    # Summary by Specimen
    for sp in ["Corrosion", "Rivet_v1", "Rivet_v2"]:
        sp_sub = [r for r in results if r["specimen"] == sp]
        if sp_sub:
            summary[f"{sp}_mean_linear_probe_auc"] = float(np.mean([r["linear_probe_auc"] for r in sp_sub]))
            summary[f"{sp}_mean_mahalanobis_auc"] = float(np.mean([r["mahalanobis_auc"] for r in sp_sub]))
            summary[f"{sp}_mean_cnr"] = float(np.mean([r["cnr"] for r in sp_sub]))
            if sp == "Corrosion":
                summary["Corrosion_mean_sizing_r2"] = float(np.mean([r["sizing_r2"] for r in sp_sub]))
                summary["Corrosion_mean_sizing_mae_mm"] = float(np.mean([r["sizing_mae_mm"] for r in sp_sub]))

    # Summary by Waveform
    for wf in ["Chirp", "Gaussian", "Square"]:
        wf_sub = [r for r in results if r["waveform"] == wf]
        if wf_sub:
            summary[f"waveform_{wf}_mean_linear_probe_auc"] = float(np.mean([r["linear_probe_auc"] for r in wf_sub]))
            summary[f"waveform_{wf}_mean_mahalanobis_auc"] = float(np.mean([r["mahalanobis_auc"] for r in wf_sub]))

    # Summary by Sensor
    for sn in ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]:
        sn_sub = [r for r in results if r["sensor"] == sn]
        if sn_sub:
            summary[f"sensor_{sn}_mean_linear_probe_auc"] = float(np.mean([r["linear_probe_auc"] for r in sn_sub]))
            summary[f"sensor_{sn}_mean_mahalanobis_auc"] = float(np.mean([r["mahalanobis_auc"] for r in sn_sub]))

    json_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"BENCHMARK EVALUATION SUMMARY")
    print(f"Total Files Evaluated:       {summary['num_files_evaluated']}")
    print(f"Linear Probe Mean AUC-ROC:   {summary['mean_linear_probe_auc']:.4f} (+/- {summary['std_linear_probe_auc']:.4f})")
    print(f"Linear Probe Mean AP:        {summary['mean_linear_probe_ap']:.4f}")
    print(f"Mahalanobis Mean AUC-ROC:    {summary['mean_mahalanobis_auc']:.4f}")
    print(f"Overall Mean CNR:            {summary['mean_cnr']:.2f}")
    print(f"Mean Segmentation IoU:       {summary['mean_segmentation_iou']:.4f}")
    if "Corrosion_mean_sizing_r2" in summary:
        print(f"Corrosion Sizing R^2:        {summary['Corrosion_mean_sizing_r2']:.4f} | Sizing MAE: {summary['Corrosion_mean_sizing_mae_mm']:.2f} mm")
    print(f"Results Directory:           {out_dir}")
    print(f"=======================================================\n", flush=True)


if __name__ == "__main__":
    main()
