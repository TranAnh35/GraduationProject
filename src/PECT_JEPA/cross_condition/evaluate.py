"""
Official Downstream Evaluation Suite for Cross-Condition PECT-JEPA.

Strict Global Readout Protocol:
- Linear Probe (Defect Detection) and Sizing Regressors are trained ONLY on the Train split representations.
- Probes are frozen and evaluated across all 3 splits:
  1. Train (In-Domain Reference)
  2. Val (In-Domain Validation)
  3. Test (Zero-Shot Out-of-Distribution Transfer across unseen waveforms, sensors, liftoff, and specimens)

Directory Structure:
  experiments/cross_condition/evaluations/<eval_name>_<timestamp>/
    ├── 1_Anomaly_Detection/
    │     ├── train/ (Mahalanobis maps, split_metrics.csv, split_summary.json)
    │     ├── val/
    │     └── test/
    ├── 2_Defect_Sizing/
    │     ├── train/ (Calibration scatter plots, split_metrics.csv, split_summary.json)
    │     ├── val/
    │     └── test/
    ├── 3_Segmentation_IoU/
    │     ├── train/ (Segmentation comparison plots, split_metrics.csv, split_summary.json)
    │     ├── val/
    │     └── test/
    ├── master_benchmark_metrics.csv
    ├── master_benchmark_summary.json
    └── EVALUATION_REPORT.md
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import CrossConditionJEPA
from src.PECT_JEPA.cross_condition.data.preprocessing import load_canonical_cscan_from_tdms
from src.PECT_JEPA.cross_condition.data.split import (
    extract_file_metadata,
    get_canonical_train_val_test_splits,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.cross_condition.evaluation.visualizations import (
    plot_cscan_anomaly_map,
    plot_sizing_calibration,
    plot_probability_heatmap,
    plot_segmentation_comparison,
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


def build_corrosion_diameter_map(config: Dict[str, Any], grid_shape: Tuple[int, int] = (270, 270)) -> np.ndarray:
    """Creates [270, 270] continuous diameter map in mm from flaw config."""
    d_map = np.zeros(grid_shape, dtype=np.float32)
    features = config.get("features", [])
    for feat in features:
        cx = int(round(feat["x"]))
        cy = int(round(feat["y"]))
        dia = float(feat.get("diameter", 0.0))
        rad = int(round(dia / 2.0))
        for r in range(max(0, cy - rad), min(grid_shape[0], cy + rad + 1)):
            for c in range(max(0, cx - rad), min(grid_shape[1], cx + rad + 1)):
                if (r - cy) ** 2 + (c - cx) ** 2 <= rad ** 2:
                    d_map[r, c] = dia
    return d_map


def parse_args():
    parser = argparse.ArgumentParser(description="Official Downstream Benchmark Evaluation for Cross-Condition PECT-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--eval_name", type=str, default="official_benchmark", help="Evaluation experiment tag")
    parser.add_argument("--add_timestamp", type=lambda v: v.lower() == "true", default=True, help="Append timestamp to evaluation directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Explicit output directory (overrides default evaluations directory)")
    parser.add_argument("--eval_splits", nargs="+", default=["train", "val", "test"], help="Splits to evaluate ('train', 'val', 'test')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for feature extraction")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Resolve output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{args.eval_name}_{timestamp}" if args.add_timestamp else args.eval_name
        out_dir = os.path.join("experiments", "cross_condition", "evaluations", run_tag)
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    task_dirs = {
        "anomaly": os.path.join(out_dir, "1_Anomaly_Detection"),
        "sizing": os.path.join(out_dir, "2_Defect_Sizing"),
        "segmentation": os.path.join(out_dir, "3_Segmentation_IoU"),
    }
    for tdir in task_dirs.values():
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(tdir, sp), exist_ok=True)

    print("\n" + "=" * 70)
    print("PECT-JEPA Official Multi-Task Benchmark Evaluation")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Output Directory: {out_dir}")
    print(f"Compute Device:   {device}")
    print(f"Target Splits:    {args.eval_splits}")
    print("=" * 70 + "\n", flush=True)

    # 2. Load Checkpoint and Initialize Model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", CrossConditionConfig())
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Ground Truth Masks & Feature Extractor
    extractor = CScanFeatureExtractor(model=model, device=device, batch_size=args.batch_size)
    suite = DownstreamBenchmarkSuite(n_splits=5, random_state=42)
    gt_masks = load_ground_truth_masks()
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])

    # 4. Discover All TDMS Files & Canonical Partitioning
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]

    splits = get_canonical_train_val_test_splits(all_files, val_files_count=4, seed=42)
    print(f"Discovered {len(all_files)} total inspection files:")
    print(f"  - Train Split (Base Domain):     {len(splits['train'])} files")
    print(f"  - Val Split (Base Domain):       {len(splits['val'])} files")
    print(f"  - Test Split (Held-out OOD):     {len(splits['test'])} files\n", flush=True)

    # 5. STEP 1: Fit Global Readout Probes Strictly on Train Pool
    print("=" * 70)
    print("STEP 1: Extracting Train Pool Latents & Fitting Global Readout Probes")
    print("=" * 70, flush=True)

    cached_train_latents: Dict[str, np.ndarray] = {}
    train_det_x, train_det_y = [], []
    train_size_x, train_size_y = [], []

    pbar_train = tqdm(splits["train"], desc="Fitting Readouts on Train Pool", file=sys.stdout, dynamic_ncols=True)
    rng = np.random.RandomState(42)

    for fp in pbar_train:
        meta = extract_file_metadata(fp)
        sp = meta["specimen"]
        if sp not in gt_masks:
            continue

        gt_mask = gt_masks[sp]["mask"]
        cscan = load_canonical_cscan_from_tdms(file_path=fp)
        latents = extractor.extract_full_cscan(cscan)
        cached_train_latents[fp] = latents

        flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
        flat_y = gt_mask.reshape(-1).astype(np.int64)

        # Defect Detection Points (all defect + balanced sound subsample)
        pos_idx = np.where(flat_y == 1)[0]
        neg_idx = np.where(flat_y == 0)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            sub_neg_count = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
            sub_neg = rng.choice(neg_idx, size=sub_neg_count, replace=False)
            sel = np.concatenate([pos_idx, sub_neg])
            train_det_x.append(flat_z[sel])
            train_det_y.append(flat_y[sel])

        # Sizing Points (from Corrosion specimens)
        if sp == "Corrosion" and corr_d_map is not None:
            flat_d = corr_d_map.reshape(-1).astype(np.float32)
            def_idx = np.where(flat_d > 0.0)[0]
            if len(def_idx) > 0:
                train_size_x.append(flat_z[def_idx])
                train_size_y.append(flat_d[def_idx])

    X_train_det = np.concatenate(train_det_x, axis=0)
    y_train_det = np.concatenate(train_det_y, axis=0)
    X_train_size = np.concatenate(train_size_x, axis=0) if train_size_x else None
    y_train_size = np.concatenate(train_size_y, axis=0) if train_size_y else None

    print(f"\n  [Train Pool Aggregation]")
    print(f"  - Defect Detection samples: {len(y_train_det)} (Defects: {np.sum(y_train_det == 1)}, Sound: {np.sum(y_train_det == 0)})")
    if y_train_size is not None:
        print(f"  - Flaw Sizing samples:      {len(y_train_size)} defect pixels")

    print(f"  Fitting global Logistic Regression & Ridge Regressors...", flush=True)
    global_readouts = suite.fit_global_readouts(
        X_det=X_train_det,
        y_det=y_train_det,
        X_size=X_train_size,
        y_size=y_train_size,
    )
    print("  Global readouts successfully fitted and frozen!\n", flush=True)

    # 6. STEP 2: Evaluate All Splits
    print("=" * 70)
    print("STEP 2: Evaluating All Splits (Train, Val, Test)")
    print("=" * 70, flush=True)

    master_results: List[Dict[str, Any]] = []
    split_task_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        sp: {"anomaly": [], "sizing": [], "segmentation": []}
        for sp in args.eval_splits
    }

    for split_name in args.eval_splits:
        split_files = splits.get(split_name, [])
        print(f"\n>>> Running Evaluation on Split [{split_name.upper()}] ({len(split_files)} files) <<<", flush=True)

        pbar = tqdm(split_files, desc=f"Evaluating [{split_name.upper()}]", file=sys.stdout, dynamic_ncols=True)
        for fp in pbar:
            meta = extract_file_metadata(fp)
            specimen = meta["specimen"]
            sensor = meta["sensor"]
            waveform = meta["waveform"]
            liftoff = meta["liftoff"]
            fname = os.path.basename(fp)
            bname = os.path.splitext(fname)[0]

            if specimen not in gt_masks:
                continue

            gt_mask = gt_masks[specimen]["mask"]

            # Load or retrieve cached latents
            if split_name == "train" and fp in cached_train_latents:
                latents = cached_train_latents[fp]
            else:
                cscan = load_canonical_cscan_from_tdms(file_path=fp)
                latents = extractor.extract_full_cscan(cscan)

            # TASK 1: Anomaly Detection (Mahalanobis Distance & Global Linear Probe)
            anom_res = suite.compute_anomaly_detection_metrics(latents=latents, ground_truth_mask=gt_mask)
            det_res = suite.evaluate_global_detection(readouts=global_readouts, features=latents, labels=gt_mask)

            # TASK 2: Defect Sizing Hurdle Protocol (Corrosion continuous diameter)
            sizing_res = {"r2_score": 0.0, "mae_mm": 0.0, "rmse_mm": 0.0, "y_true": np.array([]), "y_pred": np.array([])}
            if specimen == "Corrosion" and corr_d_map is not None:
                sizing_res = suite.evaluate_global_sizing(
                    readouts=global_readouts,
                    features=latents,
                    size_map=corr_d_map,
                    focus_defects_only=True,
                )

            # TASK 3: Segmentation IoU (Using Linear Probe probability map)
            prob_map = det_res["prob_map"]
            iou_res = suite.benchmark_segmentation_iou(prob_map=prob_map, gt_mask=gt_mask, threshold=0.5)
            pred_mask = (prob_map >= 0.5).astype(np.int64)

            # Record metrics row
            row = {
                "file": fname,
                "split": split_name,
                "specimen": specimen,
                "sensor": sensor,
                "waveform": waveform,
                "liftoff": liftoff,
                "linear_probe_auc": det_res["auc_roc"],
                "linear_probe_ap": det_res["average_precision"],
                "linear_probe_f1": det_res["f1_score"],
                "linear_probe_acc": det_res["accuracy"],
                "mlp_probe_auc": det_res["mlp_auc_roc"],
                "mlp_probe_ap": det_res["mlp_average_precision"],
                "mahalanobis_auc": anom_res["auc_roc"],
                "mahalanobis_ap": anom_res["average_precision"],
                "cnr": anom_res["cnr"],
                "segmentation_iou": iou_res["iou"],
                "segmentation_dice": iou_res["dice"],
                "segmentation_precision": iou_res["precision"],
                "segmentation_recall": iou_res["recall"],
                "sizing_r2": sizing_res["r2_score"],
                "sizing_mae_mm": sizing_res["mae_mm"],
                "sizing_rmse_mm": sizing_res["rmse_mm"],
            }
            master_results.append(row)

            # Save task-specific visualizations
            # 1. Anomaly Heatmap
            anom_plot_path = os.path.join(task_dirs["anomaly"], split_name, f"{bname}_anomaly.png")
            plot_cscan_anomaly_map(
                anomaly_map=anom_res["anomaly_scores"],
                ground_truth_mask=gt_mask,
                file_title=f"[{split_name.upper()}] {specimen} | {sensor} | {waveform} | {liftoff}",
                save_path=anom_plot_path,
                metrics=anom_res,
            )
            split_task_results[split_name]["anomaly"].append({
                "file": fname,
                "specimen": specimen,
                "sensor": sensor,
                "waveform": waveform,
                "liftoff": liftoff,
                "mahalanobis_auc": anom_res["auc_roc"],
                "mahalanobis_ap": anom_res["average_precision"],
                "cnr": anom_res["cnr"],
                "linear_probe_auc": det_res["auc_roc"],
                "linear_probe_ap": det_res["average_precision"],
            })

            # 2. Sizing Scatter Plot (if applicable)
            if specimen == "Corrosion" and len(sizing_res["y_true"]) > 0:
                sizing_plot_path = os.path.join(task_dirs["sizing"], split_name, f"{bname}_sizing.png")
                plot_sizing_calibration(
                    true_size=sizing_res["y_true"],
                    pred_size=sizing_res["y_pred"],
                    save_path=sizing_plot_path,
                    title=f"[{split_name.upper()}] {sensor} | {waveform} | {liftoff}",
                    r2=sizing_res["r2_score"],
                    mae=sizing_res["mae_mm"],
                )
                split_task_results[split_name]["sizing"].append({
                    "file": fname,
                    "sensor": sensor,
                    "waveform": waveform,
                    "liftoff": liftoff,
                    "sizing_r2": sizing_res["r2_score"],
                    "sizing_mae_mm": sizing_res["mae_mm"],
                    "sizing_rmse_mm": sizing_res["rmse_mm"],
                })

            # 3. Segmentation Comparison Plot
            seg_plot_path = os.path.join(task_dirs["segmentation"], split_name, f"{bname}_segmentation.png")
            plot_segmentation_comparison(
                prob_map=prob_map,
                pred_mask=pred_mask,
                ground_truth_mask=gt_mask,
                file_title=f"[{split_name.upper()}] {specimen} | {sensor} | {waveform}",
                save_path=seg_plot_path,
                metrics=iou_res,
            )
            split_task_results[split_name]["segmentation"].append({
                "file": fname,
                "specimen": specimen,
                "sensor": sensor,
                "waveform": waveform,
                "liftoff": liftoff,
                "iou": iou_res["iou"],
                "dice": iou_res["dice"],
                "precision": iou_res["precision"],
                "recall": iou_res["recall"],
            })

            pbar.set_postfix({
                "LP_AUC": f"{det_res['auc_roc']:.4f}",
                "LP_AP": f"{det_res['average_precision']:.4f}",
                "CNR": f"{anom_res['cnr']:.2f}",
                "IoU": f"{iou_res['iou']:.4f}",
            })

    # 7. STEP 3: Save Split-Specific Metrics & Task CSVs
    print("\n" + "=" * 70)
    print("STEP 3: Saving Structured Output Hierarchy and Aggregated Reports")
    print("=" * 70, flush=True)

    for sp in args.eval_splits:
        for tname in ["anomaly", "sizing", "segmentation"]:
            trows = split_task_results[sp][tname]
            if trows:
                t_csv = os.path.join(task_dirs[tname], sp, "split_metrics.csv")
                with open(t_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
                    writer.writeheader()
                    writer.writerows(trows)

                t_json = os.path.join(task_dirs[tname], sp, "split_summary.json")
                # Compute numeric means
                num_summary = {}
                for k in trows[0].keys():
                    vals = [r[k] for r in trows if isinstance(r[k], (int, float))]
                    if vals:
                        num_summary[f"mean_{k}"] = float(np.mean(vals))
                        num_summary[f"std_{k}"] = float(np.std(vals))
                with open(t_json, "w", encoding="utf-8") as f:
                    json.dump({"split": sp, "task": tname, "count": len(trows), "metrics": num_summary}, f, indent=2)

    # 8. STEP 4: Save Master Reports
    master_csv = os.path.join(out_dir, "master_benchmark_metrics.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_results[0].keys()))
        writer.writeheader()
        writer.writerows(master_results)

    # Compute Master Hierarchical Summary
    def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        lp_auc = [r["linear_probe_auc"] for r in rows]
        lp_ap = [r["linear_probe_ap"] for r in rows]
        mahal_auc = [r["mahalanobis_auc"] for r in rows]
        cnr = [r["cnr"] for r in rows]
        iou = [r["segmentation_iou"] for r in rows]
        dice = [r["segmentation_dice"] for r in rows]
        sz_r2 = [r["sizing_r2"] for r in rows if r["specimen"] == "Corrosion"]
        sz_mae = [r["sizing_mae_mm"] for r in rows if r["specimen"] == "Corrosion"]
        return {
            "count": len(rows),
            "linear_probe_auc_mean": float(np.mean(lp_auc)),
            "linear_probe_auc_std": float(np.std(lp_auc)),
            "linear_probe_ap_mean": float(np.mean(lp_ap)),
            "linear_probe_ap_std": float(np.std(lp_ap)),
            "mahalanobis_auc_mean": float(np.mean(mahal_auc)),
            "cnr_mean": float(np.mean(cnr)),
            "segmentation_iou_mean": float(np.mean(iou)),
            "segmentation_dice_mean": float(np.mean(dice)),
            "sizing_r2_mean": float(np.mean(sz_r2)) if sz_r2 else 0.0,
            "sizing_mae_mean": float(np.mean(sz_mae)) if sz_mae else 0.0,
        }

    master_summary = {
        "evaluation_name": args.eval_name,
        "checkpoint": args.checkpoint,
        "total_files": len(master_results),
        "overall": compute_stats(master_results),
        "by_split": {
            sp: compute_stats([r for r in master_results if r["split"] == sp])
            for sp in args.eval_splits
        },
        "by_waveform": {
            wf: compute_stats([r for r in master_results if r["waveform"] == wf])
            for wf in ["Square", "Gaussian", "Chirp"]
        },
        "by_sensor": {
            sn: compute_stats([r for r in master_results if r["sensor"] == sn])
            for sn in ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]
        },
        "by_liftoff": {
            lo: compute_stats([r for r in master_results if r["liftoff"] == lo])
            for lo in ["z1", "z2", "z3"]
        },
        "by_specimen": {
            sp: compute_stats([r for r in master_results if r["specimen"] == sp])
            for sp in ["Corrosion", "Rivet_v1", "Rivet_v2"]
        },
    }

    master_json = os.path.join(out_dir, "master_benchmark_summary.json")
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # Write Markdown Report
    md_path = os.path.join(out_dir, "EVALUATION_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PECT-JEPA Official Downstream Benchmark Report\n\n")
        f.write(f"- **Checkpoint**: `{args.checkpoint}`\n")
        f.write(f"- **Evaluation Run**: `{run_tag}`\n")
        f.write(f"- **Total Files Evaluated**: {len(master_results)}\n\n")

        f.write("## 1. Split Performance Breakdown\n\n")
        f.write("| Split | Files | Linear Probe AUC | Linear Probe AP | CNR | Mahalanobis AUC | Seg IoU | Sizing R² |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for sp in args.eval_splits:
            st = master_summary["by_split"].get(sp, {})
            if st:
                f.write(f"| **{sp.upper()}** | {st['count']} | {st['linear_probe_auc_mean']:.4f} ± {st['linear_probe_auc_std']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['mahalanobis_auc_mean']:.4f} | {st['segmentation_iou_mean']:.4f} | {st['sizing_r2_mean']:.4f} |\n")

        f.write("\n## 2. Generalization Disaggregation\n\n")
        f.write("### By Waveform\n")
        f.write("| Waveform | Files | LP AUC | LP AP | CNR | Seg IoU |\n| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for wf, st in master_summary["by_waveform"].items():
            if st.get("count", 0) > 0:
                f.write(f"| {wf} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['segmentation_iou_mean']:.4f} |\n")

        f.write("\n### By Sensor\n")
        f.write("| Sensor | Files | LP AUC | LP AP | CNR | Seg IoU |\n| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for sn, st in master_summary["by_sensor"].items():
            if st.get("count", 0) > 0:
                f.write(f"| {sn} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['segmentation_iou_mean']:.4f} |\n")

        f.write("\n### By Lift-off\n")
        f.write("| Lift-off | Files | LP AUC | LP AP | CNR | Seg IoU |\n| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for lo, st in master_summary["by_liftoff"].items():
            if st.get("count", 0) > 0:
                f.write(f"| {lo} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['segmentation_iou_mean']:.4f} |\n")

    # 9. Terminal Output
    print("\n" + "=" * 70)
    print("MASTER BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    for sp in args.eval_splits:
        st = master_summary["by_split"].get(sp, {})
        if st:
            print(f"[{sp.upper():5s}] ({st['count']} files) -> LP AUC: {st['linear_probe_auc_mean']:.4f} ± {st['linear_probe_auc_std']:.4f} | LP AP: {st['linear_probe_ap_mean']:.4f} | CNR: {st['cnr_mean']:.2f} | Seg IoU: {st['segmentation_iou_mean']:.4f} | Sizing R²: {st['sizing_r2_mean']:.4f}")
    print("-" * 70)
    ov = master_summary["overall"]
    print(f"[OVERALL] ({ov['count']} files) -> LP AUC: {ov['linear_probe_auc_mean']:.4f} ± {ov['linear_probe_auc_std']:.4f} | LP AP: {ov['linear_probe_ap_mean']:.4f} | CNR: {ov['cnr_mean']:.2f} | Seg IoU: {ov['segmentation_iou_mean']:.4f}")
    print(f"\nAll results saved to: {out_dir}")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
