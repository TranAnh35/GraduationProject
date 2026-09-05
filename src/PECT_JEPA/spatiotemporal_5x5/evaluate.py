"""
CLI Evaluation and Downstream Probing Script for 5x5 Spatiotemporal PECT-JEPA.

Core Capabilities:
1. Protocol-Aware Evaluation:
   - Evaluates on held-out test files from --split_summary JSON or dynamic --split_protocol (LOLO, LOSO, LOWO, LODO).
   - Alternatively evaluates a single scan via --file.
2. 1-to-1 C-Scan Feature Map Extraction:
   - Slides a 5x5 spatial window across full scans preserving exact [300, 300, D] resolution.
3. Unsupervised Anomaly Detection & Defect Contrast:
   - Fits AnomalyDetector5x5 on sound metal baseline.
   - Computes Defect-to-Background Contrast Ratio (CNR), peak anomaly ratio, and saves 2D heatmaps.
4. Multi-Lift-Off Invariance Analysis:
   - Computes Linear CKA (Centered Kernel Alignment) and Cosine Similarity across paired lift-offs (z1 vs z2 vs z3).
5. Consolidated Reporting:
   - Saves structured evaluation_report.json with per-file and aggregate metrics.

Usage:
    # Evaluate held-out domain from split summary:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --checkpoint checkpoints/pect_jepa_5x5/best_model_5x5.pt \\
        --split_summary checkpoints/pect_jepa_5x5/pect_jepa_5x5_base_split_summary.json

    # Evaluate dynamic LOLO (z3 held out):
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --checkpoint checkpoints/pect_jepa_5x5/best_model_5x5.pt \\
        --split_protocol leave_liftoff --holdout_target z3 --eval_liftoff

    # Evaluate single TDMS file:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --checkpoint checkpoints/pect_jepa_5x5/best_model_5x5.pt \\
        --file data/Sensor_TMR/Corrosion/Square_1mm.tdms
"""

import argparse
import json
import os
import sys
import types
from typing import List, Dict, Any, Optional

# Defensive safeguard for HPC clusters where torch._dynamo has broken imports or NumPy 2.x conflicts
try:
    import torch._dynamo
except Exception:
    fake_dynamo = types.ModuleType("torch._dynamo")
    fake_dynamo.disable = lambda fn=None, *args, **kwargs: (fn if fn is not None else (lambda f: f))
    sys.modules["torch._dynamo"] = fake_dynamo

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import find_all_tdms_files
from src.PECT_JEPA.spatiotemporal_5x5.data.split import (
    get_dataset_split,
    extract_file_metadata,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import (
    extract_full_cscan_map,
    load_cscan_from_tdms,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.anomaly_detection import (
    AnomalyDetector5x5,
    plot_anomaly_heatmap_5x5,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.liftoff_invariance import (
    compute_linear_cka,
    compute_feature_similarity_matrix,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("5x5 Spatiotemporal PECT-JEPA Evaluation")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--file", type=str, default=None, help="Path to single TDMS file for defect mapping")
    p.add_argument("--data_dir", type=str, default="data", help="Data directory containing TDMS files")
    p.add_argument("--split_summary", type=str, default=None,
                   help="Path to split summary JSON produced during training (loads held-out test files)")
    p.add_argument("--split_protocol", type=str, default="compound_ood",
                   choices=["compound_ood", "leave_liftoff", "leave_sensor", "leave_waveform", "leave_specimen", "random"],
                   help="Evaluation split protocol: compound_ood (Option A), leave_liftoff (LOLO), leave_sensor (LOSO), leave_waveform (LOWO), leave_specimen (LODO), random")
    p.add_argument("--holdout_target", type=str, default="z3",
                   help="Holdout target category for single-factor protocols (e.g. 'z3', 'TMR', 'Chirp', 'Rivet_v2')")
    p.add_argument("--holdout_liftoff", type=str, default="z3",
                   help="Lift-off level held out for compound_ood (default: 'z3')")
    p.add_argument("--holdout_sensor", type=str, default="TMR",
                   help="Sensor hardware held out for compound_ood (default: 'TMR')")
    p.add_argument("--holdout_waveform", type=str, default="Chirp",
                   help="Waveform shape held out for compound_ood (default: 'Chirp')")
    p.add_argument("--output_dir", type=str, default="evaluation_results/5x5",
                   help="Directory to save evaluation artifacts and heatmaps")
    p.add_argument("--eval_liftoff", action="store_true", default=False,
                   help="Compute Linear CKA across lift-off variations (z1 vs z2 vs z3)")
    p.add_argument("--save_features", action="store_true", default=False,
                   help="Save extracted .npy feature maps to disk")
    p.add_argument("--crop_border", type=int, default=10,
                   help="Number of outer boundary pixels to crop on each edge (default: 10 to remove air/edge effect)")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for sliding window feature extraction")
    p.add_argument("--device", type=str, default="cuda", help="Target device: 'cuda' or 'cpu'")
    p.add_argument("--max_eval_files", type=int, default=None, help="Optional limit on number of test files to evaluate")
    return p


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> PECT_JEPA_5x5:
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg_dict = ckpt.get("config", {})
    if isinstance(cfg_dict, dict) and cfg_dict:
        config = Spatiotemporal5x5Config.from_dict(cfg_dict)
    else:
        config = Spatiotemporal5x5Config()

    model = PECT_JEPA_5x5(config)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    epoch_info = ckpt.get("epoch", "?")
    step_info = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint from {checkpoint_path} (epoch: {epoch_info}, step: {step_info})")
    print(f"  Model config: resample_mode={config.resample_mode}, C={config.in_channels}, embed_dim={config.embed_dim}")
    return model


def compute_anomaly_metrics(score_map: np.ndarray, top_percentile: float = 95.0) -> Dict[str, float]:
    """
    Computes quantitative defect detection metrics on the 2D anomaly score map:
    - mean_score: average anomaly score across entire scan
    - max_score: peak anomaly score
    - background_mean: mean score in normal metal zone (< top_percentile)
    - background_std: noise standard deviation in normal metal zone
    - defect_mean: mean score in defect zone (>= top_percentile)
    - contrast_ratio (CNR): (defect_mean - background_mean) / (background_std + 1e-8)
    - peak_contrast_ratio: (max_score - background_mean) / (background_std + 1e-8)
    """
    flat = score_map.flatten().astype(np.float64)
    threshold = np.percentile(flat, top_percentile)

    defect_pts = flat[flat >= threshold]
    bg_pts = flat[flat < threshold]

    bg_mean = float(np.mean(bg_pts))
    bg_std = float(np.std(bg_pts))
    defect_mean = float(np.mean(defect_pts))
    max_score = float(np.max(flat))

    cnr = (defect_mean - bg_mean) / (bg_std + 1e-8)
    p_cnr = (max_score - bg_mean) / (bg_std + 1e-8)

    return {
        "mean_score": float(np.mean(flat)),
        "max_score": max_score,
        "background_mean": bg_mean,
        "background_std": bg_std,
        "defect_mean": defect_mean,
        "contrast_ratio_cnr": float(cnr),
        "peak_contrast_ratio": float(p_cnr),
    }


def evaluate_single_file(
    file_path: str,
    model: PECT_JEPA_5x5,
    output_dir: str,
    batch_size: int = 512,
    device: str = "cuda",
    save_features: bool = False,
    crop_border: int = 10,
) -> Dict[str, Any]:
    """
    Runs full C-scan feature extraction and unsupervised anomaly detection on one TDMS file.
    """
    fname_base = os.path.splitext(os.path.basename(file_path))[0]
    meta = extract_file_metadata(file_path)

    print(f"\n--- Extracting C-Scan Features: {os.path.basename(file_path)} ---")
    grid_3d = load_cscan_from_tdms(
        file_path,
        time_samples=model.config.time_samples,
        temporal_samples=model.config.temporal_samples,
        resample_mode=model.config.resample_mode,
        normalization=model.config.normalization,
        raster_correction=model.config.raster_correction,
        crop_border=crop_border,
    )

    feature_map = extract_full_cscan_map(
        model=model,
        full_cscan_3d=grid_3d,
        batch_size=batch_size,
        device=device,
        show_pbar=True,
    )

    # Fit unsupervised anomaly detector on the scan
    detector = AnomalyDetector5x5(n_clusters=2)
    detector.fit(feature_map)
    score_map = detector.score_map(feature_map)

    # Compute quantitative defect metrics
    metrics = compute_anomaly_metrics(score_map)

    # Plot and save high-contrast heatmap
    heatmap_path = os.path.join(output_dir, f"{fname_base}_anomaly_heatmap.png")
    title = (
        f"PECT-JEPA 5x5 Anomaly Map | {meta.get('specimen', '')} - {meta.get('sensor', '')}\n"
        f"Waveform: {meta.get('waveform', '')} | Lift-off: {meta.get('liftoff', '')} | CNR: {metrics['contrast_ratio_cnr']:.2f}"
    )
    plot_anomaly_heatmap_5x5(
        anomaly_map=score_map,
        save_path=heatmap_path,
        title=title,
    )

    # Optionally save full feature map
    if save_features:
        feat_path = os.path.join(output_dir, f"{fname_base}_features_5x5.npy")
        np.save(feat_path, feature_map)

    result = {
        "file": file_path,
        "file_name": os.path.basename(file_path),
        "metadata": meta,
        "metrics": metrics,
        "heatmap_path": heatmap_path,
    }
    print(f"  [Result] CNR: {metrics['contrast_ratio_cnr']:.2f} | Peak Anomaly: {metrics['max_score']:.4f} | Heatmap: {heatmap_path}")
    return result


def evaluate_liftoff_invariance(
    file_paths: List[str],
    model: PECT_JEPA_5x5,
    output_dir: str,
    batch_size: int = 512,
    device: str = "cuda",
    crop_border: int = 10,
) -> List[Dict[str, Any]]:
    """
    Computes Linear CKA and Cosine Similarity across lift-off variations (z1 vs z2 vs z3)
    for matched (specimen, sensor, waveform) triplets.
    """
    print("\n" + "=" * 70)
    print("  RUNNING MULTI-LIFT-OFF INVARIANCE ANALYSIS (Linear CKA & Cosine Sim)")
    print("=" * 70)

    meta_by_fp = {fp: extract_file_metadata(fp) for fp in file_paths}

    # Group by (specimen, sensor, waveform)
    groups: Dict[tuple, Dict[str, str]] = {}
    for fp, m in meta_by_fp.items():
        key = (m.get("specimen"), m.get("sensor"), m.get("waveform"))
        lo = m.get("liftoff")
        if lo:
            groups.setdefault(key, {})[lo] = fp

    liftoff_results = []
    for key, lo_files in groups.items():
        liftoff_keys = sorted(lo_files.keys())
        if len(liftoff_keys) >= 2:
            specimen, sensor, waveform = key
            print(f"\nEvaluating Invariance: Specimen={specimen}, Sensor={sensor}, Waveform={waveform}")
            print(f"  Available lift-off levels: {liftoff_keys}")

            # Extract features for each lift-off level
            features_by_lo = {}
            for lo in liftoff_keys:
                fp = lo_files[lo]
                grid = load_cscan_from_tdms(
                    fp,
                    time_samples=model.config.time_samples,
                    temporal_samples=model.config.temporal_samples,
                    resample_mode=model.config.resample_mode,
                    normalization=model.config.normalization,
                    raster_correction=model.config.raster_correction,
                    crop_border=crop_border,
                )
                fmap = extract_full_cscan_map(model, grid, batch_size=batch_size, device=device, show_pbar=False)
                features_by_lo[lo] = fmap.reshape(-1, fmap.shape[-1])

            # Pairwise CKA & Cosine comparison
            for i in range(len(liftoff_keys)):
                for j in range(i + 1, len(liftoff_keys)):
                    lo_a = liftoff_keys[i]
                    lo_b = liftoff_keys[j]
                    fa = features_by_lo[lo_a]
                    fb = features_by_lo[lo_b]

                    # Subsample 10,000 spatial points for stable, fast CKA computation
                    n_sub = min(10000, fa.shape[0])
                    sub_idx = np.random.RandomState(42).choice(fa.shape[0], size=n_sub, replace=False)

                    cka = compute_linear_cka(fa[sub_idx], fb[sub_idx])
                    cos_sim = compute_feature_similarity_matrix(fa[sub_idx], fb[sub_idx])

                    print(f"  --> Pair ({lo_a} vs {lo_b}): Linear CKA = {cka:.4f} | Mean Cosine Sim = {cos_sim:.4f}")
                    liftoff_results.append({
                        "specimen": specimen,
                        "sensor": sensor,
                        "waveform": waveform,
                        "pair": f"{lo_a}_vs_{lo_b}",
                        "liftoff_a": lo_a,
                        "liftoff_b": lo_b,
                        "linear_cka": round(cka, 5),
                        "cosine_similarity": round(cos_sim, 5),
                    })

    # Save lift-off invariance metrics
    lo_json_path = os.path.join(output_dir, "liftoff_invariance_results.json")
    with open(lo_json_path, "w", encoding="utf-8") as f:
        json.dump(liftoff_results, f, indent=2)
    print(f"\nSaved lift-off invariance results to: {lo_json_path}")
    return liftoff_results



def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  5x5 SPATIOTEMPORAL PECT-JEPA EVALUATION SUITE")
    print("=" * 70)

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)

    # 1. Determine evaluation test file set
    test_files: List[str] = []
    test_slices: Dict[str, List[str]] = {}
    protocol_name: str = "custom"
    holdout_target: str = "none"

    if args.file and os.path.exists(args.file):
        test_files = [args.file]
        protocol_name = "single_file"
        holdout_target = os.path.basename(args.file)
        print(f"Single file evaluation mode: {args.file}")

    elif args.split_summary and os.path.exists(args.split_summary):
        print(f"Loading evaluation test partition from split summary: {args.split_summary}")
        with open(args.split_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        test_files = summary.get("test_files", [])
        test_slices = summary.get("test_slices", {})
        protocol_name = summary.get("protocol", "unknown")
        holdout_target = summary.get("holdout_target", "unknown")
        print(f"  Loaded {len(test_files)} held-out test files for protocol '{protocol_name}' ({holdout_target})")

    else:
        # Dynamic split protocol from data directory
        all_files = find_all_tdms_files(args.data_dir)
        if not all_files:
            print(f"[Error] No TDMS files found in {args.data_dir}")
            sys.exit(1)

        _, _, test_files, summary = get_dataset_split(
            file_paths=all_files,
            protocol=args.split_protocol,
            holdout_target=args.holdout_target,
            holdout_liftoff=args.holdout_liftoff,
            holdout_sensor=args.holdout_sensor,
            holdout_waveform=args.holdout_waveform,
            val_ratio=0.1,
            seed=42,
        )
        test_slices = summary.get("test_slices", {})
        protocol_name = args.split_protocol
        holdout_target = args.holdout_target
        print(f"Dynamically generated split protocol: {protocol_name.upper()} (Target: {holdout_target})")
        print(f"  Discovered {len(test_files)} held-out test files.")

    if not test_files:
        print("[Error] No test files selected for evaluation.")
        sys.exit(1)

    if args.max_eval_files is not None:
        test_files = test_files[:args.max_eval_files]
        print(f"  Truncated to {len(test_files)} files via --max_eval_files")

    crop_border = args.crop_border if args.crop_border is not None else getattr(model.config, "crop_border", 10)

    # 2. Evaluate each test file for anomaly detection & CNR
    file_results = []
    print(f"\n--- Evaluating {len(test_files)} Held-out Test Scans (crop_border={crop_border}) ---")
    for idx, fp in enumerate(test_files):
        print(f"[{idx + 1}/{len(test_files)}] Processing: {os.path.basename(fp)}")
        res = evaluate_single_file(
            file_path=fp,
            model=model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            save_features=args.save_features,
            crop_border=crop_border,
        )
        file_results.append(res)

    # 3. Lift-off Invariance Analysis (if explicitly requested via --eval_liftoff)
    liftoff_results = []
    if args.eval_liftoff:
        all_pool = find_all_tdms_files(args.data_dir)
        liftoff_results = evaluate_liftoff_invariance(
            file_paths=all_pool,
            model=model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            crop_border=crop_border,
        )

    # 4. Generate Comprehensive Consolidated Report
    cnrs = [r["metrics"]["contrast_ratio_cnr"] for r in file_results]
    peak_cnrs = [r["metrics"]["peak_contrast_ratio"] for r in file_results]
    ckas = [r["linear_cka"] for r in liftoff_results]
    cos_sims = [r["cosine_similarity"] for r in liftoff_results]

    # Calculate slice-based metrics for Compound OOD
    slice_metrics = {}
    if test_slices:
        for slice_name, s_files in test_slices.items():
            s_set = {os.path.normpath(f) for f in s_files}
            slice_cnrs = [
                r["metrics"]["contrast_ratio_cnr"]
                for r in file_results
                if os.path.normpath(r["file"]) in s_set
            ]
            if slice_cnrs:
                slice_metrics[slice_name] = {
                    "count": len(slice_cnrs),
                    "mean_cnr": float(np.mean(slice_cnrs)),
                    "std_cnr": float(np.std(slice_cnrs)),
                    "max_cnr": float(np.max(slice_cnrs)),
                }

    report = {
        "evaluation_protocol": protocol_name,
        "holdout_target": holdout_target,
        "checkpoint": args.checkpoint,
        "model_architecture": {
            "resample_mode": model.config.resample_mode,
            "in_channels": model.config.in_channels,
            "embed_dim": model.config.embed_dim,
            "encoder_depth": model.config.encoder_depth,
            "predictor_depth": model.config.predictor_depth,
            "normalization": model.config.normalization,
            "crop_border": crop_border,
        },
        "aggregate_metrics": {
            "total_test_files_evaluated": len(file_results),
            "mean_contrast_ratio_cnr": float(np.mean(cnrs)) if cnrs else None,
            "std_contrast_ratio_cnr": float(np.std(cnrs)) if cnrs else None,
            "max_contrast_ratio_cnr": float(np.max(cnrs)) if cnrs else None,
            "mean_peak_contrast_ratio": float(np.mean(peak_cnrs)) if peak_cnrs else None,
            "mean_liftoff_linear_cka": float(np.mean(ckas)) if ckas else None,
            "mean_liftoff_cosine_sim": float(np.mean(cos_sims)) if cos_sims else None,
        },
        "slice_metrics": slice_metrics,
        "per_file_results": file_results,
        "liftoff_invariance_results": liftoff_results,
    }

    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Protocol: {protocol_name.upper()} | Holdout Target: {holdout_target}")
    print(f"Evaluated Test Files: {len(file_results)}")
    if cnrs:
        print(f"Overall Defect-to-Background CNR:  Mean = {np.mean(cnrs):.2f} +/- {np.std(cnrs):.2f} (Max = {np.max(cnrs):.2f})")
    if slice_metrics:
        print("\n--- OOD Domain Shift Performance Breakdown ---")
        print(f"{'Domain Slice':<25} | {'Files':<6} | {'Mean CNR':<12} | {'Max CNR':<10}")
        print("-" * 60)
        for s_name, sm in slice_metrics.items():
            print(f"{s_name:<25} | {sm['count']:<6} | {sm['mean_cnr']:<12.2f} | {sm['max_cnr']:<10.2f}")
        print("-" * 60)
    if ckas:
        print(f"Lift-off Linear CKA Score: Mean = {np.mean(ckas):.4f} across {len(ckas)} pairs")
    if cos_sims:
        print(f"Lift-off Cosine Similarity: Mean = {np.mean(cos_sims):.4f}")
    print(f"\nFull evaluation report saved to: {report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
