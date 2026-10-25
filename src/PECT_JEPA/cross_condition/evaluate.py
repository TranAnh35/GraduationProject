"""
Official Downstream Evaluation Suite for Cross-Condition PECT-JEPA.

Strict Global Readout Protocol:
- Linear Probe (Defect Detection) and Sizing Regressors are trained ONLY on the Train split representations.
- Probes are frozen and evaluated across all 3 splits:
  1. Train (In-Domain Reference)
  2. Val (In-Domain Validation)
  3. Test (Zero-Shot Out-of-Distribution Transfer across unseen waveforms, sensors, liftoff, and specimens)

Unified Experiment Storage:
- Evaluation results are stored inside the SAME experiment directory as checkpoints:
  <exp_dir>/evaluation_results/
    ├── 1_Anomaly_Detection/
    │     ├── train/ (Trained Linear Probe Probability Maps, Mahalanobis Maps, Detection Summary)
    │     ├── val/
    │     └── test/
    ├── 2_Defect_Sizing/
    │     ├── train/ (Calibration scatter plots, split_metrics.csv, split_summary.json)
    │     ├── val/
    │     └── test/
    ├── 3_Segmentation_IoU/
    │     ├── train/ (3-Panel CAD GT BBoxes vs Predicted BBoxes vs Overlap TP/FP/FN/TN Maps)
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import CrossConditionJEPA
from src.PECT_JEPA.cross_condition.data.preprocessing import (
    load_canonical_cscan_from_tdms,
    get_canonical_transformer,
)
from src.PECT_JEPA.cross_condition.data.split import (
    SPLIT_MODES,
    extract_file_metadata,
    get_canonical_train_val_test_splits,
    get_canonical_shift_files,
)
from src.PECT_JEPA.cross_condition.downstream.test_time_adaptation import (
    unsup_trimmed_bg_stats,
    TestTimeSelfCalibrator,
)
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.cross_condition.evaluation.raw_signal_features import (
    extract_cscan_raw_features,
    extract_all_raw_heatmaps,
)
from src.PECT_JEPA.cross_condition.evaluation.visualizations import (
    AsyncPlotSaver,
    plot_probability_heatmap,
    plot_detection_summary_map,
    plot_raw_multi_method_heatmaps,
    plot_cscan_anomaly_map,
    plot_sizing_calibration,
    plot_defect_contours_and_iou,
    plot_uda_summary_map,
    plot_tta_summary_map,
)


def load_ground_truth_masks(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Loads CAD ground truth masks and sizing targets for all specimens."""
    masks = {}
    candidates = []
    if data_dir is not None:
        candidates.append(os.path.join(data_dir, "ground_truth"))
        candidates.append(data_dir)
    candidates.extend([
        os.path.join(ROOT_DIR, "data", "ground_truth"),
        os.path.join(ROOT_DIR, "ground_truth"),
        "data/ground_truth",
        "ground_truth",
        "/content/drive/MyDrive/PECT_Dataset/ground_truth",
    ])
    gt_root = None
    for c in candidates:
        if os.path.exists(os.path.join(c, "corrosion", "corrosion_gt_mask.npy")):
            gt_root = c
            break

    if gt_root is None:
        return masks

    # 1. Corrosion
    corr_gt_path = os.path.join(gt_root, "corrosion", "corrosion_gt_mask.npy")
    corr_cfg_path = os.path.join(gt_root, "corrosion", "corrosion_flaws_config.json")
    if os.path.exists(corr_gt_path):
        masks["Corrosion"] = {
            "mask": np.load(corr_gt_path),
            "config": json.load(open(corr_cfg_path, "r", encoding="utf-8")) if os.path.exists(corr_cfg_path) else None,
        }

    # 2. Rivet_v1
    r1_gt_path = os.path.join(gt_root, "rivet_v1", "rivet_v1_gt_mask.npy")
    r1_cfg_path = os.path.join(gt_root, "rivet_v1", "rivet_v1_flaws_config.json")
    if os.path.exists(r1_gt_path):
        masks["Rivet_v1"] = {
            "mask": np.load(r1_gt_path),
            "config": json.load(open(r1_cfg_path, "r", encoding="utf-8")) if os.path.exists(r1_cfg_path) else None,
        }

    # 3. Rivet_v2
    r2_gt_path = os.path.join(gt_root, "rivet_v2", "rivet_v2_gt_mask.npy")
    r2_cfg_path = os.path.join(gt_root, "rivet_v2", "rivet_v2_flaws_config.json")
    if os.path.exists(r2_gt_path):
        masks["Rivet_v2"] = {
            "mask": np.load(r2_gt_path),
            "config": json.load(open(r2_cfg_path, "r", encoding="utf-8")) if os.path.exists(r2_cfg_path) else None,
        }

    return masks


def build_corrosion_diameter_map(config: Dict[str, Any], grid_shape: Tuple[int, int] = (270, 270)) -> np.ndarray:
    """Creates [270, 270] continuous diameter map in mm from flaw config."""
    d_map = np.zeros(grid_shape, dtype=np.float32)
    features = config.get("features", [])
    if not features:
        return d_map
    # Vectorized disk rasterization (identical values to the old per-pixel
    # Python loops, without the O(flaws x area) interpreter overhead)
    rr, cc = np.mgrid[0:grid_shape[0], 0:grid_shape[1]]
    for feat in features:
        cx = int(round(feat["x"]))
        cy = int(round(feat["y"]))
        dia = float(feat.get("diameter", 0.0))
        rad = int(round(dia / 2.0))
        r0, r1 = max(0, cy - rad), min(grid_shape[0], cy + rad + 1)
        c0, c1 = max(0, cx - rad), min(grid_shape[1], cx + rad + 1)
        disk = (rr[r0:r1, c0:c1] - cy) ** 2 + (cc[r0:r1, c0:c1] - cx) ** 2 <= rad ** 2
        d_map[r0:r1, c0:c1][disk] = dia
    return d_map


def parse_args():
    parser = argparse.ArgumentParser(description="Official Downstream Benchmark Evaluation for Cross-Condition PECT-JEPA")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--exp_dir", type=str, default=None, help="Unified experiment run directory containing checkpoints/ and configs")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing TDMS files")
    parser.add_argument("--out_dir", type=str, default=None, help="Explicit override for output directory")
    parser.add_argument("--eval_splits", nargs="+", default=["train", "val", "test"], help="Splits to evaluate ('train', 'val', 'test')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for feature extraction")
    parser.add_argument("--split_mode", type=str, default="cross_specimen", choices=list(SPLIT_MODES), help="Split protocol: 'cross_specimen' (16-file Base Domain) or 'cross_condition' (24-file Base Domain incl. Rivet_v2)")
    parser.add_argument("--sizing_log_target", type=lambda v: v.lower() == "true", default=True, help="Fit fallback sizing readouts on log1p(diameter) and invert at inference")
    parser.add_argument("--prefetch", type=lambda v: v.lower() == "true", default=True, help="Overlap next-file TDMS load with current-file GPU/CPU processing")
    parser.add_argument("--self_calibration", type=lambda v: v.lower() == "true", default=True, help="Enable unsupervised spatial self-calibration (zero-centering C-scan against sound-metal background median)")
    parser.add_argument("--downstream_dir", type=str, default=None, help="Directory containing pre-trained downstream models (standard_probes.joblib)")
    parser.add_argument("--eval_tta", "--run_tta", dest="eval_tta", type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=True, help="Whether to evaluate canonical Test-Time Adaptation (TTA). Default True.")
    parser.add_argument("--skip_tta", dest="skip_tta", nargs="?", const=True, type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=False, help="Skip canonical Test-Time Adaptation (TTA)")
    parser.add_argument("--tta_trim_ratio", type=float, default=0.75, help="Trim ratio for background statistics in TTA (default 0.75)")
    parser.add_argument("--tta_shrinkage", type=float, default=1e-2, help="Diagonal loading shrinkage eps for TTA whitening matrix (default 1e-2)")
    # Backward compatibility flags (aliased to TTA control)
    parser.add_argument("--skip_domain_adaptation", "--skip_uda", dest="skip_domain_adaptation", nargs="?", const=True, type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=False, help="[Deprecated alias for --skip_tta]")
    parser.add_argument("--eval_domain_adaptation", "--run_uda", dest="eval_domain_adaptation", type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=True, help="[Deprecated alias for --eval_tta]")
    return parser.parse_args()


def resolve_paths(args) -> Tuple[str, str]:
    """Resolves checkpoint path and evaluation output directory so they reside in the same experiment folder."""
    # 1. If exp_dir is explicitly given
    if args.exp_dir is not None:
        exp_dir = os.path.abspath(args.exp_dir)
        ckpt_candidates = [
            os.path.join(exp_dir, "checkpoints", "best_cross_condition_jepa.pt"),
            os.path.join(exp_dir, "checkpoints", "best_model.pt"),
            os.path.join(exp_dir, "checkpoints", "latest_cross_condition_jepa.pt"),
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
        if ckpt_path is None:
            # Look for any .pt inside checkpoints/
            pts = glob.glob(os.path.join(exp_dir, "checkpoints", "*.pt"))
            if pts:
                ckpt_path = pts[0]
            else:
                raise FileNotFoundError(f"No checkpoint found inside: {os.path.join(exp_dir, 'checkpoints')}")

        eval_out_dir = os.path.join(exp_dir, "evaluation_results") if args.out_dir is None else args.out_dir
        return ckpt_path, eval_out_dir

    # 2. If checkpoint is given
    if args.checkpoint is not None:
        ckpt_path = os.path.abspath(args.checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        if args.out_dir is not None:
            eval_out_dir = args.out_dir
        else:
            # Check if checkpoint is inside a run directory structure (e.g. .../<run_dir>/checkpoints/<model>.pt)
            parent = os.path.dirname(ckpt_path)
            if os.path.basename(parent) == "checkpoints":
                exp_dir = os.path.dirname(parent)
                eval_out_dir = os.path.join(exp_dir, "evaluation_results")
            else:
                eval_out_dir = os.path.join(os.path.dirname(ckpt_path), "evaluation_results")

        return ckpt_path, eval_out_dir

    raise ValueError("Must provide either --exp_dir or --checkpoint")


def scan_feature_energy(latents: np.ndarray) -> float:
    """Mean ||z||^2 over pixels: latent-regime fingerprint of a scan.

    Fix 5 (audit finding): in-domain vs OOD scans live in different energy
    regimes (scattering E ~6500 vs ~2200; carrier ~11 vs ~95). A scan whose
    regime mismatches training is expected to break fixed readouts.
    """
    return float(np.mean(np.sum(latents.astype(np.float64) ** 2, axis=-1)))


def regime_flag(energy: float, ref_median: float, band: float = 2.0) -> int:
    """1 when energy lies outside [ref/band, ref*band]: different latent
    regime than the training pool -> flag as OOD-regime (expect failure)."""
    if not ref_median or ref_median <= 0 or not energy or energy <= 0:
        return 0
    return int(energy < ref_median / band or energy > ref_median * band)


def _iter_with_prefetch(files: List[str], use_prefetch: bool, max_workers: int = 4, lookahead: int = 4):
    """Yields (file_path, canonical_cscan) in order, decoding up to `lookahead`
    TDMS files concurrently on background threads while the caller processes
    the current file. Processing order, RNG consumption, and results are identical to the sequential loop;
    use_prefetch=False reproduces the sequential loop exactly.
    """
    if not use_prefetch or len(files) <= 1:
        for fp in files:
            yield fp, load_canonical_cscan_from_tdms(file_path=fp)
        return
    from collections import deque
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending: deque = deque()
        nxt = 0

        def _pump():
            nonlocal nxt
            while nxt < len(files) and len(pending) < lookahead:
                pending.append((files[nxt], pool.submit(load_canonical_cscan_from_tdms, files[nxt])))
                nxt += 1

        _pump()
        for fp in files:
            wfp, fut = pending.popleft()
            assert wfp == fp, "Prefetch stream out of order"
            cscan = fut.result()
            _pump()
            yield fp, cscan


def evaluate_canonical_test_time_adaptation(
    downstream_dir: Optional[str],
    out_dir: str,
    all_files: List[str],
    gt_masks: Dict[str, Any],
    cached_latents: Dict[str, np.ndarray],
    extractor: CScanFeatureExtractor,
    self_calibration: bool,
    global_readouts: Dict[str, Any],
    device: torch.device,
    tta_trim_ratio: float = 0.75,
    tta_shrinkage: float = 1e-2,
    split_mode: str = "cross_specimen",
    cached_raw_heatmaps: Optional[Dict[str, np.ndarray]] = None,
    async_saver: Optional[AsyncPlotSaver] = None,
) -> Dict[str, Any]:
    """
    Evaluates Test-Time Adaptation (TTA) Suite across the canonical single-variable shifts:
    - cross_sensor:   Source (Base Domain) -> Target (TMR-only files)
    - cross_waveform: Source (Base Domain) -> Target (Chirp-only files)
    - cross_specimen: Source (16 Base files) -> Target (8 Rivet_v2-only files);
                      only defined under split_mode='cross_specimen' (Rivet_v2 is
                      part of the source domain under 'cross_condition').
    - cross_liftoff:  Source (Base Domain) -> Target (z2-only files)

    Structure:
      4. Test Time Adaptation/<shift>/visualizations/
      4. Test Time Adaptation/<shift>/tta_summary.csv
      4. Test Time Adaptation/<shift>/tta_summary.json
    """
    canonical_shifts = ["cross_sensor", "cross_waveform", "cross_specimen", "cross_liftoff"]
    shift_results: Dict[str, Any] = {}

    print("\n" + "=" * 75)
    print("STEP 4: Canonical Test-Time Adaptation (TTA) Benchmark (Single-Variable Controlled Shifts)")
    print("=" * 75, flush=True)

    raw_scaler = global_readouts.get("raw_scaler_det", global_readouts.get("scaler_det"))
    raw_lr = global_readouts.get("raw_lr_probe", global_readouts.get("lr_probe"))
    raw_mlp = global_readouts.get("raw_mlp_probe", global_readouts.get("mlp_probe"))

    tta_scaler = global_readouts.get("tta_scaler_det", raw_scaler)
    tta_lr = global_readouts.get("tta_lr_probe", raw_lr)
    tta_mlp = global_readouts.get("tta_mlp_probe", raw_mlp)

    calibrator = TestTimeSelfCalibrator(keep_q=tta_trim_ratio, eps=tta_shrinkage)

    for shift in canonical_shifts:
        if shift == "cross_specimen" and split_mode == "cross_condition":
            print(f"\n[SKIP] shift '{shift}': Rivet_v2 belongs to the Base Domain under split_mode='cross_condition'.", flush=True)
            continue
        src_files, tgt_files = get_canonical_shift_files(all_files, shift, split_mode=split_mode)
        shift_out_dir = os.path.join(out_dir, "4. Test Time Adaptation", shift)
        os.makedirs(shift_out_dir, exist_ok=True)

        print("\n" + "-" * 75)
        print(f"Evaluating Canonical Shift [{shift.upper()}]: Test-Time Adaptation (TTA)")
        print(f"  Source Domain: {len(src_files)} Base files | Target Domain: {len(tgt_files)} Single-Shift files")
        print("-" * 75, flush=True)

        shift_rows: List[Dict[str, Any]] = []

        for set_name, f_list in [("source", src_files), ("target", tgt_files)]:
            for fp in f_list:
                fname = os.path.basename(fp)
                bname = os.path.splitext(fname)[0]
                meta = extract_file_metadata(fp)
                sp = meta["specimen"]
                if sp not in gt_masks:
                    continue
                gt_mask = gt_masks[sp]["mask"]
                valid = np.where(gt_mask.reshape(-1) >= 0)[0]
                has_both = len(np.unique(gt_mask.reshape(-1)[valid])) > 1

                # Retrieve or extract latents and raw peak-to-peak heatmap
                if fp in cached_latents:
                    latents = cached_latents[fp]
                else:
                    cscan = load_canonical_cscan_from_tdms(fp)
                    latents = extractor.extract_full_cscan(cscan, self_calibrate=self_calibration)
                    cached_latents[fp] = latents
                    if cached_raw_heatmaps is not None and fp not in cached_raw_heatmaps:
                        cached_raw_heatmaps[fp] = np.ptp(cscan, axis=-1).astype(np.float32)

                H, W = latents.shape[0], latents.shape[1]
                flat_feats = latents.reshape(-1, latents.shape[-1]).astype(np.float32)

                # 1. Zero-Shot Raw Probes Baseline
                lp_raw_prob = None
                auc_raw_lp, ap_raw_lp = 0.0, 0.0
                if raw_lr is not None and raw_scaler is not None:
                    flat_raw_scaled = raw_scaler.transform(flat_feats)
                    lp_raw_prob = raw_lr.predict_proba(flat_raw_scaled)[:, 1]
                    if has_both:
                        auc_raw_lp = float(roc_auc_score(gt_mask.reshape(-1)[valid], lp_raw_prob[valid]))
                        ap_raw_lp = float(average_precision_score(gt_mask.reshape(-1)[valid], lp_raw_prob[valid]))
                lp_raw_map = lp_raw_prob.reshape(H, W) if lp_raw_prob is not None else np.zeros((H, W), dtype=np.float32)

                auc_raw_mlp, ap_raw_mlp = 0.0, 0.0
                if raw_mlp is not None and raw_scaler is not None:
                    mlp_raw_prob = raw_mlp.predict_proba(flat_raw_scaled)[:, 1]
                    if has_both:
                        auc_raw_mlp = float(roc_auc_score(gt_mask.reshape(-1)[valid], mlp_raw_prob[valid]))
                        ap_raw_mlp = float(average_precision_score(gt_mask.reshape(-1)[valid], mlp_raw_prob[valid]))

                # 2. TTA Self-Calibration Whitening
                mu_bg, W_bg = unsup_trimmed_bg_stats(flat_feats, keep_q=tta_trim_ratio, eps=tta_shrinkage)
                flat_feats_tta = (flat_feats - mu_bg).dot(W_bg).astype(np.float32)

                # 3. TTA Calibrated Linear Probe
                lp_tta_prob = None
                auc_tta_lp, ap_tta_lp = 0.0, 0.0
                if tta_lr is not None and tta_scaler is not None:
                    flat_tta_scaled = tta_scaler.transform(flat_feats_tta)
                    lp_tta_prob = tta_lr.predict_proba(flat_tta_scaled)[:, 1]
                    if has_both:
                        auc_tta_lp = float(roc_auc_score(gt_mask.reshape(-1)[valid], lp_tta_prob[valid]))
                        ap_tta_lp = float(average_precision_score(gt_mask.reshape(-1)[valid], lp_tta_prob[valid]))
                lp_tta_map = lp_tta_prob.reshape(H, W) if lp_tta_prob is not None else np.zeros((H, W), dtype=np.float32)

                # 4. TTA Calibrated MLP
                auc_tta_mlp, ap_tta_mlp = 0.0, 0.0
                if tta_mlp is not None and tta_scaler is not None:
                    mlp_tta_prob = tta_mlp.predict_proba(flat_tta_scaled)[:, 1]
                    if has_both:
                        auc_tta_mlp = float(roc_auc_score(gt_mask.reshape(-1)[valid], mlp_tta_prob[valid]))
                        ap_tta_mlp = float(average_precision_score(gt_mask.reshape(-1)[valid], mlp_tta_prob[valid]))

                # 5. TTA Mahalanobis Distance (Unsupervised)
                maha_scores = calibrator.compute_anomaly_scores(latents)
                flat_maha = maha_scores.reshape(-1)
                auc_maha, ap_maha = 0.0, 0.0
                if has_both:
                    auc_maha = float(roc_auc_score(gt_mask.reshape(-1)[valid], flat_maha[valid]))
                    ap_maha = float(average_precision_score(gt_mask.reshape(-1)[valid], flat_maha[valid]))

                # Save Visualizations for both source and target files in their respective folders
                set_dir = os.path.join(shift_out_dir, set_name)
                os.makedirs(set_dir, exist_ok=True)
                comp_path = os.path.join(set_dir, f"{bname}_tta_comparison.png")
                raw_ptp = cached_raw_heatmaps.get(fp) if cached_raw_heatmaps is not None else None
                if raw_ptp is None:
                    cscan_tmp = load_canonical_cscan_from_tdms(fp)
                    raw_ptp = extract_cscan_raw_features(cscan_tmp, method="filtered_ptp")
                    if cached_raw_heatmaps is not None:
                        cached_raw_heatmaps[fp] = raw_ptp

                target_cnr = 0.0
                if has_both:
                    s_idx = np.where(gt_mask.reshape(-1) == 0)[0]
                    d_idx = np.where(gt_mask.reshape(-1) == 1)[0]
                    if len(s_idx) > 0 and len(d_idx) > 0:
                        target_cnr = float((np.mean(flat_maha[d_idx]) - np.mean(flat_maha[s_idx])) / (np.std(flat_maha[s_idx]) + 1e-6))

                if async_saver is not None:
                    async_saver.submit(
                        plot_tta_summary_map,
                        ground_truth_mask=gt_mask,
                        lp_raw_prob_map=lp_raw_map,
                        lp_tta_prob_map=lp_tta_map,
                        file_title=f"[{shift.upper()} - {set_name.upper()}] {sp} | {meta['sensor']} | {meta['waveform']}",
                        save_path=comp_path,
                        raw_heatmap=raw_ptp,
                        maha_map=maha_scores,
                        lp_raw_metrics={"auc": auc_raw_lp, "ap": ap_raw_lp},
                        lp_tta_metrics={"auc": auc_tta_lp, "ap": ap_tta_lp},
                        maha_metrics={"auc": auc_maha, "ap": ap_maha, "cnr": target_cnr},
                    )
                else:
                    plot_tta_summary_map(
                        ground_truth_mask=gt_mask,
                        lp_raw_prob_map=lp_raw_map,
                        lp_tta_prob_map=lp_tta_map,
                        file_title=f"[{shift.upper()} - {set_name.upper()}] {sp} | {meta['sensor']} | {meta['waveform']}",
                        save_path=comp_path,
                        raw_heatmap=raw_ptp,
                        maha_map=maha_scores,
                        lp_raw_metrics={"auc": auc_raw_lp, "ap": ap_raw_lp},
                        lp_tta_metrics={"auc": auc_tta_lp, "ap": ap_tta_lp},
                        maha_metrics={"auc": auc_maha, "ap": ap_maha, "cnr": target_cnr},
                    )

                if not has_both:
                    print(f"    [skip] {set_name} {fname}: GT mask has a single class")
                    continue

                shift_rows.append({
                    "shift": shift,
                    "domain_set": set_name,
                    "file": fname,
                    "specimen": sp,
                    "sensor": meta["sensor"],
                    "waveform": meta["waveform"],
                    "liftoff": meta["liftoff"],
                    "auc_raw_lp": round(auc_raw_lp, 4),
                    "ap_raw_lp": round(ap_raw_lp, 4),
                    "auc_raw_mlp": round(auc_raw_mlp, 4),
                    "ap_raw_mlp": round(ap_raw_mlp, 4),
                    "auc_tta_lp": round(auc_tta_lp, 4),
                    "ap_tta_lp": round(ap_tta_lp, 4),
                    "delta_tta_lp": round(auc_tta_lp - auc_raw_lp, 4),
                    "auc_tta_mlp": round(auc_tta_mlp, 4),
                    "ap_tta_mlp": round(ap_tta_mlp, 4),
                    "delta_tta_mlp": round(auc_tta_mlp - auc_raw_lp, 4),
                    "auc_maha": round(auc_maha, 4),
                    "ap_maha": round(ap_maha, 4),
                    "delta_maha": round(auc_maha - auc_raw_lp, 4),
                })

        # Save Shift CSV & Summary
        if shift_rows:
            csv_path = os.path.join(shift_out_dir, "tta_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
                writer.writeheader()
                writer.writerows(shift_rows)

            # Save per-domain metrics.csv (source / target)
            for s_name in ["source", "target"]:
                s_rows = [r for r in shift_rows if r["domain_set"] == s_name]
                if s_rows:
                    s_dir = os.path.join(shift_out_dir, s_name)
                    os.makedirs(s_dir, exist_ok=True)
                    s_csv = os.path.join(s_dir, "metrics.csv")
                    with open(s_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(s_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(s_rows)

            shift_summary: Dict[str, Any] = {"shift": shift, "methods": {}}
            method_keys = [
                ("Zero-Shot LP (Raw)", "auc_raw_lp", "ap_raw_lp", None),
                ("Zero-Shot MLP (Raw)", "auc_raw_mlp", "ap_raw_mlp", None),
                ("TTA Calibrated LP", "auc_tta_lp", "ap_tta_lp", "delta_tta_lp"),
                ("TTA Calibrated MLP", "auc_tta_mlp", "ap_tta_mlp", "delta_tta_mlp"),
                ("TTA Mahalanobis (Unsupervised)", "auc_maha", "ap_maha", "delta_maha"),
            ]

            for m_label, auc_k, ap_k, delta_k in method_keys:
                shift_summary["methods"][m_label] = {}
                for s_name in ["source", "target"]:
                    rows = [r for r in shift_rows if r["domain_set"] == s_name]
                    aucs = [r[auc_k] for r in rows]
                    aps = [r[ap_k] for r in rows]
                    deltas = [r[delta_k] for r in rows] if delta_k is not None else [0.0] * len(rows)
                    shift_summary["methods"][m_label][s_name] = {
                        "count": len(rows),
                        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
                        "auc_std": float(np.std(aucs)) if aucs else 0.0,
                        "ap_mean": float(np.mean(aps)) if aps else 0.0,
                        "delta_auc_mean": float(np.mean(deltas)) if deltas else 0.0,
                    }

            json_path = os.path.join(shift_out_dir, "tta_summary.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(shift_summary, f, indent=2)

            # Print Console Summary Table
            print("\n" + "=" * 92)
            print(f"CANONICAL TEST-TIME ADAPTATION (TTA) SUMMARY: [{shift.upper()}]")
            print(f"  Source Domain: {len(src_files)} files (Train/Val Pool) | Target Domain: {len(tgt_files)} files (Single Shift)")
            print("=" * 92)
            print(f"{'Method / Mechanism':<30} | {'Source AUC (Retain)':<20} | {'Target AUC (Adapt)':<21} | {'Target AP':<9} | {'Delta vs Base':<13}")
            print("-" * 92)
            for m_label, auc_k, ap_k, delta_k in method_keys:
                s_stat = shift_summary["methods"][m_label]["source"]
                t_stat = shift_summary["methods"][m_label]["target"]
                s_auc = s_stat["auc_mean"]
                t_auc = t_stat["auc_mean"]
                t_std = t_stat["auc_std"]
                t_ap = t_stat["ap_mean"]
                delta = t_stat["delta_auc_mean"]
                delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}" if delta_k is not None else "baseline"
                print(f"{m_label:<30} | {s_auc:<20.4f} | {t_auc:.4f} +/- {t_std:<10.4f} | {t_ap:<9.4f} | {delta_str:<13}")
            print("=" * 92 + "\n", flush=True)

            shift_results[shift] = shift_summary

    return shift_results


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args()
    device = torch.device(args.device)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Resolve paths to unify checkpoints and evaluations in the same folder
    ckpt_path, out_dir = resolve_paths(args)
    os.makedirs(out_dir, exist_ok=True)
    async_saver = AsyncPlotSaver(max_workers=4)

    task_dirs = {
        "anomaly": os.path.join(out_dir, "1_Anomaly_Detection"),
        "sizing": os.path.join(out_dir, "2_Defect_Sizing"),
        "segmentation": os.path.join(out_dir, "3_Segmentation_IoU"),
    }
    for tdir in task_dirs.values():
        os.makedirs(tdir, exist_ok=True)

    print("\n" + "=" * 75)
    print("PECT-JEPA Unified Downstream Benchmark Evaluation")
    print(f"Checkpoint:       {ckpt_path}")
    print(f"Output Directory: {out_dir}")
    print(f"Compute Device:   {device}")
    print(f"Target Splits:    {args.eval_splits}")
    print("=" * 75 + "\n", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config", None)
    if isinstance(raw_config, CrossConditionConfig):
        config = raw_config
    else:
        config = CrossConditionConfig()
        if raw_config is not None:
            source_dict = vars(raw_config) if hasattr(raw_config, "__dict__") else (raw_config if isinstance(raw_config, dict) else {})
            for k, v in source_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
    model = CrossConditionJEPA(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Ground Truth Masks & Feature Extractor
    extractor = CScanFeatureExtractor(model=model, device=device, batch_size=args.batch_size)
    suite = DownstreamBenchmarkSuite(n_splits=5, random_state=42, device=device)
    gt_masks = load_ground_truth_masks(args.data_dir)
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])

    # 3. Discover All TDMS Files & Canonical Partitioning
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]

    splits = get_canonical_train_val_test_splits(all_files, split_mode=args.split_mode, seed=42)
    # Pre-build the canonical transformer on the main thread so background
    # prefetch workers never race on the lazy global.
    get_canonical_transformer()
    print(f"Discovered {len(all_files)} total inspection files:")
    print(f"  - Train Split (Base Domain):     {len(splits['train'])} files")
    print(f"  - Val Split (Base Domain):       {len(splits['val'])} files")
    print(f"  - Test Split (Held-out OOD):     {len(splits['test'])} files\n", flush=True)

    # 4. STEP 1: Load Pre-trained Downstream Models or Fit on Train Pool
    print("=" * 75)
    print("STEP 1: Downstream Probes & TTA Models Setup")
    print("=" * 75, flush=True)

    downstream_dir = args.downstream_dir
    parent_out = os.path.dirname(out_dir) if os.path.basename(out_dir) == "evaluation_results" else out_dir
    parent_ckpt = os.path.dirname(ckpt_path)
    grandparent_ckpt = os.path.dirname(parent_ckpt) if os.path.basename(parent_ckpt) == "checkpoints" else parent_ckpt

    candidates = [
        os.path.join(parent_out, "downstream_models"),
        os.path.join(grandparent_ckpt, "downstream_models"),
        os.path.join(parent_ckpt, "downstream_models"),
    ]
    if downstream_dir is None:
        for cand in candidates:
            if os.path.exists(cand) and os.path.exists(os.path.join(cand, "standard_probes.joblib")):
                downstream_dir = cand
                break

    global_readouts = None
    cached_all_latents: Dict[str, np.ndarray] = {}
    cached_raw_heatmaps: Dict[str, np.ndarray] = {}
    train_energies: List[float] = []

    if downstream_dir is not None and os.path.exists(os.path.join(downstream_dir, "standard_probes.joblib")):
        print(f"  [INFO] Loading pre-trained downstream models from: {downstream_dir}")
        bundle = joblib.load(os.path.join(downstream_dir, "standard_probes.joblib"))
        raw_det = bundle.get("raw_detection", bundle.get("detection"))
        raw_size = bundle.get("raw_sizing", bundle.get("sizing"))
        tta_det = bundle.get("tta_detection", None)
        tta_size = bundle.get("tta_sizing", None)

        global_readouts = {
            "scaler_det": raw_det.scaler if raw_det else None,
            "lr_probe": raw_det.linear_probe if raw_det else None,
            "mlp_probe": raw_det.mlp_probe if raw_det else None,
            "raw_scaler_det": raw_det.scaler if raw_det else None,
            "raw_lr_probe": raw_det.linear_probe if raw_det else None,
            "raw_mlp_probe": raw_det.mlp_probe if raw_det else None,
            "tta_scaler_det": tta_det.scaler if tta_det else (raw_det.scaler if raw_det else None),
            "tta_lr_probe": tta_det.linear_probe if tta_det else (raw_det.linear_probe if raw_det else None),
            "tta_mlp_probe": tta_det.mlp_probe if tta_det else (raw_det.mlp_probe if raw_det else None),
        }
        if raw_size is not None:
            global_readouts["scaler_size"] = raw_size.scaler
            global_readouts["ridge_size"] = raw_size.ridge_regressor
            global_readouts["mlp_size"] = raw_size.mlp_regressor
            global_readouts["raw_scaler_size"] = raw_size.scaler
            global_readouts["raw_ridge_size"] = raw_size.ridge_regressor
            global_readouts["raw_mlp_size"] = raw_size.mlp_regressor
        if tta_size is not None:
            global_readouts["tta_scaler_size"] = tta_size.scaler
            global_readouts["tta_ridge_size"] = tta_size.ridge_regressor
            global_readouts["tta_mlp_size"] = tta_size.mlp_regressor

        ref_energy = 0.0
        print("  Pre-trained standard downstream models loaded successfully!\n", flush=True)
    else:
        print("\n" + "!" * 75)
        print("  [WARNING] No pre-trained downstream models found!")
        print("  Checked candidate paths:")
        for cand in candidates:
            print(f"    - {cand} (exists: {os.path.exists(cand)})")
        print("  Fitting readouts on train pool as fallback (Dual Raw + TTA Calibrated Probes).")
        print("  To pre-train dedicated probes, run train_downstream.py first:")
        print(f"    python -m src.PECT_JEPA.cross_condition.downstream.train_downstream --checkpoint {ckpt_path} --data_dir {args.data_dir}")
        print("!" * 75 + "\n", flush=True)
        train_det_x_raw, train_det_x_tta, train_det_y = [], [], []
        train_size_x_raw, train_size_x_tta, train_size_y = [], [], []

        train_eval_files = [fp for fp in splits["train"]
                            if extract_file_metadata(fp)["specimen"] in gt_masks]
        pbar_train = tqdm(_iter_with_prefetch(train_eval_files, args.prefetch, max_workers=4, lookahead=4),
                          total=len(train_eval_files),
                          desc="Fitting Readouts on Train Pool", file=sys.stdout, dynamic_ncols=True)
        rng = np.random.RandomState(42)

        for fp, cscan in pbar_train:
            meta = extract_file_metadata(fp)
            sp = meta["specimen"]

            gt_mask = gt_masks[sp]["mask"]
            latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
            cached_all_latents[fp] = latents
            cached_raw_heatmaps[fp] = extract_cscan_raw_features(cscan, method="filtered_ptp")
            train_energies.append(scan_feature_energy(latents))

            flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            flat_y = gt_mask.reshape(-1).astype(np.int64)

            # TTA Whitened Features
            mu_bg, W_bg = unsup_trimmed_bg_stats(flat_z, keep_q=args.tta_trim_ratio, eps=args.tta_shrinkage)
            flat_z_tta = (flat_z - mu_bg).dot(W_bg).astype(np.float32)

            pos_idx = np.where(flat_y == 1)[0]
            neg_idx = np.where(flat_y == 0)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                sub_neg_count = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
                sub_neg = rng.choice(neg_idx, size=sub_neg_count, replace=False)
                sel = np.concatenate([pos_idx, sub_neg])
                train_det_x_raw.append(flat_z[sel])
                train_det_x_tta.append(flat_z_tta[sel])
                train_det_y.append(flat_y[sel])

            if sp == "Corrosion" and corr_d_map is not None:
                flat_d = corr_d_map.reshape(-1).astype(np.float32)
                def_idx = np.where(flat_d > 0.0)[0]
                if len(def_idx) > 0:
                    train_size_x_raw.append(flat_z[def_idx])
                    train_size_x_tta.append(flat_z_tta[def_idx])
                    train_size_y.append(flat_d[def_idx])

        X_train_det_raw = np.concatenate(train_det_x_raw, axis=0)
        X_train_det_tta = np.concatenate(train_det_x_tta, axis=0)
        y_train_det = np.concatenate(train_det_y, axis=0)
        X_train_size_raw = np.concatenate(train_size_x_raw, axis=0) if train_size_x_raw else None
        X_train_size_tta = np.concatenate(train_size_x_tta, axis=0) if train_size_x_tta else None
        y_train_size = np.concatenate(train_size_y, axis=0) if train_size_y else None

        print(f"\n  [Train Pool Aggregation]")
        print(f"  - Defect Detection samples: {len(y_train_det)} (Defects: {np.sum(y_train_det == 1)}, Sound: {np.sum(y_train_det == 0)})")
        if y_train_size is not None:
            print(f"  - Flaw Sizing samples:      {len(y_train_size)} defect pixels")
        ref_energy = float(np.median(train_energies)) if train_energies else 0.0
        print(f"  - Latent regime reference (median scan E||z||^2): {ref_energy:.1f}", flush=True)

        print(f"  Fitting Dual Readouts (Raw & TTA Calibrated)...", flush=True)
        raw_readouts = suite.fit_global_readouts(
            X_det=X_train_det_raw,
            y_det=y_train_det,
            X_size=X_train_size_raw,
            y_size=y_train_size,
            size_log_target=args.sizing_log_target,
        )
        tta_readouts = suite.fit_global_readouts(
            X_det=X_train_det_tta,
            y_det=y_train_det,
            X_size=X_train_size_tta,
            y_size=y_train_size,
            size_log_target=args.sizing_log_target,
        )
        global_readouts = {
            "scaler_det": raw_readouts["scaler_det"],
            "lr_probe": raw_readouts["lr_probe"],
            "mlp_probe": raw_readouts.get("mlp_probe"),
            "raw_scaler_det": raw_readouts["scaler_det"],
            "raw_lr_probe": raw_readouts["lr_probe"],
            "raw_mlp_probe": raw_readouts.get("mlp_probe"),
            "scaler_size": raw_readouts.get("scaler_size"),
            "ridge_size": raw_readouts.get("ridge_size"),
            "mlp_size": raw_readouts.get("mlp_size"),
            "raw_scaler_size": raw_readouts.get("scaler_size"),
            "raw_ridge_size": raw_readouts.get("ridge_size"),
            "raw_mlp_size": raw_readouts.get("mlp_size"),
            "tta_scaler_det": tta_readouts["scaler_det"],
            "tta_lr_probe": tta_readouts["lr_probe"],
            "tta_mlp_probe": tta_readouts.get("mlp_probe"),
            "tta_scaler_size": tta_readouts.get("scaler_size"),
            "tta_ridge_size": tta_readouts.get("ridge_size"),
            "tta_mlp_size": tta_readouts.get("mlp_size"),
        }
        print("  Dual readouts successfully fitted and frozen!\n", flush=True)
        print("  Global readouts successfully fitted and frozen!\n", flush=True)

    # 5. STEP 2: Multi-Task Evaluation across all splits
    print("=" * 75)
    print("STEP 2: Evaluating All Splits (Train, Val, Test)")
    print("=" * 75, flush=True)

    master_results: List[Dict[str, Any]] = []
    split_task_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        sp: {"anomaly": [], "sizing": [], "segmentation": []}
        for sp in args.eval_splits
    }

    for split_name in args.eval_splits:
        split_files = splits.get(split_name, [])
        print(f"\n>>> Running Evaluation on Split [{split_name.upper()}] ({len(split_files)} files) <<<", flush=True)

        # Metadata-only pre-filter (same survivor set/order as old `continue`)
        eval_files = [fp for fp in split_files
                      if extract_file_metadata(fp)["specimen"] in gt_masks]
        uncached = [fp for fp in eval_files if fp not in cached_all_latents]
        load_stream = iter(_iter_with_prefetch(uncached, args.prefetch, max_workers=4, lookahead=4))
        next_loaded = next(load_stream, None)

        pbar = tqdm(eval_files, desc=f"Evaluating [{split_name.upper()}]", file=sys.stdout, dynamic_ncols=True)
        for fp in pbar:
            meta = extract_file_metadata(fp)
            specimen = meta["specimen"]
            sensor = meta["sensor"]
            waveform = meta["waveform"]
            liftoff = meta["liftoff"]
            fname = os.path.basename(fp)
            bname = os.path.splitext(fname)[0]

            gt_mask = gt_masks[specimen]["mask"]

            # Load or retrieve cached latents (prefetched in order)
            if fp in cached_all_latents:
                latents = cached_all_latents[fp]
                raw_ptp = cached_raw_heatmaps.get(fp)
                if raw_ptp is None:
                    cscan_temp = load_canonical_cscan_from_tdms(fp)
                    raw_ptp = extract_cscan_raw_features(cscan_temp, method="filtered_ptp")
                    cached_raw_heatmaps[fp] = raw_ptp
            else:
                assert next_loaded is not None and next_loaded[0] == fp, \
                    "Prefetch stream out of order"
                cscan_cur = next_loaded[1]
                latents = extractor.extract_full_cscan(cscan_cur, self_calibrate=args.self_calibration)
                raw_ptp = extract_cscan_raw_features(cscan_cur, method="filtered_ptp")
                cached_raw_heatmaps[fp] = raw_ptp
                next_loaded = next(load_stream, None)
                cached_all_latents[fp] = latents

            # Fix 5: latent-regime OOD flag (energy vs train-pool reference)
            E_feat = scan_feature_energy(latents)
            E_ratio = (E_feat / ref_energy) if ref_energy > 0 else 1.0
            is_ood_regime = regime_flag(E_feat, ref_energy)

            # TASK 1: Anomaly Detection
            # A) Supervised Trained Linear Probe (raw latents)
            det_res = suite.evaluate_global_detection(readouts=global_readouts, features=latents, labels=gt_mask)
            prob_map = det_res["prob_map"]

            # A2) TTA Calibrated Linear Probe (P1u_trim_s2 whitened + P2h hard-negative mined).
            #     Unsupervised per-scan background whitening removes the feature-energy
            #     shift between specimens (Rivet_v2 is ~10x lower energy than Rivet_v1),
            #     which otherwise hides the probe's real detection power.
            flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            mu_bg, W_bg = unsup_trimmed_bg_stats(
                flat_z, keep_q=args.tta_trim_ratio, eps=args.tta_shrinkage
            )
            latents_tta = (flat_z - mu_bg).dot(W_bg).astype(np.float32).reshape(latents.shape)
            tta_readouts_view = {
                "scaler_det": global_readouts.get("tta_scaler_det") or global_readouts["scaler_det"],
                "lr_probe": global_readouts.get("tta_lr_probe") or global_readouts["lr_probe"],
                "mlp_probe": global_readouts.get("tta_mlp_probe") or global_readouts.get("mlp_probe"),
            }
            det_tta_res = suite.evaluate_global_detection(
                readouts=tta_readouts_view, features=latents_tta, labels=gt_mask
            )

            # B) Unsupervised Mahalanobis Distance
            anom_res = suite.compute_anomaly_detection_metrics(latents=latents, ground_truth_mask=gt_mask)

            # TASK 2: Defect Sizing Hurdle Protocol
            sizing_res = {"r2_score": 0.0, "mae_mm": 0.0, "rmse_mm": 0.0, "y_true": np.array([]), "y_pred": np.array([])}
            if specimen == "Corrosion" and corr_d_map is not None:
                sizing_res = suite.evaluate_global_sizing(
                    readouts=global_readouts,
                    features=latents,
                    size_map=corr_d_map,
                    focus_defects_only=True,
                )

            # TASK 3: Segmentation IoU & Bounding Box Overlap Map (spatiotemporal_5x5 standard)
            seg_metrics = suite.benchmark_segmentation_iou(prob_map=prob_map, gt_mask=gt_mask)
            split_seg_dir = os.path.join(task_dirs["segmentation"], split_name)
            os.makedirs(split_seg_dir, exist_ok=True)
            seg_plot_path = os.path.join(split_seg_dir, f"{bname}_segmentation_iou.png")
            iou_res = plot_defect_contours_and_iou(
                prob_map=prob_map,
                gt_mask=gt_mask,
                save_path=seg_plot_path,
                threshold=0.5,
                min_defect_area=8,
                title=f"[{split_name.upper()}] Defect Segmentation Bounding Boxes | {specimen} - {sensor} - {waveform}",
            )

            # TASK 1c: Probability of Detection (POD) & Sound Metal False Positive Rate
            specimen_cfg = gt_masks.get(specimen, {}).get("config")
            pod_rate_05 = 0.0
            pod_rate_otsu = 0.0
            if specimen_cfg is not None:
                pod_05_res = suite.compute_instance_pod(score_map=prob_map, flaw_config=specimen_cfg, threshold=0.5)
                pod_otsu_res = suite.compute_instance_pod(score_map=prob_map, flaw_config=specimen_cfg, threshold=seg_metrics.get("threshold_otsu", 0.5))
                pod_rate_05 = pod_05_res["pod"]
                pod_rate_otsu = pod_otsu_res["pod"]

            sound_fpr_05 = seg_metrics.get("sound_fpr_05", iou_res.get("sound_metal_fpr", 0.0))
            sound_fpr_otsu = seg_metrics.get("sound_fpr_otsu", 0.0)
            iou_otsu = seg_metrics.get("iou_otsu", 0.0)
            dice_otsu = seg_metrics.get("dice_otsu", 0.0)
            f1_max = seg_metrics.get("dice_opt", seg_metrics.get("dice", 0.0))

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
                "tta_lp_auc": det_tta_res["auc_roc"],
                "tta_lp_ap": det_tta_res["average_precision"],
                "mlp_probe_auc": det_res["mlp_auc_roc"],
                "mlp_probe_ap": det_res["mlp_average_precision"],
                "mahalanobis_auc": anom_res["auc_roc"],
                "mahalanobis_ap": anom_res["average_precision"],
                "cnr": anom_res["cnr"],
                "pod_rate_05": pod_rate_05,
                "pod_rate_otsu": pod_rate_otsu,
                "sound_fpr_05": sound_fpr_05,
                "sound_fpr_otsu": sound_fpr_otsu,
                "f1_max": f1_max,
                "segmentation_iou": iou_res["iou"],
                "segmentation_dice": iou_res["dice"],
                "segmentation_iou_otsu": iou_otsu,
                "segmentation_dice_otsu": dice_otsu,
                "segmentation_precision": iou_res["precision"],
                "segmentation_recall": iou_res["recall"],
                "sizing_r2": sizing_res["r2_score"],
                "sizing_mae_mm": sizing_res["mae_mm"],
                "sizing_rmse_mm": sizing_res["rmse_mm"],
                "feat_energy": round(E_feat, 1),
                "energy_ratio": round(E_ratio, 3),
                "regime_ood": is_ood_regime,
            }
            master_results.append(row)

            # Save task-specific visualizations
            # 1. Anomaly Detection: 4-Panel Summary Map (1. CAD GT, 2. Raw Peak-to-Peak, 3. Mahalanobis, 4. Linear Probe)
            split_anom_dir = os.path.join(task_dirs["anomaly"], split_name)
            os.makedirs(split_anom_dir, exist_ok=True)
            summary_plot_path = os.path.join(split_anom_dir, f"{bname}_detection_summary.png")
            raw_ptp = cached_raw_heatmaps.get(fp)
            if raw_ptp is None:
                cscan_temp = load_canonical_cscan_from_tdms(fp)
                raw_ptp = extract_cscan_raw_features(cscan_temp, method="filtered_ptp")
                cached_raw_heatmaps[fp] = raw_ptp

            async_saver.submit(
                plot_detection_summary_map,
                prob_map=prob_map,
                anomaly_map=anom_res["anomaly_scores"],
                ground_truth_mask=gt_mask,
                file_title=f"[{split_name.upper()}] {specimen} | {sensor} | {waveform} | {liftoff}",
                save_path=summary_plot_path,
                raw_heatmap=raw_ptp,
                raw_method_name="Filtered PECT (V_pp)",
                metrics={"linear_probe_auc": det_res["auc_roc"], "linear_probe_ap": det_res["average_precision"], "mahalanobis_auc": anom_res["auc_roc"], "cnr": anom_res["cnr"]},
            )

            split_task_results[split_name]["anomaly"].append({
                "file": fname,
                "specimen": specimen,
                "sensor": sensor,
                "waveform": waveform,
                "liftoff": liftoff,
                "linear_probe_auc": det_res["auc_roc"],
                "linear_probe_ap": det_res["average_precision"],
                "linear_probe_f1": det_res["f1_score"],
                "mahalanobis_auc": anom_res["auc_roc"],
                "mahalanobis_ap": anom_res["average_precision"],
                "cnr": anom_res["cnr"],
                "pod_rate_05": pod_rate_05,
                "pod_rate_otsu": pod_rate_otsu,
                "sound_fpr_05": sound_fpr_05,
            })

            # 2. Sizing Scatter Plot (if applicable)
            if specimen == "Corrosion" and len(sizing_res["y_true"]) > 0:
                split_sizing_dir = os.path.join(task_dirs["sizing"], split_name)
                os.makedirs(split_sizing_dir, exist_ok=True)
                sizing_plot_path = os.path.join(split_sizing_dir, f"{bname}_sizing.png")
                async_saver.submit(
                    plot_sizing_calibration,
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

            # 3. Segmentation Metrics Record
            split_task_results[split_name]["segmentation"].append({
                "file": fname,
                "specimen": specimen,
                "sensor": sensor,
                "waveform": waveform,
                "liftoff": liftoff,
                "iou": iou_res["iou"],
                "dice": iou_res["dice"],
                "iou_otsu": iou_otsu,
                "dice_otsu": dice_otsu,
                "f1_max": f1_max,
                "sound_fpr_05": sound_fpr_05,
                "sound_fpr_otsu": sound_fpr_otsu,
                "precision": iou_res["precision"],
                "recall": iou_res["recall"],
            })

            pbar.set_postfix({
                "LP_AUC": f"{det_res['auc_roc']:.4f}",
                "LP_AP": f"{det_res['average_precision']:.4f}",
                "POD": f"{pod_rate_otsu:.2f}",
                "CNR": f"{anom_res['cnr']:.2f}",
                "IoU": f"{iou_res['iou']:.4f}",
            })

    # Wait for split evaluation plots to finish saving
    async_saver.wait_all()

    # 6. STEP 3: Save Split-Specific Metrics & Task CSVs
    print("\n" + "=" * 75)
    print("STEP 3: Saving Structured Output Hierarchy and Aggregated Reports")
    print("=" * 75, flush=True)

    for tname in ["anomaly", "sizing", "segmentation"]:
        all_task_rows = []
        for sp in args.eval_splits:
            sp_rows = split_task_results[sp][tname]
            all_task_rows.extend(sp_rows)
            if sp_rows:
                sp_dir = os.path.join(task_dirs[tname], sp)
                os.makedirs(sp_dir, exist_ok=True)
                with open(os.path.join(sp_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(sp_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(sp_rows)

                sp_summary = {}
                for k in sp_rows[0].keys():
                    vals = [r[k] for r in sp_rows if isinstance(r[k], (int, float))]
                    if vals:
                        sp_summary[f"mean_{k}"] = float(np.mean(vals))
                        sp_summary[f"std_{k}"] = float(np.std(vals))
                with open(os.path.join(sp_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"task": tname, "split": sp, "count": len(sp_rows), "metrics": sp_summary}, f, indent=2)

        if all_task_rows:
            t_csv = os.path.join(task_dirs[tname], "metrics.csv")
            with open(t_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_task_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_task_rows)

            t_json = os.path.join(task_dirs[tname], "summary.json")
            num_summary = {}
            for k in all_task_rows[0].keys():
                vals = [r[k] for r in all_task_rows if isinstance(r[k], (int, float))]
                if vals:
                    num_summary[f"mean_{k}"] = float(np.mean(vals))
                    num_summary[f"std_{k}"] = float(np.std(vals))
            with open(t_json, "w", encoding="utf-8") as f:
                json.dump({"task": tname, "count": len(all_task_rows), "metrics": num_summary}, f, indent=2)

    # 7. STEP 4: Canonical Test-Time Adaptation (TTA) Benchmark (Single-Variable Controlled Shifts)
    canonical_tta_results = None
    run_tta = bool(args.eval_tta) and not bool(args.skip_tta) and not bool(args.skip_domain_adaptation)
    if run_tta:
        canonical_tta_results = evaluate_canonical_test_time_adaptation(
            downstream_dir=downstream_dir,
            out_dir=out_dir,
            all_files=all_files,
            gt_masks=gt_masks,
            cached_latents=cached_all_latents,
            extractor=extractor,
            self_calibration=args.self_calibration,
            global_readouts=global_readouts,
            device=device,
            tta_trim_ratio=args.tta_trim_ratio,
            tta_shrinkage=args.tta_shrinkage,
            split_mode=args.split_mode,
            cached_raw_heatmaps=cached_raw_heatmaps,
            async_saver=async_saver,
        )
        async_saver.wait_all()
    else:
        print("\n" + "=" * 75)
        print("STEP 4: Canonical Test-Time Adaptation (TTA) Benchmark [SKIPPED by user configuration]")
        print("=" * 75, flush=True)

    # 8. STEP 5: Save Master Reports
    master_csv = os.path.join(out_dir, "master_benchmark_metrics.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_results[0].keys()))
        writer.writeheader()
        writer.writerows(master_results)

    def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        lp_auc = [r["linear_probe_auc"] for r in rows]
        lp_ap = [r["linear_probe_ap"] for r in rows]
        tta_lp_auc = [r.get("tta_lp_auc", 0.0) for r in rows]
        tta_lp_ap = [r.get("tta_lp_ap", 0.0) for r in rows]
        mahal_auc = [r["mahalanobis_auc"] for r in rows]
        cnr = [r["cnr"] for r in rows]
        pod_05 = [r.get("pod_rate_05", 0.0) for r in rows]
        pod_otsu = [r.get("pod_rate_otsu", 0.0) for r in rows]
        sound_fpr_05 = [r.get("sound_fpr_05", 0.0) for r in rows]
        sound_fpr_otsu = [r.get("sound_fpr_otsu", 0.0) for r in rows]
        f1_max = [r.get("f1_max", 0.0) for r in rows]
        iou = [r["segmentation_iou"] for r in rows]
        dice = [r["segmentation_dice"] for r in rows]
        iou_otsu = [r.get("segmentation_iou_otsu", 0.0) for r in rows]
        dice_otsu = [r.get("segmentation_dice_otsu", 0.0) for r in rows]
        sz_r2 = [r["sizing_r2"] for r in rows if r["specimen"] == "Corrosion"]
        sz_mae = [r["sizing_mae_mm"] for r in rows if r["specimen"] == "Corrosion"]
        return {
            "count": len(rows),
            "linear_probe_auc_mean": float(np.mean(lp_auc)),
            "linear_probe_auc_std": float(np.std(lp_auc)),
            "linear_probe_ap_mean": float(np.mean(lp_ap)),
            "linear_probe_ap_std": float(np.std(lp_ap)),
            "tta_lp_auc_mean": float(np.mean(tta_lp_auc)),
            "tta_lp_auc_std": float(np.std(tta_lp_auc)),
            "tta_lp_ap_mean": float(np.mean(tta_lp_ap)),
            "mahalanobis_auc_mean": float(np.mean(mahal_auc)),
            "cnr_mean": float(np.mean(cnr)),
            "pod_rate_05_mean": float(np.mean(pod_05)),
            "pod_rate_otsu_mean": float(np.mean(pod_otsu)),
            "sound_fpr_05_mean": float(np.mean(sound_fpr_05)),
            "sound_fpr_otsu_mean": float(np.mean(sound_fpr_otsu)),
            "f1_max_mean": float(np.mean(f1_max)),
            "segmentation_iou_mean": float(np.mean(iou)),
            "segmentation_dice_mean": float(np.mean(dice)),
            "segmentation_iou_otsu_mean": float(np.mean(iou_otsu)),
            "segmentation_dice_otsu_mean": float(np.mean(dice_otsu)),
            "sizing_r2_mean": float(np.mean(sz_r2)) if sz_r2 else 0.0,
            "sizing_mae_mean": float(np.mean(sz_mae)) if sz_mae else 0.0,
        }

    def _effective_latent_dim(cfg) -> int:
        return int(cfg.embed_dim * 2)

    _test_rows = [r for r in master_results if r["split"] == "test"]
    by_axis_test = {}
    if _test_rows:
        by_axis_test = {
            "by_waveform_test": {wf: compute_stats([r for r in _test_rows if r["waveform"] == wf])
                                 for wf in ["Square", "Gaussian", "Chirp"]},
            "by_sensor_test": {sn: compute_stats([r for r in _test_rows if r["sensor"] == sn])
                               for sn in ["Hall_Air_Core", "Hall_Pot_Core", "TMR"]},
            "by_liftoff_test": {lo: compute_stats([r for r in _test_rows if r["liftoff"] == lo])
                                for lo in ["z1", "z2", "z3"]},
            "by_specimen_test": {sp: compute_stats([r for r in _test_rows if r["specimen"] == sp])
                                 for sp in ["Corrosion", "Rivet_v1", "Rivet_v2"]},
        }

    master_summary = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": ckpt_path,
        "output_directory": out_dir,
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
        # AUDIT-2026-10-23: the by_* tables above span EVERY evaluated file, so a
        # train/val file leaks into a per-axis number. Any per-axis claim must use
        # the *_test variants below (test-only rows).
        "scope_note": ("by_* tables span all evaluated splits; use by_*_test for "
                       "per-axis claims. latent_config records the effective "
                       "representation and calibration flags."),
        "latent_config": {
            "split_mode": args.split_mode,
            "self_calibration": bool(getattr(args, "self_calibration", False)),
            "eval_tta": bool(getattr(args, "eval_tta", False)),
            "sizing_log_target": bool(getattr(args, "sizing_log_target", False)),
            "latent_dim": _effective_latent_dim(config),
        },
        **by_axis_test,
        "canonical_test_time_adaptation": canonical_tta_results,
        "canonical_domain_adaptation": canonical_tta_results,
    }

    master_json = os.path.join(out_dir, "master_benchmark_summary.json")
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # Write Markdown Report
    md_path = os.path.join(out_dir, "EVALUATION_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PECT-JEPA Unified Downstream Benchmark Report\n\n")
        f.write(f"- **Checkpoint**: `{ckpt_path}`\n")
        f.write(f"- **Evaluation Results Directory**: `{out_dir}`\n")
        f.write(f"- **Total Files Evaluated**: {len(master_results)}\n\n")

        f.write("## 1. Split Performance Breakdown\n\n")
        f.write("| Split | Files | Linear Probe AUC | TTA LP AUC (P1u_trim_s2 + P2h) | Linear Probe AP | CNR | Mahalanobis AUC | POD (Otsu) | Seg IoU (Otsu) | Sound FPR | Sizing R² |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for sp in args.eval_splits:
            st = master_summary["by_split"].get(sp, {})
            if st:
                f.write(f"| **{sp.upper()}** | {st['count']} | {st['linear_probe_auc_mean']:.4f} ± {st['linear_probe_auc_std']:.4f} | {st.get('tta_lp_auc_mean', 0.0):.4f} ± {st.get('tta_lp_auc_std', 0.0):.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['mahalanobis_auc_mean']:.4f} | {st.get('pod_rate_otsu_mean', 0.0):.4f} | {st.get('segmentation_iou_otsu_mean', 0.0):.4f} | {st.get('sound_fpr_otsu_mean', 0.0):.4f} | {st['sizing_r2_mean']:.4f} |\n")

        f.write("\n## 2. Generalization Disaggregation\n\n")
        f.write("> AUDIT-2026-10-23: these tables are **test-only** (`by_*_test`). The "
                "`by_*` fields in the JSON still span every evaluated split and must not "
                "be quoted as generalization.\n\n")
        f.write("### By Waveform\n")
        f.write("| Waveform | Files | LP AUC | LP AP | CNR | POD (Otsu) | Otsu IoU | Sound FPR |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for wf, st in master_summary.get("by_waveform_test", master_summary["by_waveform"]).items():
            if st.get("count", 0) > 0:
                f.write(f"| {wf} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st.get('pod_rate_otsu_mean', 0.0):.4f} | {st.get('segmentation_iou_otsu_mean', 0.0):.4f} | {st.get('sound_fpr_otsu_mean', 0.0):.4f} |\n")

        f.write("\n### By Sensor\n")
        f.write("| Sensor | Files | LP AUC | LP AP | CNR | POD (Otsu) | Otsu IoU | Sound FPR |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for sn, st in master_summary.get("by_sensor_test", master_summary["by_sensor"]).items():
            if st.get("count", 0) > 0:
                f.write(f"| {sn} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st.get('pod_rate_otsu_mean', 0.0):.4f} | {st.get('segmentation_iou_otsu_mean', 0.0):.4f} | {st.get('sound_fpr_otsu_mean', 0.0):.4f} |\n")

        f.write("\n### By Lift-off\n")
        f.write("| Lift-off | Files | LP AUC | LP AP | CNR | POD (Otsu) | Otsu IoU | Sound FPR |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for lo, st in master_summary.get("by_liftoff_test", master_summary["by_liftoff"]).items():
            if st.get("count", 0) > 0:
                f.write(f"| {lo} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st.get('pod_rate_otsu_mean', 0.0):.4f} | {st.get('segmentation_iou_otsu_mean', 0.0):.4f} | {st.get('sound_fpr_otsu_mean', 0.0):.4f} |\n")

        f.write("\n### By Specimen\n")
        f.write("| Specimen | Files | LP AUC | LP AP | CNR | POD (Otsu) | Otsu IoU | Sound FPR |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for sp, st in master_summary.get("by_specimen_test", master_summary["by_specimen"]).items():
            if st.get("count", 0) > 0:
                f.write(f"| {sp} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st.get('pod_rate_otsu_mean', 0.0):.4f} | {st.get('segmentation_iou_otsu_mean', 0.0):.4f} | {st.get('sound_fpr_otsu_mean', 0.0):.4f} |\n")

        # Section 3: Canonical TTA Benchmark
        if canonical_tta_results:
            f.write("\n## 3. Canonical Test-Time Adaptation (TTA) Benchmark (Controlled Single-Variable Shifts)\n\n")
            f.write("| Shift | Method / Mechanism | Source AUC (Retention) | Target AUC (Adaptation) | Target AP | Δ AUC vs Zero-Shot LP | Strategy |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
            tta_strat_desc = {
                "Zero-Shot LP (Raw)": "Source-only frozen linear readout",
                "Zero-Shot MLP (Raw)": "Source-only non-linear MLP readout",
                "TTA Calibrated LP": "Unsupervised background whitening + Linear Probe",
                "TTA Calibrated MLP": "Unsupervised background whitening + MLP Probe",
                "TTA Mahalanobis (Unsupervised)": "Threshold-free energy ranking on whitened background",
            }
            for s_name, s_data in canonical_tta_results.items():
                m_dict = s_data.get("methods", {})
                for m_name, m_res in m_dict.items():
                    s_auc = m_res["source"]["auc_mean"]
                    t_auc = m_res["target"]["auc_mean"]
                    t_std = m_res["target"]["auc_std"]
                    t_ap = m_res["target"]["ap_mean"]
                    delta = m_res["target"]["delta_auc_mean"]
                    delta_str = f"**{'+' if delta >= 0 else ''}{delta:.4f}**" if "Zero-Shot" not in m_name else "*baseline*"
                    strat = tta_strat_desc.get(m_name, "Test-Time Self-Calibration")
                    f.write(f"| **{s_name.upper()}** | **{m_name}** | {s_auc:.4f} | {t_auc:.4f} ± {t_std:.4f} | {t_ap:.4f} | {delta_str} | {strat} |\n")

    # 9. Terminal Output
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    for sp in args.eval_splits:
        st = master_summary["by_split"].get(sp, {})
        if st:
            print(f"[{sp.upper():5s}] ({st['count']} files) -> LP AUC: {st['linear_probe_auc_mean']:.4f} +/- {st['linear_probe_auc_std']:.4f} | TTA LP AUC: {st.get('tta_lp_auc_mean', 0.0):.4f} | LP AP: {st['linear_probe_ap_mean']:.4f} | CNR: {st['cnr_mean']:.2f} | Seg IoU: {st['segmentation_iou_mean']:.4f} | Sizing R2: {st['sizing_r2_mean']:.4f}")
    print("-" * 80)
    ov = master_summary["overall"]
    print(f"[OVERALL] ({ov['count']} files) -> LP AUC: {ov['linear_probe_auc_mean']:.4f} +/- {ov['linear_probe_auc_std']:.4f} | LP AP: {ov['linear_probe_ap_mean']:.4f} | CNR: {ov['cnr_mean']:.2f} | Seg IoU: {ov['segmentation_iou_mean']:.4f}")

    # Canonical TTA Summary in Console
    if canonical_tta_results:
        print("\n" + "=" * 80)
        print("CANONICAL TEST-TIME ADAPTATION (TTA) BENCHMARK SUMMARY")
        print("=" * 80)
        for s_name, s_data in canonical_tta_results.items():
            m_dict = s_data.get("methods", {})
            print(f"\n--- Shift: [{s_name.upper()}] ---")
            print(f"{'Method / Mechanism':<32} | {'Source AUC':<12} | {'Target AUC':<16} | {'Target AP':<10} | {'Delta vs LP':<12}")
            print("-" * 80)
            for m_name, m_res in m_dict.items():
                s_auc = m_res["source"]["auc_mean"]
                t_auc = m_res["target"]["auc_mean"]
                t_std = m_res["target"]["auc_std"]
                t_ap = m_res["target"]["ap_mean"]
                delta = m_res["target"]["delta_auc_mean"]
                delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}" if "Zero-Shot" not in m_name else "baseline"
                auc_str = f"{t_auc:.4f} +/- {t_std:.4f}"
                print(f"{m_name:<32} | {s_auc:<12.4f} | {auc_str:<16} | {t_ap:<10.4f} | {delta_str:<12}")
        print("=" * 80)
    elif not run_tta:
        print("\n" + "-" * 80)
        print("[INFO] Canonical Test-Time Adaptation (TTA) benchmark was SKIPPED by option (--skip_tta / --eval_tta=False).")
        print("-" * 80)

    for sp in args.eval_splits:
        n_flag = sum(1 for r in master_results if r["split"] == sp and r.get("regime_ood", 0) == 1)
        n_sp = sum(1 for r in master_results if r["split"] == sp)
        print(f"[REGIME] {sp.upper():5s}: {n_flag}/{n_sp} files flagged OOD-regime (E||z||^2 outside 2x train median {ref_energy:.1f})")
    async_saver.shutdown()
    print(f"\nAll results saved to: {out_dir}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
