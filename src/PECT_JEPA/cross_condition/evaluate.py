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
    extract_file_metadata,
    get_canonical_train_val_test_splits,
    get_canonical_shift_files,
    is_base_domain_file,
)
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.coral import CORALAdapter
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.dann import DANNModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.dan_mmd import DANModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.cdan import CDANModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.shot import SHOTModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.otda import OTDAModel
from src.PECT_JEPA.cross_condition.downstream.domain_adaptation.cbst import CBSTModel
from src.PECT_JEPA.cross_condition.evaluation.cscan_extractor import CScanFeatureExtractor
from src.PECT_JEPA.cross_condition.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.cross_condition.evaluation.visualizations import (
    plot_probability_heatmap,
    plot_detection_summary_map,
    plot_cscan_anomaly_map,
    plot_sizing_calibration,
    plot_defect_contours_and_iou,
    plot_uda_summary_map,
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
    parser.add_argument("--prefetch", type=lambda v: v.lower() == "true", default=True, help="Overlap next-file TDMS load with current-file GPU/CPU processing")
    parser.add_argument("--self_calibration", type=lambda v: v.lower() == "true", default=True, help="Enable unsupervised spatial self-calibration (zero-centering C-scan against sound-metal background median)")
    parser.add_argument("--downstream_dir", type=str, default=None, help="Directory containing pre-trained downstream models (standard_probes.joblib, coral_model.pt, dann_model.pt)")
    parser.add_argument("--skip_domain_adaptation", "--skip_uda", dest="skip_domain_adaptation", nargs="?", const=True, type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=False, help="Skip canonical domain adaptation evaluation task (UDA models)")
    parser.add_argument("--eval_domain_adaptation", "--run_uda", dest="eval_domain_adaptation", type=lambda v: v.lower() == "true" if isinstance(v, str) else bool(v), default=True, help="Whether to evaluate canonical domain adaptation (UDA models). Set to False to skip.")
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


def _iter_with_prefetch(files: List[str], use_prefetch: bool, max_workers: int = 2, lookahead: int = 2):
    """Yields (file_path, canonical_cscan) in order, decoding up to `lookahead`
    TDMS files concurrently on background threads while the caller processes
    the current file (TDMS decode + filtering ~3 s hidden behind GPU
    extraction + sklearn + plots). Processing order, RNG consumption, and
    results are identical to the sequential loop; use_prefetch=False
    reproduces the old loop exactly.
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


def evaluate_canonical_domain_adaptation(
    downstream_dir: str,
    out_dir: str,
    all_files: List[str],
    gt_masks: Dict[str, Any],
    cached_latents: Dict[str, np.ndarray],
    extractor: CScanFeatureExtractor,
    self_calibration: bool,
    lr_probe: Any,
    scaler_det: Any,
    device: torch.device,
    in_dim: int,
) -> Dict[str, Any]:
    """
    Evaluates UDA Suite across the 4 canonical single-variable domain shifts:
    - cross_sensor:   Source (16 Base files) -> Target (8 TMR-only files)
    - cross_waveform: Source (16 Base files) -> Target (8 Chirp-only files)
    - cross_specimen: Source (16 Base files) -> Target (8 Rivet_v2-only files)
    - cross_liftoff:  Source (16 Base files) -> Target (8 z2-only files)
    
    Structure:
      4. Domain Adaptation/<shift>/<MODEL_NAME>/source/
      4. Domain Adaptation/<shift>/<MODEL_NAME>/target/
      4. Domain Adaptation/<shift>/uda_summary.csv
      4. Domain Adaptation/<shift>/uda_summary.json
    """
    canonical_shifts = ["cross_sensor", "cross_waveform", "cross_specimen", "cross_liftoff"]
    shift_results: Dict[str, Any] = {}

    uda_constructors = {
        "CORAL": (CORALAdapter, "coral_model.pt"),
        "DANN": (DANNModel, "dann_model.pt"),
        "DAN": (DANModel, "dan_model.pt"),
        "CDAN": (CDANModel, "cdan_model.pt"),
        "SHOT": (SHOTModel, "shot_model.pt"),
        "OTDA": (OTDAModel, "otda_model.pt"),
        "CBST": (CBSTModel, "cbst_model.pt"),
    }

    print("\n" + "=" * 75)
    print("STEP 4: Canonical Domain Adaptation Benchmark (Single-Variable Controlled Shifts)")
    print("=" * 75, flush=True)

    for shift in canonical_shifts:
        shift_dir = os.path.join(downstream_dir, shift)
        if not os.path.isdir(shift_dir):
            shift_dir = downstream_dir

        loaded_models: Dict[str, Any] = {}
        for m_name, (cls, weight_file) in uda_constructors.items():
            w_path = os.path.join(shift_dir, weight_file)
            if os.path.exists(w_path):
                try:
                    m = cls(in_dim=in_dim, hidden_dim=64).to(device)
                    m.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
                    m.eval()
                    loaded_models[m_name] = m
                except Exception as e:
                    print(f"    [WARN] Failed to load {m_name} from {w_path}: {e}")

        if not loaded_models:
            continue

        src_files, tgt_files = get_canonical_shift_files(all_files, shift)
        shift_out_dir = os.path.join(out_dir, "4. Domain Adaptation", shift)
        for m_name in loaded_models:
            os.makedirs(os.path.join(shift_out_dir, m_name, "source"), exist_ok=True)
            os.makedirs(os.path.join(shift_out_dir, m_name, "target"), exist_ok=True)

        print("\n" + "-" * 75)
        print(f"Evaluating Canonical Shift [{shift.upper()}]: {len(loaded_models)} Models Loaded")
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

                # Retrieve or extract latents
                if fp in cached_latents:
                    latents = cached_latents[fp]
                else:
                    cscan = load_canonical_cscan_from_tdms(fp)
                    latents = extractor.extract_full_cscan(cscan, self_calibrate=self_calibration)
                    cached_latents[fp] = latents

                H, W = latents.shape[0], latents.shape[1]
                flat_feats = latents.reshape(-1, latents.shape[-1]).astype(np.float32)

                # Linear Probe baseline prediction
                lp_prob = None
                lp_auc, lp_ap = 0.0, 0.0
                if lr_probe is not None and scaler_det is not None:
                    flat_scaled = scaler_det.transform(flat_feats)
                    lp_prob = lr_probe.predict_proba(flat_scaled)[:, 1]
                    if has_both:
                        lp_auc = float(roc_auc_score(gt_mask.reshape(-1)[valid], lp_prob[valid]))
                        lp_ap = float(average_precision_score(gt_mask.reshape(-1)[valid], lp_prob[valid]))
                lp_prob_map = lp_prob.reshape(H, W) if lp_prob is not None else np.zeros((H, W), dtype=np.float32)

                # Evaluate each UDA model
                for m_name, m_net in loaded_models.items():
                    prob = m_net.predict_proba(flat_feats, device=str(device))
                    prob_map = prob.reshape(H, W)
                    m_auc, m_ap = 0.0, 0.0
                    if has_both:
                        m_auc = float(roc_auc_score(gt_mask.reshape(-1)[valid], prob[valid]))
                        m_ap = float(average_precision_score(gt_mask.reshape(-1)[valid], prob[valid]))

                    # Save visualizations
                    save_dir = os.path.join(shift_out_dir, m_name, set_name)
                    prob_path = os.path.join(save_dir, f"{bname}_{m_name.lower()}_prob.png")
                    plot_probability_heatmap(
                        prob_map=prob_map,
                        save_path=prob_path,
                        title=f"{m_name} [{shift.upper()} | {set_name.upper()}]\n{sp} - {meta['sensor']} - {meta['waveform']} | AUC: {m_auc:.4f} | AP: {m_ap:.4f}",
                        cmap="jet",
                    )

                    comp_path = os.path.join(save_dir, f"{bname}_{m_name.lower()}_comparison.png")
                    plot_uda_summary_map(
                        lp_prob_map=lp_prob_map,
                        uda_prob_map=prob_map,
                        ground_truth_mask=gt_mask,
                        model_name=m_name,
                        file_title=f"[{shift.upper()} | {set_name.upper()}] {sp} | {meta['sensor']} | {meta['waveform']}",
                        save_path=comp_path,
                        lp_metrics={"auc": lp_auc, "ap": lp_ap},
                        uda_metrics={"auc": m_auc, "ap": m_ap},
                    )

                    # A file whose GT mask has a single class has no defined AUC; averaging a
                    # placeholder 0.0 into auc_mean silently drags the shift summary down.
                    if not has_both:
                        print(f"    [skip] {m_name} {set_name} {fname}: GT mask has a single class")
                        continue
                    shift_rows.append({
                        "shift": shift,
                        "domain_set": set_name,
                        "model": m_name,
                        "file": fname,
                        "specimen": sp,
                        "sensor": meta["sensor"],
                        "waveform": meta["waveform"],
                        "liftoff": meta["liftoff"],
                        "auc": round(m_auc, 4),
                        "ap": round(m_ap, 4),
                        "lp_auc": round(lp_auc, 4),
                        "lp_ap": round(lp_ap, 4),
                        "delta_auc": round(m_auc - lp_auc, 4),
                    })

        # Save Shift CSV & Summary
        if shift_rows:
            csv_path = os.path.join(shift_out_dir, "uda_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
                writer.writeheader()
                writer.writerows(shift_rows)

            shift_summary: Dict[str, Any] = {"shift": shift, "models": {}}
            for m_name in loaded_models:
                shift_summary["models"][m_name] = {}
                for s_name in ["source", "target"]:
                    rows = [r for r in shift_rows if r["model"] == m_name and r["domain_set"] == s_name]
                    aucs = [r["auc"] for r in rows]
                    aps = [r["ap"] for r in rows]
                    lp_aucs = [r["lp_auc"] for r in rows]
                    deltas = [r["delta_auc"] for r in rows]
                    shift_summary["models"][m_name][s_name] = {
                        "count": len(rows),
                        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
                        "auc_std": float(np.std(aucs)) if aucs else 0.0,
                        "ap_mean": float(np.mean(aps)) if aps else 0.0,
                        "lp_auc_mean": float(np.mean(lp_aucs)) if lp_aucs else 0.0,
                        "delta_auc_mean": float(np.mean(deltas)) if deltas else 0.0,
                    }

            json_path = os.path.join(shift_out_dir, "uda_summary.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(shift_summary, f, indent=2)

            # Print Console Summary Table
            tgt_rows = [r for r in shift_rows if r["domain_set"] == "target"]
            lp_tgt_mean = np.mean([r["lp_auc"] for r in tgt_rows if r["model"] == list(loaded_models.keys())[0]]) if tgt_rows else 0.0
            lp_tgt_ap = np.mean([r["lp_ap"] for r in tgt_rows if r["model"] == list(loaded_models.keys())[0]]) if tgt_rows else 0.0

            src_rows = [r for r in shift_rows if r["domain_set"] == "source"]
            lp_src_mean = np.mean([r["lp_auc"] for r in src_rows if r["model"] == list(loaded_models.keys())[0]]) if src_rows else 0.0

            print("\n" + "=" * 90)
            print(f"CANONICAL UDA BENCHMARK SUMMARY: [{shift.upper()}]")
            print(f"  Source Domain: {len(src_files)} files (Train/Val Pool) | Target Domain: {len(tgt_files)} files (1-Variable Shift)")
            print("=" * 90)
            print(f"{'Model / Method':<24} | {'Source AUC (Retention)':<22} | {'Target AUC (Adaptation)':<23} | {'Target AP':<9} | {'Δ AUC vs LP':<12}")
            print("-" * 90)
            print(f"{'Linear Probe (Zero-Shot)':<24} | {lp_src_mean:<22.4f} | {lp_tgt_mean:<23.4f} | {lp_tgt_ap:<9.4f} | {'baseline':<12}")
            for m_name in loaded_models:
                m_src = shift_summary["models"][m_name]["source"]["auc_mean"]
                m_tgt = shift_summary["models"][m_name]["target"]["auc_mean"]
                m_tgt_std = shift_summary["models"][m_name]["target"]["auc_std"]
                m_ap = shift_summary["models"][m_name]["target"]["ap_mean"]
                delta = shift_summary["models"][m_name]["target"]["delta_auc_mean"]
                sign = "+" if delta >= 0 else ""
                print(f"{m_name:<24} | {m_src:<22.4f} | {m_tgt:.4f} ± {m_tgt_std:<15.4f} | {m_ap:<9.4f} | {sign}{delta:<11.4f}")
            print("=" * 90 + "\n", flush=True)

            shift_results[shift] = shift_summary

    return shift_results


def main():
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

    task_dirs = {
        "anomaly": os.path.join(out_dir, "1_Anomaly_Detection"),
        "sizing": os.path.join(out_dir, "2_Defect_Sizing"),
        "segmentation": os.path.join(out_dir, "3_Segmentation_IoU"),
    }
    for tname, tdir in task_dirs.items():
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(tdir, sp), exist_ok=True)

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
    gt_masks = load_ground_truth_masks()
    corr_d_map = None
    if "Corrosion" in gt_masks and gt_masks["Corrosion"]["config"] is not None:
        corr_d_map = build_corrosion_diameter_map(gt_masks["Corrosion"]["config"])

    # 3. Discover All TDMS Files & Canonical Partitioning
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.tdms"), recursive=True))
    all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]

    splits = get_canonical_train_val_test_splits(all_files, val_files_count=4, seed=42)
    # Pre-build the canonical transformer on the main thread so background
    # prefetch workers never race on the lazy global.
    get_canonical_transformer()
    print(f"Discovered {len(all_files)} total inspection files:")
    print(f"  - Train Split (Base Domain):     {len(splits['train'])} files")
    print(f"  - Val Split (Base Domain):       {len(splits['val'])} files")
    print(f"  - Test Split (Held-out OOD):     {len(splits['test'])} files\n", flush=True)

    # 4. STEP 1: Load Pre-trained Downstream Models or Fit on Train Pool
    print("=" * 75)
    print("STEP 1: Downstream Probes & UDA Models Setup")
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
    train_energies: List[float] = []

    if downstream_dir is not None and os.path.exists(os.path.join(downstream_dir, "standard_probes.joblib")):
        print(f"  [INFO] Loading pre-trained downstream models from: {downstream_dir}")
        bundle = joblib.load(os.path.join(downstream_dir, "standard_probes.joblib"))
        det_bundle = bundle["detection"]
        size_bundle = bundle.get("sizing")
        global_readouts = {
            "scaler_det": det_bundle.scaler,
            "lr_probe": det_bundle.linear_probe,
            "mlp_probe": det_bundle.mlp_probe,
        }
        if size_bundle is not None:
            global_readouts["scaler_size"] = size_bundle.scaler
            global_readouts["ridge_size"] = size_bundle.ridge_regressor
            global_readouts["mlp_size"] = size_bundle.mlp_regressor

        ref_energy = 0.0
        print("  Pre-trained standard downstream models loaded successfully!\n", flush=True)
    else:
        print("\n" + "!" * 75)
        print("  [WARNING] No pre-trained downstream models found!")
        print("  Checked candidate paths:")
        for cand in candidates:
            print(f"    - {cand} (exists: {os.path.exists(cand)})")
        print("  Fitting readouts on train pool as fallback (UDA models will NOT be evaluated).")
        print("  To train and evaluate UDA models, run train_downstream.py first:")
        print(f"    python -m src.PECT_JEPA.cross_condition.downstream.train_downstream --checkpoint {ckpt_path} --data_dir {args.data_dir}")
        print("!" * 75 + "\n", flush=True)
        train_det_x, train_det_y = [], []
        train_size_x, train_size_y = [], []

        train_eval_files = [fp for fp in splits["train"]
                            if extract_file_metadata(fp)["specimen"] in gt_masks]
        pbar_train = tqdm(_iter_with_prefetch(train_eval_files, args.prefetch),
                          total=len(train_eval_files),
                          desc="Fitting Readouts on Train Pool", file=sys.stdout, dynamic_ncols=True)
        rng = np.random.RandomState(42)

        for fp, cscan in pbar_train:
            meta = extract_file_metadata(fp)
            sp = meta["specimen"]

            gt_mask = gt_masks[sp]["mask"]
            latents = extractor.extract_full_cscan(cscan, self_calibrate=args.self_calibration)
            cached_train_latents[fp] = latents
            train_energies.append(scan_feature_energy(latents))

            flat_z = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
            flat_y = gt_mask.reshape(-1).astype(np.int64)

            pos_idx = np.where(flat_y == 1)[0]
            neg_idx = np.where(flat_y == 0)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                sub_neg_count = min(len(neg_idx), max(len(pos_idx) * 4, 2500))
                sub_neg = rng.choice(neg_idx, size=sub_neg_count, replace=False)
                sel = np.concatenate([pos_idx, sub_neg])
                train_det_x.append(flat_z[sel])
                train_det_y.append(flat_y[sel])

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
        ref_energy = float(np.median(train_energies)) if train_energies else 0.0
        print(f"  - Latent regime reference (median scan E||z||^2): {ref_energy:.1f}", flush=True)

        print(f"  Fitting global Logistic Regression & Ridge Regressors...", flush=True)
        global_readouts = suite.fit_global_readouts(
            X_det=X_train_det,
            y_det=y_train_det,
            X_size=X_train_size,
            y_size=y_train_size,
        )
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
        load_stream = iter(_iter_with_prefetch(uncached, args.prefetch))
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
            else:
                assert next_loaded is not None and next_loaded[0] == fp, \
                    "Prefetch stream out of order"
                latents = extractor.extract_full_cscan(next_loaded[1], self_calibrate=args.self_calibration)
                next_loaded = next(load_stream, None)
                cached_all_latents[fp] = latents

            # Fix 5: latent-regime OOD flag (energy vs train-pool reference)
            E_feat = scan_feature_energy(latents)
            E_ratio = (E_feat / ref_energy) if ref_energy > 0 else 1.0
            is_ood_regime = regime_flag(E_feat, ref_energy)

            # TASK 1: Anomaly Detection
            # A) Supervised Trained Linear Probe
            det_res = suite.evaluate_global_detection(readouts=global_readouts, features=latents, labels=gt_mask)
            prob_map = det_res["prob_map"]

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
            seg_plot_path = os.path.join(task_dirs["segmentation"], split_name, f"{bname}_segmentation_iou.png")
            iou_res = plot_defect_contours_and_iou(
                prob_map=prob_map,
                gt_mask=gt_mask,
                save_path=seg_plot_path,
                threshold=0.5,
                min_defect_area=8,
                title=f"[{split_name.upper()}] Defect Segmentation Bounding Boxes | {specimen} - {sensor} - {waveform}",
            )

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
                "feat_energy": round(E_feat, 1),
                "energy_ratio": round(E_ratio, 3),
                "regime_ood": is_ood_regime,
            }
            master_results.append(row)

            # Save task-specific visualizations
            # 1. Anomaly Detection: Trained Linear Probe Probability Heatmap (cmap='jet')
            lp_plot_path = os.path.join(task_dirs["anomaly"], split_name, f"{bname}_linear_probe_prob.png")
            plot_probability_heatmap(
                prob_map=prob_map,
                save_path=lp_plot_path,
                title=f"Trained Linear Probe P(Defect | z) [{split_name.upper()}]\n{specimen} - {sensor} - {waveform} | AUC: {det_res['auc_roc']:.4f} | AP: {det_res['average_precision']:.4f}",
                cmap="jet",
            )

            # Unsupervised Mahalanobis Anomaly Map (cmap='jet')
            maha_plot_path = os.path.join(task_dirs["anomaly"], split_name, f"{bname}_mahalanobis.png")
            plot_cscan_anomaly_map(
                anomaly_map=anom_res["anomaly_scores"],
                ground_truth_mask=gt_mask,
                file_title=f"[{split_name.upper()}] {specimen} | {sensor} | {waveform} | {liftoff}",
                save_path=maha_plot_path,
                metrics=anom_res,
            )

            # 3-Panel Detection Summary (Trained Linear Probe vs Mahalanobis vs CAD GT)
            summary_plot_path = os.path.join(task_dirs["anomaly"], split_name, f"{bname}_detection_summary.png")
            plot_detection_summary_map(
                prob_map=prob_map,
                anomaly_map=anom_res["anomaly_scores"],
                ground_truth_mask=gt_mask,
                file_title=f"[{split_name.upper()}] {specimen} | {sensor} | {waveform} | {liftoff}",
                save_path=summary_plot_path,
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

            # 3. Segmentation Metrics Record
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

    # 6. STEP 3: Save Split-Specific Metrics & Task CSVs
    print("\n" + "=" * 75)
    print("STEP 3: Saving Structured Output Hierarchy and Aggregated Reports")
    print("=" * 75, flush=True)

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
                num_summary = {}
                for k in trows[0].keys():
                    vals = [r[k] for r in trows if isinstance(r[k], (int, float))]
                    if vals:
                        num_summary[f"mean_{k}"] = float(np.mean(vals))
                        num_summary[f"std_{k}"] = float(np.std(vals))
                with open(t_json, "w", encoding="utf-8") as f:
                    json.dump({"split": sp, "task": tname, "count": len(trows), "metrics": num_summary}, f, indent=2)

    # 7. STEP 4: Canonical Domain Adaptation Benchmark (Single-Variable Controlled Shifts)
    in_dim = config.carrier_dim + config.scattering_dim if config.latent_disentanglement else config.embed_dim
    canonical_uda_results = None
    run_uda = bool(args.eval_domain_adaptation) and not bool(args.skip_domain_adaptation)
    if run_uda:
        if downstream_dir is not None and os.path.isdir(downstream_dir):
            canonical_uda_results = evaluate_canonical_domain_adaptation(
                downstream_dir=downstream_dir,
                out_dir=out_dir,
                all_files=all_files,
                gt_masks=gt_masks,
                cached_latents=cached_all_latents,
                extractor=extractor,
                self_calibration=args.self_calibration,
                lr_probe=global_readouts.get("lr_probe") if global_readouts else None,
                scaler_det=global_readouts.get("scaler_det") if global_readouts else None,
                device=device,
                in_dim=in_dim,
            )
    else:
        print("\n" + "=" * 75)
        print("STEP 4: Canonical Domain Adaptation Benchmark [SKIPPED by user configuration]")
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
        "canonical_domain_adaptation": canonical_uda_results,
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
        f.write("| Split | Files | Linear Probe AUC | Linear Probe AP | CNR | Mahalanobis AUC | Seg IoU | Sizing R² |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for sp in args.eval_splits:
            st = master_summary["by_split"].get(sp, {})
            if st:
                f.write(f"| **{sp.upper()}** | {st['count']} | {st['linear_probe_auc_mean']:.4f} ± {st['linear_probe_auc_std']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['mahalanobis_auc_mean']:.4f} | {st['segmentation_iou_mean']:.4f} | {st['sizing_r2_mean']:.4f} |\n")

        f.write("\n## 2. Generalization Disaggregation\n\n")
        f.write("### By Waveform\n")
        f.write("| Waveform | Files | LP AUC | LP AP | CNR | Seg IoU |\n| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for wf, st in master_summary["by_waveform"].items():
            if st.get("count", 0) > 0:
                f.write(f"| {wf} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['segmentation_iou_mean']:.4f} |\n")

        f.write("\n### By Sensor\n")
        f.write("| Sensor | Files | LP AUC | LP AP | CNR | Seg IoU |\n| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for sn, st in master_summary["by_sensor"].items():
            if st.get("count", 0) > 0:
                f.write(f"| {sn} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['segmentation_iou_mean']:.4f} |\n")

        f.write("\n### By Lift-off\n")
        f.write("| Lift-off | Files | LP AUC | LP AP | CNR | Seg IoU |\n| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for lo, st in master_summary["by_liftoff"].items():
            if st.get("count", 0) > 0:
                f.write(f"| {lo} | {st['count']} | {st['linear_probe_auc_mean']:.4f} | {st['linear_probe_ap_mean']:.4f} | {st['cnr_mean']:.2f} | {st['segmentation_iou_mean']:.4f} |\n")

        # Section 3: Canonical UDA Benchmark
        if canonical_uda_results:
            f.write("\n## 3. Canonical Domain Adaptation (UDA) Benchmark (Controlled Single-Variable Shifts)\n\n")
            f.write("| Shift | Model | Source AUC (Retention) | Target AUC (Adaptation) | Target AP | Δ AUC vs Zero-Shot LP | Adaptation Strategy |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
            uda_meta = {
                "CORAL": "Feature Covariance Alignment",
                "DANN": "Adversarial Domain Invariance (GRL)",
                "DAN": "Multi-Kernel Maximum Mean Discrepancy",
                "CDAN": "Multilinear Conditioning Adversarial",
                "SHOT": "Information Maximization (Source-Free)",
                "OTDA": "Wasserstein Sinkhorn Optimal Transport",
                "CBST": "Class-Balanced Self-Training",
            }
            for s_name, s_data in canonical_uda_results.items():
                m_dict = s_data.get("models", {})
                if not m_dict:
                    continue
                first_m = list(m_dict.keys())[0]
                lp_src = m_dict[first_m]["source"]["lp_auc_mean"]
                lp_tgt = m_dict[first_m]["target"]["lp_auc_mean"]
                f.write(f"| **{s_name.upper()}** | **Zero-Shot Linear Probe (Baseline)** | {lp_src:.4f} | {lp_tgt:.4f} | - | *baseline* | Source-only frozen readout |\n")
                for m_name, m_res in m_dict.items():
                    s_auc = m_res["source"]["auc_mean"]
                    t_auc = m_res["target"]["auc_mean"]
                    t_std = m_res["target"]["auc_std"]
                    t_ap = m_res["target"]["ap_mean"]
                    delta = m_res["target"]["delta_auc_mean"]
                    sign = "+" if delta >= 0 else ""
                    strat = uda_meta.get(m_name, "Domain Adaptation")
                    f.write(f"| **{s_name.upper()}** | {m_name} | {s_auc:.4f} | {t_auc:.4f} ± {t_std:.4f} | {t_ap:.4f} | **{sign}{delta:.4f}** | {strat} |\n")

    # 9. Terminal Output
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    for sp in args.eval_splits:
        st = master_summary["by_split"].get(sp, {})
        if st:
            print(f"[{sp.upper():5s}] ({st['count']} files) -> LP AUC: {st['linear_probe_auc_mean']:.4f} ± {st['linear_probe_auc_std']:.4f} | LP AP: {st['linear_probe_ap_mean']:.4f} | CNR: {st['cnr_mean']:.2f} | Seg IoU: {st['segmentation_iou_mean']:.4f} | Sizing R²: {st['sizing_r2_mean']:.4f}")
    print("-" * 80)
    ov = master_summary["overall"]
    print(f"[OVERALL] ({ov['count']} files) -> LP AUC: {ov['linear_probe_auc_mean']:.4f} ± {ov['linear_probe_auc_std']:.4f} | LP AP: {ov['linear_probe_ap_mean']:.4f} | CNR: {ov['cnr_mean']:.2f} | Seg IoU: {ov['segmentation_iou_mean']:.4f}")

    # Canonical UDA Summary in Console
    if canonical_uda_results:
        print("\n" + "=" * 80)
        print("CANONICAL DOMAIN ADAPTATION (UDA) BENCHMARK SUMMARY")
        print("=" * 80)
        for s_name, s_data in canonical_uda_results.items():
            m_dict = s_data.get("models", {})
            if not m_dict:
                continue
            first_m = list(m_dict.keys())[0]
            lp_src = m_dict[first_m]["source"]["lp_auc_mean"]
            lp_tgt = m_dict[first_m]["target"]["lp_auc_mean"]
            print(f"\n--- Shift: [{s_name.upper()}] ---")
            print(f"{'Model / Method':<24} | {'Source AUC':<12} | {'Target AUC':<16} | {'Target AP':<10} | {'Δ vs LP AUC':<12}")
            print("-" * 80)
            print(f"{'Linear Probe (Zero-Shot)':<24} | {lp_src:<12.4f} | {lp_tgt:<16.4f} | {'-':<10} | {'baseline':<12}")
            for m_name, m_res in m_dict.items():
                s_auc = m_res["source"]["auc_mean"]
                t_auc = m_res["target"]["auc_mean"]
                t_std = m_res["target"]["auc_std"]
                t_ap = m_res["target"]["ap_mean"]
                delta = m_res["target"]["delta_auc_mean"]
                sign = "+" if delta >= 0 else ""
                auc_str = f"{t_auc:.4f} ± {t_std:.4f}"
                print(f"{m_name:<24} | {s_auc:<12.4f} | {auc_str:<16} | {t_ap:<10.4f} | {sign}{delta:<12.4f}")
        print("=" * 80)
    elif not run_uda:
        print("\n" + "-" * 80)
        print("[INFO] Canonical Domain Adaptation benchmark was SKIPPED by option (--skip_domain_adaptation / --eval_domain_adaptation=False).")
        print("-" * 80)
    else:
        print("\n" + "!" * 80)
        print("[NOTICE] No pre-trained UDA models were evaluated.")
        print("To train and evaluate UDA models (CORAL, DANN, DAN, CDAN, SHOT, OTDA, CBST):")
        print("  1. Run downstream training first:")
        print(f"     python -m src.PECT_JEPA.cross_condition.downstream.train_downstream --checkpoint {ckpt_path} --data_dir {args.data_dir}")
        print("  2. Re-run evaluate:")
        print(f"     python -m src.PECT_JEPA.cross_condition.evaluate --checkpoint {ckpt_path} --data_dir {args.data_dir}")
        print("!" * 80)

    for sp in args.eval_splits:
        n_flag = sum(1 for r in master_results if r["split"] == sp and r.get("regime_ood", 0) == 1)
        n_sp = sum(1 for r in master_results if r["split"] == sp)
        print(f"[REGIME] {sp.upper():5s}: {n_flag}/{n_sp} files flagged OOD-regime (E||z||^2 outside 2x train median {ref_energy:.1f})")
    print(f"\nAll results saved to: {out_dir}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
