"""
CLI Evaluation and Downstream Probing Suite for 5x5 Spatiotemporal PECT-JEPA.

Modular Downstream Benchmark Architecture supporting Dual Evaluation Modes:
- Mode A: Within_File/ (In-scan cross-validated / single-scan fitted readouts)
- Mode B: Zero_Shot/ (Frozen global readouts trained strictly on the train set pool, evaluated zero-shot across unseen scans)

Partition Hierarchy:
  <output_dir>/
    Within_File/
      test/
        1_Anomaly_Detection/<specimen>/
        2_Depth_Regression/<specimen>/
        2b_Size_Regression/<specimen>/
        3D_Tomography/<specimen>/
        Latent_Diagnostics/<specimen>/
        evaluation_summary.json
        evaluation_summary.csv
      val/
        ...
      train/
        ...
    Zero_Shot/
      test/
        1_Anomaly_Detection/<specimen>/
        2_Depth_Regression/<specimen>/
        2b_Size_Regression/<specimen>/
        3D_Tomography/<specimen>/
        Latent_Diagnostics/<specimen>/
        evaluation_summary.json
        evaluation_summary.csv
      val/
        ...
      train/
        ...
    master_benchmark_summary.json
    master_benchmark_summary.csv
    evaluation_summary.json (backward-compatible copy of Within_File/test)
    evaluation_summary.csv (backward-compatible copy of Within_File/test)

Usage:
    # Evaluate all partitions (test, val, train) across Within-File and Zero-Shot:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --exp_name exp37b_unified_4stage_harmonized \\
        --eval_splits test val train

    # Evaluate single TDMS file:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --exp_name exp37b_unified_4stage_harmonized \\
        --file data/TMR/Corrosion/Square/tmr_corosion_frontside_square_300x300_z1_20260126_190655.tdms
"""

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import types
from typing import List, Dict, Any, Optional, Tuple

# Force unbuffered streaming output so background logs update immediately
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

from tqdm import tqdm
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
    precision_recall_curve,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.cscan_extractor import (
    extract_full_cscan_map,
    load_cscan_from_tdms,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.anomaly_detection import (
    compute_anomaly_metrics,
    MahalanobisDetector,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.linear_probe import LinearProbeEvaluator
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.visualizations import (
    plot_probability_heatmap,
    plot_roc_pr_curves,
    plot_depth_regression_maps,
    plot_depth_calibration_scatter,
    plot_defect_contours_and_iou,
    plot_flaw_size_maps,
    plot_flaw_size_calibration_scatter,
    apply_spatial_coherence_filter,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.morphology_classifier import (
    DefectMorphologyClassifier,
    build_morphology_labels_from_scan,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.latent_diagnostics import (
    compute_channel_sensitivity_spectrum,
    compute_morphology_subspace_separation,
)


def to_safe_path(path: str) -> str:
    """Ensures paths on Windows bypass the MAX_PATH (260 char) limitation using extended prefix only if needed."""
    if not path:
        return path
    abs_path = os.path.abspath(path)
    if os.name == "nt" and len(abs_path) >= 240 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def find_ground_truth_mask(file_path: str, data_dir: str = "data") -> Optional[np.ndarray]:
    """Finds and loads authoritative CAD ground-truth mask corresponding to TDMS file."""
    try:
        gt_mgr = get_ground_truth_manager(data_dir=data_dir)
        return gt_mgr.get_ground_truth_mask_for_file(file_path, aligned_scan=True)
    except Exception:
        pass

    fname_lower = os.path.basename(file_path).lower()
    specimen_key = None
    if "corosion" in fname_lower or "corrosion" in fname_lower:
        specimen_key = "corrosion"
    elif "rivet_v1" in fname_lower or "rivet1" in fname_lower:
        specimen_key = "rivet_v1"
    elif "rivet_v2" in fname_lower or "rivet2" in fname_lower or "mixed" in fname_lower:
        specimen_key = "rivet_v2"

    if specimen_key:
        candidates = [
            os.path.join(data_dir, "ground_truth", specimen_key, f"{specimen_key}_gt_mask.npy"),
            os.path.join(ROOT_DIR, "data", "ground_truth", specimen_key, f"{specimen_key}_gt_mask.npy"),
            os.path.join("data", "ground_truth", specimen_key, f"{specimen_key}_gt_mask.npy"),
        ]
        for c_gt in candidates:
            if os.path.isfile(c_gt):
                try:
                    return np.load(c_gt)
                except Exception:
                    pass
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("5x5 Spatiotemporal PECT-JEPA Downstream Evaluation")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (.pt), or keyword ('best', 'latest', 'auto')")
    p.add_argument("--exp_name", type=str, default=None,
                   help="Experiment name or prefix to evaluate (auto-discovers latest run in experiments/5x5/)")
    p.add_argument("--file", type=str, default=None, help="Path to single TDMS file for defect mapping")
    p.add_argument("--data_dir", type=str, default="data", help="Data directory containing TDMS files")
    p.add_argument("--split_summary", type=str, default=None,
                   help="Path to split summary JSON produced during training (loads held-out test files)")
    p.add_argument("--split_protocol", type=str, default="compound_ood",
                   choices=["compound_ood", "leave_liftoff", "leave_sensor", "leave_waveform", "leave_specimen", "random"],
                   help="Evaluation split protocol")
    p.add_argument("--holdout_target", type=str, default="z3", help="Holdout target category for single-factor protocols")
    p.add_argument("--holdout_liftoff", type=str, default="z3", help="Lift-off level held out for compound_ood")
    p.add_argument("--holdout_sensor", type=str, default="TMR", help="Sensor hardware held out for compound_ood")
    p.add_argument("--holdout_waveform", type=str, default="Chirp", help="Waveform shape held out for compound_ood")
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation artifacts")
    p.add_argument("--eval_splits", nargs="+", default=["test", "val", "train"],
                   help="Dataset splits to evaluate: 'test', 'val', 'train', or combinations")
    p.add_argument("--eval_modes", nargs="+", default=["Within_File", "Zero_Shot"],
                   help="Evaluation modes: 'Within_File', 'Zero_Shot', or both")
    p.add_argument("--save_features", action="store_true", default=False, help="Save extracted .npy feature maps to disk")
    p.add_argument("--crop_border", type=int, default=15, help="Outer boundary pixels cropped to remove air/edge effect")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for sliding window feature extraction")
    p.add_argument("--device", type=str, default="cuda", help="Target device: 'cuda' or 'cpu'")
    p.add_argument("--max_eval_files", type=int, default=None, help="Optional limit on number of test files to evaluate")
    p.add_argument("--eval_3d", type=lambda v: v.lower() == "true", default=True, help="Extract 3D volumetric slices and topological defect graphs (default: True)")
    p.add_argument("--eval_diagnostics", type=lambda v: v.lower() == "true", default=True, help="Run high-dimensional latent forensics suite (default: True)")
    return p


def resolve_checkpoint_path(checkpoint_path: Optional[str] = None, exp_name: Optional[str] = None) -> str:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    search_base = "experiments/5x5"

    if checkpoint_path:
        base = os.path.basename(checkpoint_path)
        if base.endswith(".pt") and os.path.isfile(checkpoint_path):
            return checkpoint_path
        if os.path.isdir(checkpoint_path):
            candidates = [
                os.path.join(checkpoint_path, "checkpoints", "best_model_5x5.pt"),
                os.path.join(checkpoint_path, "checkpoints", "latest_model_5x5.pt"),
                os.path.join(checkpoint_path, "best_model_5x5.pt"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    return c

    if exp_name:
        if os.path.isdir(search_base):
            runs = [os.path.join(search_base, d) for d in os.listdir(search_base) if d.startswith(exp_name)]
            runs = [r for r in runs if os.path.isdir(r)]
            runs.sort(key=os.path.getmtime, reverse=True)
            for r in runs:
                cand = os.path.join(r, "checkpoints", "best_model_5x5.pt")
                if os.path.isfile(cand):
                    return cand
                cand2 = os.path.join(r, "checkpoints", "latest_model_5x5.pt")
                if os.path.isfile(cand2):
                    return cand2

    if os.path.isdir(search_base):
        all_runs = [os.path.join(search_base, d) for d in os.listdir(search_base) if os.path.isdir(os.path.join(search_base, d))]
        all_runs.sort(key=os.path.getmtime, reverse=True)
        for r in all_runs:
            cand = os.path.join(r, "checkpoints", "best_model_5x5.pt")
            if os.path.isfile(cand):
                return cand
            cand2 = os.path.join(r, "checkpoints", "latest_model_5x5.pt")
            if os.path.isfile(cand2):
                return cand2

    fallbacks = [
        "checkpoints/pect_jepa_5x5/best_model_5x5.pt",
        "checkpoints/pect_jepa_5x5/latest_model_5x5.pt",
    ]
    for fb in fallbacks:
        if os.path.isfile(fb):
            return fb
    return checkpoint_path or "checkpoints/pect_jepa_5x5/best_model_5x5.pt"


def resolve_split_summary_path(split_summary_path: Optional[str] = None, checkpoint_path: Optional[str] = None) -> Optional[str]:
    if split_summary_path and os.path.isfile(split_summary_path):
        return split_summary_path
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt_dir = os.path.dirname(checkpoint_path)
        exp_dir = os.path.dirname(ckpt_dir)
        candidates = [
            os.path.join(ckpt_dir, "pect_jepa_split_summary.json"),
            os.path.join(exp_dir, "pect_jepa_split_summary.json"),
        ]
        if os.path.isdir(ckpt_dir):
            for fname in os.listdir(ckpt_dir):
                if fname.endswith("_split_summary.json"):
                    candidates.append(os.path.join(ckpt_dir, fname))
        if os.path.isdir(exp_dir):
            for fname in os.listdir(exp_dir):
                if fname.endswith("_split_summary.json"):
                    candidates.append(os.path.join(exp_dir, fname))
        for c in candidates:
            if os.path.isfile(c):
                return c
    return split_summary_path


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> PECT_JEPA_5x5:
    resolved = resolve_checkpoint_path(checkpoint_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint_path = resolved
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg_dict = ckpt.get("config", {})

    config = None
    if isinstance(cfg_dict, dict) and cfg_dict:
        config = Spatiotemporal5x5Config.from_dict(cfg_dict)
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        exp_dir = os.path.dirname(ckpt_dir)
        for cand in [
            os.path.join(ckpt_dir, "config_5x5.json"),
            os.path.join(exp_dir, "config_5x5.json"),
            os.path.join(ckpt_dir, "config.json"),
            os.path.join(exp_dir, "config.json"),
        ]:
            if os.path.isfile(cand):
                try:
                    config = Spatiotemporal5x5Config.from_json(cand)
                    break
                except Exception:
                    pass

    if config is None:
        config = Spatiotemporal5x5Config()

    state_dict = ckpt.get("model_state_dict", ckpt)

    # Inferences for backward compatibility
    if not (isinstance(cfg_dict, dict) and "tokenizer_type" in cfg_dict):
        if "tokenizer.time_proj.0.weight" in state_dict:
            if state_dict["tokenizer.time_proj.0.weight"].shape[1] == (config.in_channels + 3):
                config.tokenizer_type = "energy_adaptive_dual_domain"
            else:
                config.tokenizer_type = "continuous_linear_field"
        elif "tokenizer.scale_shallow" in state_dict:
            config.tokenizer_type = "uncrushed_diffusion"
        elif "tokenizer.pos_scale" in state_dict or "tokenizer.gate_proj.0.weight" in state_dict:
            config.tokenizer_type = "spatio_spectral"
        elif "tokenizer.chunk_proj.weight" in state_dict:
            config.tokenizer_type = "spatiotemporal_patch"
        elif "tokenizer.conv_short.weight" in state_dict:
            config.tokenizer_type = "continuous_stf"
        elif "tokenizer.cross_domain_attn.in_proj_weight" in state_dict or "tokenizer.fuse_proj.weight" in state_dict:
            config.tokenizer_type = "dual_domain_attention"
        elif "tokenizer.time_proj.weight" in state_dict:
            config.tokenizer_type = "dual_domain"
        elif "tokenizer.proj.weight" in state_dict:
            config.tokenizer_type = "time_only"

    if not (isinstance(cfg_dict, dict) and "predictor_type" in cfg_dict):
        if "predictor.rel_diff_mlp.0.weight" in state_dict:
            config.predictor_type = "freq_conditioned_diffusion"
        elif "predictor.raw_alpha_x" in state_dict or "predictor.raw_alpha_y" in state_dict:
            config.predictor_type = "anisotropic_diffusion"
        elif "predictor.raw_d_scale" in state_dict:
            config.predictor_type = "continuous_helmholtz"
        elif "predictor.gamma_raw" in state_dict:
            config.predictor_type = "residual_diffusion"
        elif "predictor.op_embedding.default_op" in state_dict:
            config.predictor_type = "operator_diffusion"
        else:
            config.predictor_type = "standard"

    if not (isinstance(cfg_dict, dict) and "encoder_type" in cfg_dict):
        if "context_encoder.final_ada_ln.1.weight" in state_dict or any("ada_ln" in k for k in state_dict.keys()):
            config.encoder_type = "dispersion_conditioned"
        else:
            config.encoder_type = "standard"

    if "encoder.pos_embed" in state_dict:
        actual_dim = state_dict["encoder.pos_embed"].shape[-1]
        if config.embed_dim != actual_dim:
            config.embed_dim = actual_dim

    model = PECT_JEPA_5x5(config)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    epoch_info = ckpt.get("epoch", "?")
    step_info = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint from {checkpoint_path} (epoch: {epoch_info}, step: {step_info})")
    print(f"  Model config: tokenizer={config.tokenizer_type}, encoder={getattr(config, 'encoder_type', 'standard')}, predictor={config.predictor_type}, embed_dim={config.embed_dim}")
    return model


def fmt_val(v: Optional[float], decimals: int = 4) -> str:
    """Safely formats floating point metrics for tabular display."""
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def get_or_extract_features(
    file_path: str,
    model: PECT_JEPA_5x5,
    cache_dir: Optional[str] = None,
    batch_size: int = 512,
    device: str = "cuda",
    crop_border: int = 15,
    eval_3d: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Retrieves feature map and 3D volume from persistent disk cache or extracts from TDMS once."""
    if cache_dir:
        fname = os.path.splitext(os.path.basename(file_path))[0]
        fmap_path = to_safe_path(os.path.join(cache_dir, f"{fname}_fmap.npy"))
        vol_path = to_safe_path(os.path.join(cache_dir, f"{fname}_vol.npy"))
        if os.path.isfile(fmap_path):
            try:
                fmap = np.load(fmap_path)
                vol = np.load(vol_path) if eval_3d and os.path.isfile(vol_path) else None
                return fmap, vol
            except Exception:
                pass

    grid_3d = load_cscan_from_tdms(
        file_path,
        time_samples=model.config.time_samples,
        temporal_samples=model.config.temporal_samples,
        resample_mode=model.config.resample_mode,
        normalization=model.config.normalization,
        raster_correction=model.config.raster_correction,
        crop_border=crop_border,
        apply_lowpass=getattr(model.config, "apply_lowpass", True),
        lowpass_cutoff=getattr(model.config, "lowpass_cutoff", 2500.0),
        lowpass_order=getattr(model.config, "lowpass_order", 4),
    )
    if eval_3d:
        fmap, vol = extract_full_cscan_map(
            model=model, full_cscan_3d=grid_3d, batch_size=batch_size, device=device, show_pbar=False, return_volume_3d=True
        )
    else:
        fmap = extract_full_cscan_map(
            model=model, full_cscan_3d=grid_3d, batch_size=batch_size, device=device, show_pbar=False, return_volume_3d=False
        )
        vol = None

    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(fmap_path, fmap)
            if vol is not None:
                np.save(vol_path, vol)
        except Exception:
            pass

    return fmap, vol


def train_base_zero_shot_readouts(
    model: PECT_JEPA_5x5,
    train_files: List[str],
    data_dir: str = "data",
    cache_dir: Optional[str] = None,
    batch_size: int = 512,
    device: str = "cuda",
    crop_border: int = 15,
    eval_3d: bool = True,
    pts_per_train_file: int = 2000,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]]:
    """
    Extracts features across the base training files (train_files) and fits global readouts:
      1. Detection Probe: StandardScaler + LogisticRegression + MLPClassifier (balanced)
      2. Quantitative Depth Probe: StandardScaler + Ridge (alpha=1.0) on surface defects (d > 0.0)
      3. Flaw Size Probe: StandardScaler + Ridge (alpha=1.0) on defect clusters (s > 0.0)
      4. Morphology Classifier: DefectMorphologyClassifier
    Caches extracted features for train files in memory so evaluating the train split later takes 0 GPU time.
    """
    gt_mgr = get_ground_truth_manager(data_dir=data_dir)
    cached_train_features: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    det_feats_list, det_labels_list = [], []
    depth_feats_list, depth_labels_list = [], []
    size_feats_list, size_labels_list = [], []
    morph_feats_list, morph_labels_list = [], []

    print(f"\n--- Extracting Base Feature Pool from {len(train_files)} Train Files ---", flush=True)
    for fp in tqdm(train_files, desc="  Fitting Base Readouts", file=sys.stdout):
        mask = gt_mgr.get_ground_truth_mask_for_file(fp, aligned_scan=True)
        if mask is None:
            continue
        sp_key = gt_mgr.canonical_specimen_key(fp)
        depth_gt = gt_mgr.generate_depth_map(sp_key)
        size_gt = gt_mgr.generate_size_map(sp_key, mode="diameter")

        try:
            fmap, vol3d = get_or_extract_features(
                file_path=fp,
                model=model,
                cache_dir=cache_dir,
                batch_size=batch_size,
                device=device,
                crop_border=crop_border,
                eval_3d=eval_3d,
            )
            cached_train_features[fp] = (fmap, vol3d)

            min_Y = min(fmap.shape[0], mask.shape[0])
            min_X = min(fmap.shape[1], mask.shape[1])
            sub_f = fmap[:min_Y, :min_X].reshape(-1, fmap.shape[-1]).astype(np.float32)
            sub_y = mask[:min_Y, :min_X].reshape(-1).astype(np.int64)

            # 1. Binary Detection Pool
            v_idx = np.where(sub_y >= 0)[0]
            if len(v_idx) > 0:
                f_v, y_v = sub_f[v_idx], sub_y[v_idx]
                pos = np.where(y_v == 1)[0]
                neg = np.where(y_v == 0)[0]
                if len(pos) > 0 and len(neg) > 0:
                    n_neg = min(len(neg), max(len(pos) * 5, pts_per_train_file))
                    rng = np.random.RandomState(42)
                    sub_neg = rng.choice(neg, size=n_neg, replace=False)
                    keep = np.concatenate([pos, sub_neg])
                    det_feats_list.append(f_v[keep])
                    det_labels_list.append(y_v[keep])

            # 2. Depth Regression Pool (Surface-breaking flaws)
            if depth_gt is not None:
                sub_d = depth_gt[:min_Y, :min_X].reshape(-1).astype(np.float32)
                def_d = np.where(sub_d > 0.0)[0]
                if len(def_d) > 0:
                    depth_feats_list.append(sub_f[def_d])
                    depth_labels_list.append(sub_d[def_d])

            # 3. Flaw Size Regression Pool (Defect clusters)
            if size_gt is not None:
                sub_s = size_gt[:min_Y, :min_X].reshape(-1).astype(np.float32)
                def_s = np.where(sub_s > 0.0)[0]
                if len(def_s) > 0:
                    size_feats_list.append(sub_f[def_s])
                    size_labels_list.append(sub_s[def_s])

            # 4. Morphology Classification Pool
            sub_m = build_morphology_labels_from_scan(
                sp_key,
                mask[:min_Y, :min_X],
                depth_gt[:min_Y, :min_X] if depth_gt is not None else None,
            ).reshape(-1)
            classes_in_scan = np.unique(sub_m)
            rng_m = np.random.RandomState(42)
            keep_m = []
            for c in classes_in_scan:
                c_idx = np.where(sub_m == c)[0]
                if len(c_idx) > 1500:
                    c_idx = rng_m.choice(c_idx, size=1500, replace=False)
                keep_m.append(c_idx)
            if keep_m:
                sel_m = np.concatenate(keep_m)
                morph_feats_list.append(sub_f[sel_m])
                morph_labels_list.append(sub_m[sel_m])

        except Exception as e:
            print(f"  [Train Pool Warning] Skipping {os.path.basename(fp)}: {e}", flush=True)

    base_readouts: Dict[str, Any] = {}

    # Fit Detection Probe
    if det_feats_list:
        X_det = np.concatenate(det_feats_list, axis=0)
        y_det = np.concatenate(det_labels_list, axis=0)
        print(f"  Fitting Base Detection Probe on {len(y_det)} points ({np.sum(y_det == 1)} defect, {np.sum(y_det == 0)} sound)...", flush=True)
        scaler_det = StandardScaler()
        X_det_s = scaler_det.fit_transform(X_det)
        lr_probe = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42, solver="lbfgs")
        lr_probe.fit(X_det_s, y_det)

        mlp_probe = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=150, early_stopping=True, random_state=42)
        sample_weights = compute_sample_weight("balanced", y_det)
        mlp_probe.fit(X_det_s, y_det, sample_weight=sample_weights)

        base_readouts["scaler_det"] = scaler_det
        base_readouts["lr_probe"] = lr_probe
        base_readouts["mlp_probe"] = mlp_probe

    # Fit Depth Regression Probe
    if depth_feats_list:
        X_depth = np.concatenate(depth_feats_list, axis=0)
        y_depth = np.concatenate(depth_labels_list, axis=0)
        print(f"  Fitting Base Depth Probe on {len(y_depth)} surface defect points...", flush=True)
        scaler_depth = StandardScaler()
        X_depth_s = scaler_depth.fit_transform(X_depth)
        ridge_depth = Ridge(alpha=1.0, random_state=42)
        ridge_depth.fit(X_depth_s, y_depth)
        base_readouts["scaler_depth"] = scaler_depth
        base_readouts["ridge_depth"] = ridge_depth

    # Fit Flaw Size Regression Probe
    if size_feats_list:
        X_size = np.concatenate(size_feats_list, axis=0)
        y_size = np.concatenate(size_labels_list, axis=0)
        print(f"  Fitting Base Size Probe on {len(y_size)} flaw cluster points...", flush=True)
        scaler_size = StandardScaler()
        X_size_s = scaler_size.fit_transform(X_size)
        ridge_size = Ridge(alpha=1.0, random_state=42)
        ridge_size.fit(X_size_s, y_size)
        base_readouts["scaler_size"] = scaler_size
        base_readouts["ridge_size"] = ridge_size

    # Fit Morphology Classifier
    if morph_feats_list:
        X_morph = np.concatenate(morph_feats_list, axis=0)
        y_morph = np.concatenate(morph_labels_list, axis=0)
        if len(np.unique(y_morph)) >= 2:
            print(f"  Fitting Base Morphology Classifier on {len(y_morph)} samples across classes {np.unique(y_morph).tolist()}...", flush=True)
            morph_clf = DefectMorphologyClassifier(classifier_type="linear", random_state=42)
            morph_clf.fit(X_morph, y_morph)
            base_readouts["morphology_clf"] = morph_clf

    print("  Global Base Readouts successfully initialized!\n", flush=True)
    return base_readouts, cached_train_features


def evaluate_single_file_within_file(
    file_path: str,
    feature_map: np.ndarray,
    volume_3d: Optional[np.ndarray],
    output_dir: str,
    model_config: Any,
    save_features: bool = False,
    crop_border: int = 15,
    eval_3d: bool = True,
    eval_diagnostics: bool = True,
    morphology_clf: Optional[DefectMorphologyClassifier] = None,
) -> Dict[str, Any]:
    """
    Mode A: Within_File Evaluation.
    Fits in-scan cross-validated readouts (Linear Probe CV, In-Scan Hurdle) on this specific C-scan.
    Saves artifacts to <output_dir>/<Task>/<specimen>/
    """
    fname_base = os.path.splitext(os.path.basename(file_path))[0]
    meta = extract_file_metadata(file_path)
    gt_mgr = get_ground_truth_manager(data_dir=getattr(model_config, "data_dir", "data"))
    specimen_key = gt_mgr.canonical_specimen_key(file_path)

    task1_dir = to_safe_path(os.path.join(output_dir, "1_Anomaly_Detection", specimen_key))
    task2_dir = to_safe_path(os.path.join(output_dir, "2_Depth_Regression", specimen_key))
    task2b_dir = to_safe_path(os.path.join(output_dir, "2b_Size_Regression", specimen_key))
    task3d_dir = to_safe_path(os.path.join(output_dir, "3D_Tomography", specimen_key))
    task_diag_dir = to_safe_path(os.path.join(output_dir, "Latent_Diagnostics", specimen_key))

    for d in [task1_dir, task2_dir, task2b_dir]:
        os.makedirs(d, exist_ok=True)
    if eval_3d:
        os.makedirs(task3d_dir, exist_ok=True)
    if eval_diagnostics:
        os.makedirs(task_diag_dir, exist_ok=True)

    gt_mask = gt_mgr.get_ground_truth_mask_for_file(file_path, aligned_scan=True)
    depth_map_gt = gt_mgr.generate_depth_map(specimen_key)
    size_map_gt = gt_mgr.generate_size_map(specimen_key, mode="diameter")

    has_gt = gt_mask is not None
    min_Y = min(feature_map.shape[0], gt_mask.shape[0]) if has_gt else feature_map.shape[0]
    min_X = min(feature_map.shape[1], gt_mask.shape[1]) if has_gt else feature_map.shape[1]

    sub_feat = feature_map[:min_Y, :min_X]
    sub_gt = gt_mask[:min_Y, :min_X] if has_gt else None
    sub_depth = depth_map_gt[:min_Y, :min_X] if depth_map_gt is not None else None
    sub_size = size_map_gt[:min_Y, :min_X] if size_map_gt is not None else None

    # Task 1: Anomaly Detection (Linear Probe CV)
    task1_res: Dict[str, Any] = {}
    prob_map = None
    if has_gt and sub_gt is not None:
        try:
            evaluator = LinearProbeEvaluator(n_splits=5)
            lp_res, prob_map = evaluator.fit_and_predict_probability_map(sub_feat, sub_gt)
            cnr_res = compute_anomaly_metrics(prob_map, gt_mask=sub_gt)
            cnr = float(cnr_res.get("contrast_ratio_cnr", 0.0))
            peak_cnr = float(cnr_res.get("peak_contrast_ratio", 0.0))

            lp_auc = lp_res.get("linear_probe_auc_roc")
            lp_ap = lp_res.get("linear_probe_average_precision")
            lp_f1 = lp_res.get("linear_probe_f1")
            mlp_auc = lp_res.get("mlp_2layer_auc_roc")
            mlp_ap = lp_res.get("mlp_2layer_average_precision")
            mlp_f1 = lp_res.get("mlp_2layer_f1")
            delta_auc = lp_res.get("representation_gap_delta_auc")

            task1_res = {
                "linear_probe": {
                    "auc_roc": lp_auc,
                    "average_precision": lp_ap,
                    "f1_score": lp_f1,
                    "accuracy": lp_res.get("linear_probe_accuracy"),
                },
                "mlp_2layer": {
                    "auc_roc": mlp_auc,
                    "average_precision": mlp_ap,
                    "f1_score": mlp_f1,
                    "accuracy": lp_res.get("mlp_2layer_accuracy"),
                },
                "representation_gap": {
                    "delta_auc_roc": delta_auc,
                    "delta_average_precision": lp_res.get("representation_gap_delta_ap"),
                },
                "contrast_ratio_cnr": cnr,
                "peak_contrast_ratio": peak_cnr,
                "knn_5_accuracy": lp_res.get("knn_5_accuracy"),
            }

            # Unsupervised Mahalanobis
            try:
                maha = MahalanobisDetector(regularize_eps=1e-4)
                maha.fit(sub_feat, contamination=0.05)
                maha_score = maha.score_map(sub_feat)
                valid_m = (sub_gt.reshape(-1) >= 0)
                y_m = sub_gt.reshape(-1)[valid_m]
                s_m = maha_score.reshape(-1)[valid_m]
                if len(np.unique(y_m)) > 1:
                    maha_auc = float(roc_auc_score(y_m, s_m))
                    maha_ap = float(average_precision_score(y_m, s_m))
                    def_pts = maha_score[sub_gt == 1]
                    snd_pts = maha_score[sub_gt == 0]
                    maha_cnr = float((np.mean(def_pts) - np.mean(snd_pts)) / (np.std(snd_pts) + 1e-8)) if len(def_pts) > 0 and len(snd_pts) > 0 else 0.0
                else:
                    maha_auc, maha_ap, maha_cnr = None, None, None
            except Exception:
                maha_auc, maha_ap, maha_cnr = None, None, None

            task1_res["unsupervised_mahalanobis"] = {
                "auc_roc": maha_auc,
                "average_precision": maha_ap,
                "contrast_ratio_cnr": maha_cnr,
            }

            # Plots
            prob_heatmap_path = os.path.join(task1_dir, f"{fname_base}_prob_heatmap.png")
            plot_probability_heatmap(
                prob_map=prob_map,
                save_path=prob_heatmap_path,
                title=f"Defect Probability Map (Within-File) | {meta.get('specimen', '')} - {meta.get('sensor', '')}\n"
                      f"Lift-off: {meta.get('liftoff', '')} | Waveform: {meta.get('waveform', '')} | AUC: {lp_auc:.4f} | CNR: {cnr:.2f}",
            )
            task1_res["prob_heatmap_path"] = prob_heatmap_path

            curves = lp_res.get("curve_data", {})
            if curves and "fpr" in curves:
                roc_pr_path = os.path.join(task1_dir, f"{fname_base}_roc_pr_curve.png")
                plot_roc_pr_curves(
                    fpr=curves["fpr"],
                    tpr=curves["tpr"],
                    auc_roc=lp_auc if lp_auc is not None else 0.5,
                    precision=curves["precision"],
                    recall=curves["recall"],
                    avg_prec=lp_ap if lp_ap is not None else 0.0,
                    save_path=roc_pr_path,
                    title=f"ROC & PR Curves | {fname_base}",
                )
                task1_res["roc_pr_path"] = roc_pr_path

            contour_path = os.path.join(task1_dir, f"{fname_base}_defect_contours_iou.png")
            iou_dict = plot_defect_contours_and_iou(
                prob_map=prob_map,
                gt_mask=sub_gt,
                save_path=contour_path,
                title=f"Defect Contours & IoU | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
            )
            task1_res["segmentation_iou"] = iou_dict
            task1_res["defect_contours_path"] = contour_path

        except Exception as e:
            task1_res = {"error": str(e)}

    # Task 2: Quantitative Depth Regression (Surface Flaws)
    task2_res: Dict[str, Any] = {}
    morph_preds = None
    if morphology_clf is not None and getattr(morphology_clf, "is_fitted", False):
        try:
            morph_preds = morphology_clf.predict(sub_feat)
        except Exception:
            pass
    if morph_preds is None:
        morph_preds = build_morphology_labels_from_scan(specimen_key, sub_gt, sub_depth)

    surface_flaw_detected = bool(np.sum(morph_preds == 1) >= 10)
    has_depth_labels = (has_gt and sub_depth is not None and np.sum(sub_depth > 0.0) >= 5)

    if surface_flaw_detected and has_depth_labels:
        try:
            bench = DownstreamBenchmarkSuite(n_splits=5, random_state=42)
            reg_benchmark = bench.benchmark_depth_regression(sub_feat, sub_depth, focus_defects_only=False)
            try:
                reg_benchmark_def = bench.benchmark_depth_regression(sub_feat, sub_depth, focus_defects_only=True)
            except Exception:
                reg_benchmark_def = {}
            try:
                hurdle_benchmark = bench.benchmark_hurdle_depth_regression(sub_feat, sub_depth)
            except Exception as e:
                hurdle_benchmark = {"error": str(e)}

            flat_feats = sub_feat.reshape(-1, sub_feat.shape[-1]).astype(np.float32)
            flat_depth = sub_depth.reshape(-1).astype(np.float32)
            def_idx = np.where(flat_depth > 0.0)[0]
            snd_idx = np.where(flat_depth == 0.0)[0]
            if len(snd_idx) > 8000:
                rng = np.random.RandomState(42)
                sub_snd = rng.choice(snd_idx, size=8000, replace=False)
                fit_idx = np.concatenate([def_idx, sub_snd])
            else:
                fit_idx = np.arange(len(flat_depth))

            scaler = StandardScaler()
            X_fit_s = scaler.fit_transform(flat_feats[fit_idx])
            y_fit = flat_depth[fit_idx]

            clf_gate = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
            y_bin_fit = (y_fit > 0.0).astype(int)
            def_fit_mask = (y_fit > 0.0)

            if len(np.unique(y_bin_fit)) >= 2 and np.sum(def_fit_mask) >= 5:
                clf_gate.fit(X_fit_s, y_bin_fit)
                p_all = clf_gate.predict_proba(scaler.transform(flat_feats))[:, 1]
                ridge_cond = Ridge(alpha=1.0, random_state=42)
                ridge_cond.fit(X_fit_s[def_fit_mask], y_fit[def_fit_mask])
                d_cond_all = np.maximum(0.0, ridge_cond.predict(scaler.transform(flat_feats)))

                p_fit = clf_gate.predict_proba(X_fit_s)[:, 1]
                d_fit_pred = np.maximum(0.0, ridge_cond.predict(X_fit_s))
                best_tau, best_score = 0.5, -1e9
                for c_tau in np.linspace(0.3, 0.95, 27):
                    score = r2_score(y_fit, np.where(p_fit >= c_tau, d_fit_pred, 0.0))
                    if score > best_score:
                        best_score, best_tau = score, float(c_tau)

                p_gate_bin = (p_all >= best_tau).reshape(min_Y, min_X).astype(int)
                p_gate_clean = apply_spatial_coherence_filter(p_gate_bin, min_area=8)
                pred_depth_flat = np.where(p_gate_clean.reshape(-1) == 1, d_cond_all, 0.0)
            else:
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_fit_s, y_fit)
                pred_depth_flat = np.clip(ridge.predict(scaler.transform(flat_feats)), 0.0, None)

            pred_depth_map = pred_depth_flat.reshape(min_Y, min_X)
            lp_reg = reg_benchmark.get("linear_probe", {})
            mlp_reg = reg_benchmark.get("mlp_2layer", {})
            r2_val = lp_reg.get("r2_score", 0.0)
            mae_val = lp_reg.get("mae_mm", 0.0)
            rmse_val = lp_reg.get("rmse_mm", 0.0)

            hurdle_r2_val = hurdle_benchmark.get("compound_hurdle", {}).get("plate_r2_score") if isinstance(hurdle_benchmark, dict) else None
            defect_r2_val = hurdle_benchmark.get("conditional_defect_sizing", {}).get("r2_score") if isinstance(hurdle_benchmark, dict) else None

            pred_depth_map_path = os.path.join(task2_dir, f"{fname_base}_predicted_depth_map.png")
            depth_scatter_path = os.path.join(task2_dir, f"{fname_base}_depth_scatter.png")

            plot_depth_regression_maps(
                true_depth_map=sub_depth,
                pred_depth_map=pred_depth_map,
                save_path=pred_depth_map_path,
                title=f"Quantitative Depth Sizing (Within-File) | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
                r2=r2_val,
                mae=mae_val,
                rmse=rmse_val,
            )
            plot_depth_calibration_scatter(
                true_depth=flat_depth[fit_idx],
                pred_depth=pred_depth_flat[fit_idx],
                save_path=depth_scatter_path,
                title=f"Depth Calibration Scatter | {fname_base}",
                r2=r2_val,
                mae=mae_val,
                rmse=rmse_val,
            )

            task2_res = {
                "linear_probe": lp_reg,
                "mlp_2layer": mlp_reg,
                "representation_gap": reg_benchmark.get("representation_gap", {}),
                "defects_only": reg_benchmark_def.get("linear_probe", {}),
                "hurdle_depth_protocol": hurdle_benchmark,
                "hurdle_plate_r2": hurdle_r2_val,
                "defect_only_r2": defect_r2_val,
                "pred_depth_map_path": pred_depth_map_path,
                "depth_scatter_path": depth_scatter_path,
                "morphology_routing": "Surface-breaking defect (Lo thien) -> Quantitative Depth Regression executed",
            }
        except Exception as e:
            task2_res = {"error": str(e)}
    else:
        task2_res = {
            "notice": "Subsurface defect / fastener structure; routed to Task 2b Universal Flaw Sizing",
            "morphology_routing": "Subsurface defect / Fastener -> Task 2b Flaw Sizing",
            "linear_probe": {},
            "defects_only": {},
            "defect_only_r2": None,
            "hurdle_plate_r2": None,
        }

    # Task 2b: Universal Flaw Size Sizing (All 3 Plates)
    task2b_res: Dict[str, Any] = {}
    if has_gt and sub_size is not None and np.sum(sub_size > 0.0) >= 5:
        try:
            bench_size = DownstreamBenchmarkSuite(n_splits=5, random_state=42)
            hurdle_size_res = bench_size.benchmark_hurdle_depth_regression(sub_feat, sub_size)

            flat_feats = sub_feat.reshape(-1, sub_feat.shape[-1]).astype(np.float32)
            flat_size = sub_size.reshape(-1).astype(np.float32)
            def_idx = np.where(flat_size > 0.0)[0]
            snd_idx = np.where(flat_size == 0.0)[0]
            if len(snd_idx) > 8000:
                rng = np.random.RandomState(42)
                sub_snd = rng.choice(snd_idx, size=8000, replace=False)
                fit_idx = np.concatenate([def_idx, sub_snd])
            else:
                fit_idx = np.arange(len(flat_size))

            scaler = StandardScaler()
            X_fit_s = scaler.fit_transform(flat_feats[fit_idx])
            y_fit = flat_size[fit_idx]

            clf_gate = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
            y_bin_fit = (y_fit > 0.0).astype(int)
            def_fit_mask = (y_fit > 0.0)

            if len(np.unique(y_bin_fit)) >= 2 and np.sum(def_fit_mask) >= 5:
                clf_gate.fit(X_fit_s, y_bin_fit)
                p_all = clf_gate.predict_proba(scaler.transform(flat_feats))[:, 1]
                ridge_cond = Ridge(alpha=1.0, random_state=42)
                ridge_cond.fit(X_fit_s[def_fit_mask], y_fit[def_fit_mask])
                s_cond_all = np.maximum(0.0, ridge_cond.predict(scaler.transform(flat_feats)))

                p_fit = clf_gate.predict_proba(X_fit_s)[:, 1]
                s_fit_pred = np.maximum(0.0, ridge_cond.predict(X_fit_s))
                best_tau, best_score = 0.5, -1e9
                for c_tau in np.linspace(0.3, 0.95, 27):
                    score = r2_score(y_fit, np.where(p_fit >= c_tau, s_fit_pred, 0.0))
                    if score > best_score:
                        best_score, best_tau = score, float(c_tau)

                p_gate_bin = (p_all >= best_tau).reshape(min_Y, min_X).astype(int)
                p_gate_clean = apply_spatial_coherence_filter(p_gate_bin, min_area=8)
                pred_size_flat = np.where(p_gate_clean.reshape(-1) == 1, s_cond_all, 0.0)
            else:
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_fit_s, y_fit)
                pred_size_flat = np.clip(ridge.predict(scaler.transform(flat_feats)), 0.0, None)

            pred_size_map = pred_size_flat.reshape(min_Y, min_X)
            size_hurdle_r2 = hurdle_size_res.get("compound_hurdle", {}).get("plate_r2_score")
            size_defect_r2 = hurdle_size_res.get("conditional_defect_sizing", {}).get("r2_score")
            size_defect_mae = hurdle_size_res.get("conditional_defect_sizing", {}).get("mae_mm")

            pred_size_map_path = os.path.join(task2b_dir, f"{fname_base}_predicted_size_map.png")
            size_scatter_path = os.path.join(task2b_dir, f"{fname_base}_flaw_size_scatter.png")

            plot_flaw_size_maps(
                true_size_map=sub_size,
                pred_size_map=pred_size_map,
                save_path=pred_size_map_path,
                title=f"Universal Flaw Sizing Map (Within-File) | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
            )
            plot_flaw_size_calibration_scatter(
                true_sizes=flat_size[fit_idx],
                pred_sizes=pred_size_flat[fit_idx],
                save_path=size_scatter_path,
                title=f"Flaw Size Calibration Scatter | {fname_base}",
                r2=size_defect_r2,
                mae=size_defect_mae,
                unit="mm",
            )

            task2b_res = {
                "flaw_size_hurdle_r2": size_hurdle_r2,
                "flaw_size_defect_r2": size_defect_r2,
                "flaw_size_defect_mae_mm": size_defect_mae,
                "pred_size_map_path": pred_size_map_path,
                "size_scatter_path": size_scatter_path,
                "size_pairs": (flat_size[fit_idx].copy(), pred_size_flat[fit_idx].copy()),
            }
        except Exception as e:
            task2b_res = {"error": str(e)}

    # Task 3: 3D Defect Tomography & Topological Graph
    task3d_res: Dict[str, Any] = {}
    if eval_3d and volume_3d is not None:
        try:
            from .evaluation.tomography_3d import plot_3d_ortho_slices, export_3d_interactive_html
            from .evaluation.graph_defect import (
                build_defect_graph,
                compute_crack_metrics,
                compute_corrosion_volume,
                plot_defect_graph_3d,
                export_graph_interactive_html,
            )
            v_sub = volume_3d[:min_Y, :min_X].copy()
            if prob_map is not None:
                p_gate = np.clip((prob_map - 0.20) / 0.30, 0.0, 1.0)
                v_gated = v_sub * p_gate[:, :, np.newaxis]
            else:
                v_gated = v_sub

            ortho_path = os.path.join(task3d_dir, f"{fname_base}_3d_ortho_slices.png")
            plot_3d_ortho_slices(volume_3d=v_gated, mask_2d=sub_gt, save_path=ortho_path, title=f"3D Tomography: {fname_base}", specimen=specimen_key)

            html_3d_path = os.path.join(task3d_dir, f"{fname_base}_3d_tomography.html")
            export_3d_interactive_html(volume_3d=v_gated, save_path=html_3d_path, title=f"3D PECT-JEPA Tomography: {fname_base}")

            graph_data = build_defect_graph(v_gated, threshold_percentile=95.0)
            crack_metrics = compute_crack_metrics(graph_data) if "rivet" in specimen_key.lower() else None
            corrosion_metrics = compute_corrosion_volume(graph_data) if "corrosion" in specimen_key.lower() else None

            graph_png_path = os.path.join(task3d_dir, f"{fname_base}_defect_graph_3d.png")
            plot_defect_graph_3d(graph_data=graph_data, crack_metrics=crack_metrics, save_path=graph_png_path, title=f"3D Defect Graph Network: {fname_base}")

            graph_html_path = os.path.join(task3d_dir, f"{fname_base}_defect_graph_3d.html")
            export_graph_interactive_html(graph_data=graph_data, crack_metrics=crack_metrics, save_path=graph_html_path, title=f"Interactive 3D Defect Graph: {fname_base}")

            task3d_res = {
                "ortho_slices_path": ortho_path,
                "interactive_3d_html": html_3d_path,
                "graph_plot_path": graph_png_path,
                "graph_interactive_html": graph_html_path,
                "num_defect_nodes": graph_data.get("num_nodes", 0),
                "num_defect_edges": graph_data.get("num_edges", 0),
                "crack_metrics": crack_metrics,
                "corrosion_metrics": corrosion_metrics,
            }
        except Exception as e:
            task3d_res = {"error": str(e)}

    # Task 4: High-Dimensional Latent Diagnostics
    task_diag_res: Dict[str, Any] = {}
    if eval_diagnostics and has_gt and sub_gt is not None:
        try:
            sens_path = os.path.join(task_diag_dir, f"{fname_base}_channel_sensitivity.png")
            sens_res = compute_channel_sensitivity_spectrum(
                features=sub_feat,
                gt_labels=sub_gt,
                save_path=sens_path,
                title=f"Latent Defect Sensitivity Spectrum | {fname_base}",
            )
            morph_path = os.path.join(task_diag_dir, f"{fname_base}_morphology_separation.png")
            morph_labels = build_morphology_labels_from_scan(specimen_key, sub_gt, sub_depth)
            sep_res = compute_morphology_subspace_separation(
                features=sub_feat,
                morphology_labels=morph_labels,
                save_path=morph_path,
                title=f"Morphology Subspace Separation | {fname_base}",
            )
            task_diag_res = {
                "mean_channel_sensitivity": sens_res.get("mean_sensitivity"),
                "max_channel_sensitivity": sens_res.get("max_sensitivity"),
                "top_active_channels": sens_res.get("top_channels"),
                "channel_sensitivity_path": sens_path,
                "morphology_separation_path": morph_path,
                "cosine_distance_matrix": sep_res.get("cosine_distance_matrix"),
            }
        except Exception as e:
            task_diag_res = {"error": str(e)}

    if save_features:
        feat_path = to_safe_path(os.path.join(output_dir, f"{fname_base}_features_5x5.npy"))
        np.save(feat_path, feature_map)

    t1_lp = task1_res.get("linear_probe", {})
    t1_mlp = task1_res.get("mlp_2layer", {})
    t2_lp = task2_res.get("linear_probe", {})

    metrics_flat: Dict[str, Any] = {
        "has_ground_truth": has_gt,
        "contrast_ratio_cnr": task1_res.get("contrast_ratio_cnr"),
        "peak_contrast_ratio": task1_res.get("peak_contrast_ratio"),
        "auc_roc": t1_lp.get("auc_roc"),
        "average_precision": t1_lp.get("average_precision"),
        "best_f1": t1_lp.get("f1_score"),
        "linear_probe_auc_roc": t1_lp.get("auc_roc"),
        "linear_probe_average_precision": t1_lp.get("average_precision"),
        "linear_probe_f1": t1_lp.get("f1_score"),
        "mlp_2layer_auc_roc": t1_mlp.get("auc_roc"),
        "mlp_2layer_ap": t1_mlp.get("average_precision"),
        "mlp_2layer_f1": t1_mlp.get("f1_score"),
        "delta_auc": task1_res.get("representation_gap", {}).get("delta_auc_roc"),
        "depth_r2": t2_lp.get("r2_score"),
        "hurdle_plate_r2": task2_res.get("hurdle_plate_r2"),
        "defect_only_r2": task2_res.get("defect_only_r2"),
        "depth_mae_mm": t2_lp.get("mae_mm"),
        "depth_rmse_mm": t2_lp.get("rmse_mm"),
        "knn_5_accuracy": task1_res.get("knn_5_accuracy"),
        "unsupervised_maha_auc": task1_res.get("unsupervised_mahalanobis", {}).get("auc_roc"),
        "unsupervised_maha_ap": task1_res.get("unsupervised_mahalanobis", {}).get("average_precision"),
        "unsupervised_maha_cnr": task1_res.get("unsupervised_mahalanobis", {}).get("contrast_ratio_cnr"),
        "defect_iou_jaccard": task1_res.get("segmentation_iou", {}).get("iou"),
        "defect_dice_f1": task1_res.get("segmentation_iou", {}).get("dice"),
        "flaw_size_defect_r2": task2b_res.get("flaw_size_defect_r2"),
        "flaw_size_defect_mae_mm": task2b_res.get("flaw_size_defect_mae_mm"),
        "flaw_size_plate_r2": task2b_res.get("flaw_size_hurdle_r2"),
        "mean_latent_sensitivity": task_diag_res.get("mean_channel_sensitivity"),
        "max_latent_sensitivity": task_diag_res.get("max_channel_sensitivity"),
    }

    result = {
        "file": file_path,
        "file_name": os.path.basename(file_path),
        "specimen": specimen_key,
        "metadata": meta,
        "metrics": metrics_flat,
        "task1_anomaly_detection": task1_res,
        "task2_depth_regression": task2_res,
        "task2b_flaw_size_sizing": task2b_res,
        "task3_3d_tomography": task3d_res,
        "task4_latent_diagnostics": task_diag_res,
    }
    return result


def evaluate_single_file_zeroshot(
    file_path: str,
    feature_map: np.ndarray,
    volume_3d: Optional[np.ndarray],
    base_readouts: Dict[str, Any],
    output_dir: str,
    model_config: Any,
    save_features: bool = False,
    crop_border: int = 15,
    eval_3d: bool = True,
    eval_diagnostics: bool = True,
) -> Dict[str, Any]:
    """
    Mode B: Zero_Shot Evaluation.
    Evaluates pre-fitted frozen base readouts (trained strictly on the train set pool)
    zero-shot on this scan.
    Saves artifacts to <output_dir>/<Task>/<specimen>/
    """
    fname_base = os.path.splitext(os.path.basename(file_path))[0]
    meta = extract_file_metadata(file_path)
    gt_mgr = get_ground_truth_manager(data_dir=getattr(model_config, "data_dir", "data"))
    specimen_key = gt_mgr.canonical_specimen_key(file_path)

    task1_dir = to_safe_path(os.path.join(output_dir, "1_Anomaly_Detection", specimen_key))
    task2_dir = to_safe_path(os.path.join(output_dir, "2_Depth_Regression", specimen_key))
    task2b_dir = to_safe_path(os.path.join(output_dir, "2b_Size_Regression", specimen_key))
    task3d_dir = to_safe_path(os.path.join(output_dir, "3D_Tomography", specimen_key))
    task_diag_dir = to_safe_path(os.path.join(output_dir, "Latent_Diagnostics", specimen_key))

    for d in [task1_dir, task2_dir, task2b_dir]:
        os.makedirs(d, exist_ok=True)
    if eval_3d:
        os.makedirs(task3d_dir, exist_ok=True)
    if eval_diagnostics:
        os.makedirs(task_diag_dir, exist_ok=True)

    gt_mask = gt_mgr.get_ground_truth_mask_for_file(file_path, aligned_scan=True)
    depth_map_gt = gt_mgr.generate_depth_map(specimen_key)
    size_map_gt = gt_mgr.generate_size_map(specimen_key, mode="diameter")

    has_gt = gt_mask is not None
    min_Y = min(feature_map.shape[0], gt_mask.shape[0]) if has_gt else feature_map.shape[0]
    min_X = min(feature_map.shape[1], gt_mask.shape[1]) if has_gt else feature_map.shape[1]

    sub_feat = feature_map[:min_Y, :min_X]
    sub_gt = gt_mask[:min_Y, :min_X] if has_gt else None
    sub_depth = depth_map_gt[:min_Y, :min_X] if depth_map_gt is not None else None
    sub_size = size_map_gt[:min_Y, :min_X] if size_map_gt is not None else None

    flat_feats = sub_feat.reshape(-1, sub_feat.shape[-1]).astype(np.float32)

    # 1. Zero-Shot Anomaly Detection (Frozen Base Probes)
    task1_res: Dict[str, Any] = {}
    prob_map = None
    scaler_det = base_readouts.get("scaler_det")
    lr_probe = base_readouts.get("lr_probe")
    mlp_probe = base_readouts.get("mlp_probe")

    if scaler_det is not None and lr_probe is not None:
        try:
            X_s = scaler_det.transform(flat_feats)
            p_lr_all = lr_probe.predict_proba(X_s)[:, 1]
            p_mlp_all = mlp_probe.predict_proba(X_s)[:, 1] if mlp_probe is not None else p_lr_all
            prob_map = p_lr_all.reshape(min_Y, min_X)

            lp_auc, lp_ap, lp_f1, lp_acc = None, None, None, None
            mlp_auc, mlp_ap, mlp_f1, mlp_acc = None, None, None, None
            cnr, peak_cnr = 0.0, 0.0

            if has_gt and sub_gt is not None:
                flat_gt = sub_gt.reshape(-1).astype(np.int64)
                v_idx = np.where(flat_gt >= 0)[0]
                if len(v_idx) > 0 and len(np.unique(flat_gt[v_idx])) >= 2:
                    y_eval = flat_gt[v_idx]
                    p_eval_lr = p_lr_all[v_idx]
                    p_eval_mlp = p_mlp_all[v_idx]

                    lp_auc = float(roc_auc_score(y_eval, p_eval_lr))
                    lp_ap = float(average_precision_score(y_eval, p_eval_lr))
                    lp_f1 = float(f1_score(y_eval, (p_eval_lr >= 0.5).astype(int), zero_division=0))
                    lp_acc = float(accuracy_score(y_eval, (p_eval_lr >= 0.5).astype(int)))

                    mlp_auc = float(roc_auc_score(y_eval, p_eval_mlp))
                    mlp_ap = float(average_precision_score(y_eval, p_eval_mlp))
                    mlp_f1 = float(f1_score(y_eval, (p_eval_mlp >= 0.5).astype(int), zero_division=0))
                    mlp_acc = float(accuracy_score(y_eval, (p_eval_mlp >= 0.5).astype(int)))

                    cnr_res = compute_anomaly_metrics(prob_map, gt_mask=sub_gt)
                    cnr = float(cnr_res.get("contrast_ratio_cnr", 0.0))
                    peak_cnr = float(cnr_res.get("peak_contrast_ratio", 0.0))

            task1_res = {
                "linear_probe": {
                    "auc_roc": lp_auc,
                    "average_precision": lp_ap,
                    "f1_score": lp_f1,
                    "accuracy": lp_acc,
                },
                "mlp_2layer": {
                    "auc_roc": mlp_auc,
                    "average_precision": mlp_ap,
                    "f1_score": mlp_f1,
                    "accuracy": mlp_acc,
                },
                "representation_gap": {
                    "delta_auc_roc": round(mlp_auc - lp_auc, 4) if (mlp_auc is not None and lp_auc is not None) else None,
                    "delta_average_precision": round(mlp_ap - lp_ap, 4) if (mlp_ap is not None and lp_ap is not None) else None,
                },
                "contrast_ratio_cnr": cnr,
                "peak_contrast_ratio": peak_cnr,
            }

            # Unsupervised Mahalanobis
            try:
                maha = MahalanobisDetector(regularize_eps=1e-4)
                maha.fit(sub_feat, contamination=0.05)
                maha_score = maha.score_map(sub_feat)
                if has_gt and sub_gt is not None:
                    valid_m = (sub_gt.reshape(-1) >= 0)
                    y_m = sub_gt.reshape(-1)[valid_m]
                    s_m = maha_score.reshape(-1)[valid_m]
                    if len(np.unique(y_m)) > 1:
                        maha_auc = float(roc_auc_score(y_m, s_m))
                        maha_ap = float(average_precision_score(y_m, s_m))
                        def_pts = maha_score[sub_gt == 1]
                        snd_pts = maha_score[sub_gt == 0]
                        maha_cnr = float((np.mean(def_pts) - np.mean(snd_pts)) / (np.std(snd_pts) + 1e-8)) if len(def_pts) > 0 and len(snd_pts) > 0 else 0.0
                    else:
                        maha_auc, maha_ap, maha_cnr = None, None, None
                else:
                    maha_auc, maha_ap, maha_cnr = None, None, None
            except Exception:
                maha_auc, maha_ap, maha_cnr = None, None, None

            task1_res["unsupervised_mahalanobis"] = {
                "auc_roc": maha_auc,
                "average_precision": maha_ap,
                "contrast_ratio_cnr": maha_cnr,
            }

            # Visualizations
            prob_heatmap_path = os.path.join(task1_dir, f"{fname_base}_prob_heatmap.png")
            auc_disp = f"{lp_auc:.4f}" if lp_auc is not None else "N/A"
            plot_probability_heatmap(
                prob_map=prob_map,
                save_path=prob_heatmap_path,
                title=f"Defect Probability Map (Zero-Shot) | {meta.get('specimen', '')} - {meta.get('sensor', '')}\n"
                      f"Lift-off: {meta.get('liftoff', '')} | Waveform: {meta.get('waveform', '')} | Zero-Shot AUC: {auc_disp} | CNR: {cnr:.2f}",
            )
            task1_res["prob_heatmap_path"] = prob_heatmap_path

            if has_gt and sub_gt is not None and len(np.unique(flat_gt[v_idx])) >= 2:
                fpr_lr, tpr_lr, _ = roc_curve(y_eval, p_eval_lr)
                prec_lr, rec_lr, _ = precision_recall_curve(y_eval, p_eval_lr)
                roc_pr_path = os.path.join(task1_dir, f"{fname_base}_roc_pr_curve.png")
                plot_roc_pr_curves(
                    fpr=fpr_lr.tolist(),
                    tpr=tpr_lr.tolist(),
                    auc_roc=lp_auc if lp_auc is not None else 0.5,
                    precision=prec_lr.tolist(),
                    recall=rec_lr.tolist(),
                    avg_prec=lp_ap if lp_ap is not None else 0.0,
                    save_path=roc_pr_path,
                    title=f"ROC & PR Curves (Zero-Shot) | {fname_base}",
                )
                task1_res["roc_pr_path"] = roc_pr_path

                contour_path = os.path.join(task1_dir, f"{fname_base}_defect_contours_iou.png")
                iou_dict = plot_defect_contours_and_iou(
                    prob_map=prob_map,
                    gt_mask=sub_gt,
                    save_path=contour_path,
                    title=f"Defect Contours & IoU (Zero-Shot) | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
                )
                task1_res["segmentation_iou"] = iou_dict
                task1_res["defect_contours_path"] = contour_path

        except Exception as e:
            task1_res = {"error": str(e)}

    # 2. Zero-Shot Quantitative Depth Sizing (Surface Flaws)
    task2_res: Dict[str, Any] = {}
    is_surface = ("corrosion" in specimen_key.lower())
    has_depth_labels = (has_gt and sub_depth is not None and np.sum(sub_depth > 0.0) >= 5)
    scaler_depth = base_readouts.get("scaler_depth")
    ridge_depth = base_readouts.get("ridge_depth")

    if is_surface and has_depth_labels and scaler_depth is not None and ridge_depth is not None:
        try:
            flat_depth = sub_depth.reshape(-1).astype(np.float32)
            d_pred_raw = np.maximum(0.0, ridge_depth.predict(scaler_depth.transform(flat_feats)))

            # Two-Stage Hurdle Map
            p_bin = (prob_map >= 0.5).astype(int) if prob_map is not None else np.ones((min_Y, min_X), dtype=int)
            p_clean = apply_spatial_coherence_filter(p_bin, min_area=8)
            pred_depth_flat = np.where(p_clean.reshape(-1) == 1, d_pred_raw, 0.0)
            pred_depth_map = pred_depth_flat.reshape(min_Y, min_X)

            # Metrics
            def_idx = np.where(flat_depth > 0.0)[0]
            if len(def_idx) >= 5:
                defect_r2 = float(r2_score(flat_depth[def_idx], d_pred_raw[def_idx]))
                defect_mae = float(mean_absolute_error(flat_depth[def_idx], d_pred_raw[def_idx]))
                defect_rmse = float(np.sqrt(mean_squared_error(flat_depth[def_idx], d_pred_raw[def_idx])))
            else:
                defect_r2, defect_mae, defect_rmse = None, None, None

            hurdle_r2 = float(r2_score(flat_depth, pred_depth_flat))
            hurdle_mae = float(mean_absolute_error(flat_depth, pred_depth_flat))

            pred_depth_map_path = os.path.join(task2_dir, f"{fname_base}_predicted_depth_map.png")
            depth_scatter_path = os.path.join(task2_dir, f"{fname_base}_depth_scatter.png")

            plot_depth_regression_maps(
                true_depth_map=sub_depth,
                pred_depth_map=pred_depth_map,
                save_path=pred_depth_map_path,
                title=f"Quantitative Depth Sizing (Zero-Shot) | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
                r2=defect_r2,
                mae=defect_mae,
                rmse=defect_rmse,
            )
            plot_depth_calibration_scatter(
                true_depth=flat_depth[def_idx],
                pred_depth=d_pred_raw[def_idx],
                save_path=depth_scatter_path,
                title=f"Depth Calibration Scatter (Zero-Shot) | {fname_base}",
                r2=defect_r2,
                mae=defect_mae,
                rmse=defect_rmse,
            )

            task2_res = {
                "linear_probe": {"r2_score": hurdle_r2, "mae_mm": hurdle_mae, "rmse_mm": defect_rmse},
                "defect_only_r2": defect_r2,
                "hurdle_plate_r2": hurdle_r2,
                "pred_depth_map_path": pred_depth_map_path,
                "depth_scatter_path": depth_scatter_path,
                "morphology_routing": "Surface-breaking defect -> Zero-Shot Depth Regression executed",
            }
        except Exception as e:
            task2_res = {"error": str(e)}
    else:
        task2_res = {
            "notice": "Subsurface defect / fastener structure; routed to Task 2b Universal Flaw Sizing",
            "morphology_routing": "Subsurface defect / Fastener -> Task 2b Flaw Sizing",
            "linear_probe": {},
            "defect_only_r2": None,
            "hurdle_plate_r2": None,
        }

    # 3. Zero-Shot Universal Flaw Size Regression (All 3 Plates)
    task2b_res: Dict[str, Any] = {}
    scaler_size = base_readouts.get("scaler_size")
    ridge_size = base_readouts.get("ridge_size")

    if has_gt and sub_size is not None and np.sum(sub_size > 0.0) >= 5 and scaler_size is not None and ridge_size is not None:
        try:
            flat_size = sub_size.reshape(-1).astype(np.float32)
            s_pred_raw = np.maximum(0.0, ridge_size.predict(scaler_size.transform(flat_feats)))

            p_bin = (prob_map >= 0.5).astype(int) if prob_map is not None else np.ones((min_Y, min_X), dtype=int)
            p_clean = apply_spatial_coherence_filter(p_bin, min_area=8)
            pred_size_flat = np.where(p_clean.reshape(-1) == 1, s_pred_raw, 0.0)
            pred_size_map = pred_size_flat.reshape(min_Y, min_X)

            sz_idx = np.where(flat_size > 0.0)[0]
            if len(sz_idx) >= 5:
                size_defect_r2 = float(r2_score(flat_size[sz_idx], s_pred_raw[sz_idx]))
                size_defect_mae = float(mean_absolute_error(flat_size[sz_idx], s_pred_raw[sz_idx]))
            else:
                size_defect_r2, size_defect_mae = None, None

            size_hurdle_r2 = float(r2_score(flat_size, pred_size_flat))

            pred_size_map_path = os.path.join(task2b_dir, f"{fname_base}_predicted_size_map.png")
            size_scatter_path = os.path.join(task2b_dir, f"{fname_base}_flaw_size_scatter.png")

            plot_flaw_size_maps(
                true_size_map=sub_size,
                pred_size_map=pred_size_map,
                save_path=pred_size_map_path,
                title=f"Universal Flaw Sizing Map (Zero-Shot) | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
            )
            plot_flaw_size_calibration_scatter(
                true_sizes=flat_size[sz_idx],
                pred_sizes=s_pred_raw[sz_idx],
                save_path=size_scatter_path,
                title=f"Flaw Size Calibration Scatter (Zero-Shot) | {fname_base}",
                r2=size_defect_r2,
                mae=size_defect_mae,
                unit="mm",
            )

            task2b_res = {
                "flaw_size_hurdle_r2": size_hurdle_r2,
                "flaw_size_defect_r2": size_defect_r2,
                "flaw_size_defect_mae_mm": size_defect_mae,
                "pred_size_map_path": pred_size_map_path,
                "size_scatter_path": size_scatter_path,
                "size_pairs": (flat_size[sz_idx].copy(), s_pred_raw[sz_idx].copy()),
            }
        except Exception as e:
            task2b_res = {"error": str(e)}

    # 4. Zero-Shot 3D Volumetric Tomography
    task3d_res: Dict[str, Any] = {}
    if eval_3d and volume_3d is not None:
        try:
            from .evaluation.tomography_3d import plot_3d_ortho_slices, export_3d_interactive_html
            from .evaluation.graph_defect import (
                build_defect_graph,
                compute_crack_metrics,
                compute_corrosion_volume,
                plot_defect_graph_3d,
                export_graph_interactive_html,
            )
            v_sub = volume_3d[:min_Y, :min_X].copy()
            if prob_map is not None:
                p_gate = np.clip((prob_map - 0.20) / 0.30, 0.0, 1.0)
                v_gated = v_sub * p_gate[:, :, np.newaxis]
            else:
                v_gated = v_sub

            ortho_path = os.path.join(task3d_dir, f"{fname_base}_3d_ortho_slices.png")
            plot_3d_ortho_slices(volume_3d=v_gated, mask_2d=sub_gt, save_path=ortho_path, title=f"3D Tomography (Zero-Shot): {fname_base}", specimen=specimen_key)

            html_3d_path = os.path.join(task3d_dir, f"{fname_base}_3d_tomography.html")
            export_3d_interactive_html(volume_3d=v_gated, save_path=html_3d_path, title=f"3D PECT-JEPA Tomography (Zero-Shot): {fname_base}")

            graph_data = build_defect_graph(v_gated, threshold_percentile=95.0)
            crack_metrics = compute_crack_metrics(graph_data) if "rivet" in specimen_key.lower() else None
            corrosion_metrics = compute_corrosion_volume(graph_data) if "corrosion" in specimen_key.lower() else None

            graph_png_path = os.path.join(task3d_dir, f"{fname_base}_defect_graph_3d.png")
            plot_defect_graph_3d(graph_data=graph_data, crack_metrics=crack_metrics, save_path=graph_png_path, title=f"3D Defect Graph Network (Zero-Shot): {fname_base}")

            graph_html_path = os.path.join(task3d_dir, f"{fname_base}_defect_graph_3d.html")
            export_graph_interactive_html(graph_data=graph_data, crack_metrics=crack_metrics, save_path=graph_html_path, title=f"Interactive 3D Defect Graph (Zero-Shot): {fname_base}")

            task3d_res = {
                "ortho_slices_path": ortho_path,
                "interactive_3d_html": html_3d_path,
                "graph_plot_path": graph_png_path,
                "graph_interactive_html": graph_html_path,
                "num_defect_nodes": graph_data.get("num_nodes", 0),
                "num_defect_edges": graph_data.get("num_edges", 0),
                "crack_metrics": crack_metrics,
                "corrosion_metrics": corrosion_metrics,
            }
        except Exception as e:
            task3d_res = {"error": str(e)}

    # 5. Zero-Shot Latent Diagnostics
    task_diag_res: Dict[str, Any] = {}
    if eval_diagnostics and has_gt and sub_gt is not None:
        try:
            sens_path = os.path.join(task_diag_dir, f"{fname_base}_channel_sensitivity.png")
            sens_res = compute_channel_sensitivity_spectrum(
                features=sub_feat,
                gt_labels=sub_gt,
                save_path=sens_path,
                title=f"Latent Defect Sensitivity Spectrum | {fname_base}",
            )
            morph_path = os.path.join(task_diag_dir, f"{fname_base}_morphology_separation.png")
            morph_labels = build_morphology_labels_from_scan(specimen_key, sub_gt, sub_depth)
            sep_res = compute_morphology_subspace_separation(
                features=sub_feat,
                morphology_labels=morph_labels,
                save_path=morph_path,
                title=f"Morphology Subspace Separation | {fname_base}",
            )
            task_diag_res = {
                "mean_channel_sensitivity": sens_res.get("mean_sensitivity"),
                "max_channel_sensitivity": sens_res.get("max_sensitivity"),
                "top_active_channels": sens_res.get("top_channels"),
                "channel_sensitivity_path": sens_path,
                "morphology_separation_path": morph_path,
                "cosine_distance_matrix": sep_res.get("cosine_distance_matrix"),
            }
        except Exception as e:
            task_diag_res = {"error": str(e)}

    if save_features:
        feat_path = to_safe_path(os.path.join(output_dir, f"{fname_base}_features_5x5.npy"))
        np.save(feat_path, feature_map)

    t1_lp = task1_res.get("linear_probe", {})
    t1_mlp = task1_res.get("mlp_2layer", {})
    t2_lp = task2_res.get("linear_probe", {})

    metrics_flat: Dict[str, Any] = {
        "has_ground_truth": has_gt,
        "contrast_ratio_cnr": task1_res.get("contrast_ratio_cnr"),
        "peak_contrast_ratio": task1_res.get("peak_contrast_ratio"),
        "auc_roc": t1_lp.get("auc_roc"),
        "average_precision": t1_lp.get("average_precision"),
        "best_f1": t1_lp.get("f1_score"),
        "linear_probe_auc_roc": t1_lp.get("auc_roc"),
        "linear_probe_average_precision": t1_lp.get("average_precision"),
        "linear_probe_f1": t1_lp.get("f1_score"),
        "mlp_2layer_auc_roc": t1_mlp.get("auc_roc"),
        "mlp_2layer_ap": t1_mlp.get("average_precision"),
        "mlp_2layer_f1": t1_mlp.get("f1_score"),
        "delta_auc": task1_res.get("representation_gap", {}).get("delta_auc_roc"),
        "depth_r2": t2_lp.get("r2_score"),
        "hurdle_plate_r2": task2_res.get("hurdle_plate_r2"),
        "defect_only_r2": task2_res.get("defect_only_r2"),
        "depth_mae_mm": t2_lp.get("mae_mm"),
        "depth_rmse_mm": t2_lp.get("rmse_mm"),
        "knn_5_accuracy": None,
        "unsupervised_maha_auc": task1_res.get("unsupervised_mahalanobis", {}).get("auc_roc"),
        "unsupervised_maha_ap": task1_res.get("unsupervised_mahalanobis", {}).get("average_precision"),
        "unsupervised_maha_cnr": task1_res.get("unsupervised_mahalanobis", {}).get("contrast_ratio_cnr"),
        "defect_iou_jaccard": task1_res.get("segmentation_iou", {}).get("iou"),
        "defect_dice_f1": task1_res.get("segmentation_iou", {}).get("dice"),
        "flaw_size_defect_r2": task2b_res.get("flaw_size_defect_r2"),
        "flaw_size_defect_mae_mm": task2b_res.get("flaw_size_defect_mae_mm"),
        "flaw_size_plate_r2": task2b_res.get("flaw_size_hurdle_r2"),
        "mean_latent_sensitivity": task_diag_res.get("mean_channel_sensitivity"),
        "max_latent_sensitivity": task_diag_res.get("max_channel_sensitivity"),
    }

    result = {
        "file": file_path,
        "file_name": os.path.basename(file_path),
        "specimen": specimen_key,
        "metadata": meta,
        "metrics": metrics_flat,
        "task1_anomaly_detection": task1_res,
        "task2_depth_regression": task2_res,
        "task2b_flaw_size_sizing": task2b_res,
        "task3_3d_tomography": task3d_res,
        "task4_latent_diagnostics": task_diag_res,
    }
    return result


def evaluate_single_file(
    file_path: str,
    model: PECT_JEPA_5x5,
    output_dir: str,
    batch_size: int = 512,
    device: str = "cuda",
    save_features: bool = False,
    crop_border: int = 15,
    eval_3d: bool = True,
    eval_diagnostics: bool = True,
    morphology_clf: Optional[DefectMorphologyClassifier] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for single-file evaluation."""
    grid_3d = load_cscan_from_tdms(
        file_path,
        time_samples=model.config.time_samples,
        temporal_samples=model.config.temporal_samples,
        resample_mode=model.config.resample_mode,
        normalization=model.config.normalization,
        raster_correction=model.config.raster_correction,
        crop_border=crop_border,
        apply_lowpass=getattr(model.config, "apply_lowpass", True),
        lowpass_cutoff=getattr(model.config, "lowpass_cutoff", 2500.0),
        lowpass_order=getattr(model.config, "lowpass_order", 4),
    )
    if eval_3d:
        feature_map, volume_3d = extract_full_cscan_map(
            model=model, full_cscan_3d=grid_3d, batch_size=batch_size, device=device, show_pbar=True, return_volume_3d=True
        )
    else:
        feature_map = extract_full_cscan_map(
            model=model, full_cscan_3d=grid_3d, batch_size=batch_size, device=device, show_pbar=True, return_volume_3d=False
        )
        volume_3d = None

    return evaluate_single_file_within_file(
        file_path=file_path,
        feature_map=feature_map,
        volume_3d=volume_3d,
        output_dir=output_dir,
        model_config=model.config,
        save_features=save_features,
        crop_border=crop_border,
        eval_3d=eval_3d,
        eval_diagnostics=eval_diagnostics,
        morphology_clf=morphology_clf,
    )


def save_split_summaries(
    file_results: List[Dict[str, Any]],
    output_dir: str,
    split_name: str,
    mode_name: str,
    protocol_name: str,
    holdout_target: str,
    model: PECT_JEPA_5x5,
    crop_border: int,
    all_size_pairs: List[Any],
) -> Dict[str, Any]:
    """
    Saves dedicated specimen summaries for Tasks 1, 2, 2b, 3, 4,
    global size calibration scatter plot, and consolidated evaluation_summary.json/csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    specimen_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in file_results:
        sp = r.get("specimen", "unknown")
        specimen_groups.setdefault(sp, []).append(r)

    # 1. Specimen summaries for Task 1
    for sp, sp_results in specimen_groups.items():
        t1_sp_dir = os.path.join(output_dir, "1_Anomaly_Detection", sp)
        os.makedirs(t1_sp_dir, exist_ok=True)
        aucs = [r["metrics"]["linear_probe_auc_roc"] for r in sp_results if r["metrics"].get("linear_probe_auc_roc") is not None]
        aps = [r["metrics"]["linear_probe_average_precision"] for r in sp_results if r["metrics"].get("linear_probe_average_precision") is not None]
        f1s = [r["metrics"]["linear_probe_f1"] for r in sp_results if r["metrics"].get("linear_probe_f1") is not None]
        cnrs = [r["metrics"]["contrast_ratio_cnr"] for r in sp_results if r["metrics"].get("contrast_ratio_cnr") is not None]
        ious = [r["metrics"]["defect_iou_jaccard"] for r in sp_results if r["metrics"].get("defect_iou_jaccard") is not None]
        dices = [r["metrics"]["defect_dice_f1"] for r in sp_results if r["metrics"].get("defect_dice_f1") is not None]
        maha_aucs = [r["metrics"]["unsupervised_maha_auc"] for r in sp_results if r["metrics"].get("unsupervised_maha_auc") is not None]

        sp_t1_summary = {
            "mode": mode_name,
            "split": split_name,
            "specimen": sp,
            "total_files": len(sp_results),
            "labeled_files": len(aucs),
            "linear_probe": {
                "mean_auc_roc": float(np.mean(aucs)) if aucs else None,
                "std_auc_roc": float(np.std(aucs)) if aucs else None,
                "mean_average_precision": float(np.mean(aps)) if aps else None,
                "mean_f1": float(np.mean(f1s)) if f1s else None,
            },
            "mean_contrast_ratio_cnr": float(np.mean(cnrs)) if cnrs else None,
            "mean_defect_iou_jaccard": float(np.mean(ious)) if ious else None,
            "mean_defect_dice_f1": float(np.mean(dices)) if dices else None,
            "mean_unsupervised_maha_auc": float(np.mean(maha_aucs)) if maha_aucs else None,
            "files": [
                {
                    "file_name": r["file_name"],
                    "sensor": r["metadata"].get("sensor"),
                    "waveform": r["metadata"].get("waveform"),
                    "liftoff": r["metadata"].get("liftoff"),
                    "auc_roc": r["metrics"].get("linear_probe_auc_roc"),
                    "average_precision": r["metrics"].get("linear_probe_average_precision"),
                    "f1": r["metrics"].get("linear_probe_f1"),
                    "cnr": r["metrics"].get("contrast_ratio_cnr"),
                    "iou": r["metrics"].get("defect_iou_jaccard"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t1_sp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t1_summary, f, indent=2)

    # 2. Specimen summaries for Task 2
    for sp, sp_results in specimen_groups.items():
        t2_sp_dir = os.path.join(output_dir, "2_Depth_Regression", sp)
        os.makedirs(t2_sp_dir, exist_ok=True)
        r2s = [r["metrics"]["depth_r2"] for r in sp_results if r["metrics"].get("depth_r2") is not None]
        def_r2s = [r["metrics"]["defect_only_r2"] for r in sp_results if r["metrics"].get("defect_only_r2") is not None]
        maes = [r["metrics"]["depth_mae_mm"] for r in sp_results if r["metrics"].get("depth_mae_mm") is not None]
        rmses = [r["metrics"]["depth_rmse_mm"] for r in sp_results if r["metrics"].get("depth_rmse_mm") is not None]

        sp_t2_summary = {
            "mode": mode_name,
            "split": split_name,
            "specimen": sp,
            "total_files": len(sp_results),
            "evaluated_surface_files": len(r2s),
            "mean_plate_r2_score": float(np.mean(r2s)) if r2s else None,
            "mean_defect_only_r2": float(np.mean(def_r2s)) if def_r2s else None,
            "mean_mae_mm": float(np.mean(maes)) if maes else None,
            "mean_rmse_mm": float(np.mean(rmses)) if rmses else None,
            "files": [
                {
                    "file_name": r["file_name"],
                    "sensor": r["metadata"].get("sensor"),
                    "waveform": r["metadata"].get("waveform"),
                    "liftoff": r["metadata"].get("liftoff"),
                    "defect_only_r2": r["metrics"].get("defect_only_r2"),
                    "depth_r2": r["metrics"].get("depth_r2"),
                    "mae_mm": r["metrics"].get("depth_mae_mm"),
                    "rmse_mm": r["metrics"].get("depth_rmse_mm"),
                    "notice": r.get("task2_depth_regression", {}).get("notice"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t2_sp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t2_summary, f, indent=2)

    # 3. Specimen summaries for Task 2b
    for sp, sp_results in specimen_groups.items():
        t2b_sp_dir = os.path.join(output_dir, "2b_Size_Regression", sp)
        os.makedirs(t2b_sp_dir, exist_ok=True)
        size_def_r2s = [r["metrics"]["flaw_size_defect_r2"] for r in sp_results if r["metrics"].get("flaw_size_defect_r2") is not None]
        size_maes = [r["metrics"]["flaw_size_defect_mae_mm"] for r in sp_results if r["metrics"].get("flaw_size_defect_mae_mm") is not None]
        size_plate_r2s = [r["metrics"]["flaw_size_plate_r2"] for r in sp_results if r["metrics"].get("flaw_size_plate_r2") is not None]

        sp_t2b_summary = {
            "mode": mode_name,
            "split": split_name,
            "specimen": sp,
            "total_files": len(sp_results),
            "evaluated_files": len(size_def_r2s),
            "mean_flaw_size_defect_r2": float(np.mean(size_def_r2s)) if size_def_r2s else None,
            "mean_flaw_size_defect_mae_mm": float(np.mean(size_maes)) if size_maes else None,
            "mean_flaw_size_plate_r2": float(np.mean(size_plate_r2s)) if size_plate_r2s else None,
            "files": [
                {
                    "file_name": r["file_name"],
                    "sensor": r["metadata"].get("sensor"),
                    "waveform": r["metadata"].get("waveform"),
                    "liftoff": r["metadata"].get("liftoff"),
                    "flaw_size_defect_r2": r["metrics"].get("flaw_size_defect_r2"),
                    "flaw_size_defect_mae_mm": r["metrics"].get("flaw_size_defect_mae_mm"),
                    "flaw_size_plate_r2": r["metrics"].get("flaw_size_plate_r2"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t2b_sp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t2b_summary, f, indent=2)

    # 4. Specimen summaries for Task 3
    for sp, sp_results in specimen_groups.items():
        t3d_sp_dir = os.path.join(output_dir, "3D_Tomography", sp)
        os.makedirs(t3d_sp_dir, exist_ok=True)
        t3d_list = [r.get("task3_3d_tomography", {}) for r in sp_results if r.get("task3_3d_tomography")]
        sp_t3d_summary = {
            "mode": mode_name,
            "split": split_name,
            "specimen": sp,
            "total_files": len(sp_results),
            "tomography_count": len(t3d_list),
            "files": [
                {
                    "file_name": r["file_name"],
                    "num_defect_nodes": r.get("task3_3d_tomography", {}).get("num_defect_nodes"),
                    "num_defect_edges": r.get("task3_3d_tomography", {}).get("num_defect_edges"),
                    "crack_metrics": r.get("task3_3d_tomography", {}).get("crack_metrics"),
                    "corrosion_metrics": r.get("task3_3d_tomography", {}).get("corrosion_metrics"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t3d_sp_dir, "tomography_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t3d_summary, f, indent=2)

    # 5. Specimen summaries for Task 4
    for sp, sp_results in specimen_groups.items():
        tdiag_sp_dir = os.path.join(output_dir, "Latent_Diagnostics", sp)
        os.makedirs(tdiag_sp_dir, exist_ok=True)
        mean_sens = [r["metrics"]["mean_latent_sensitivity"] for r in sp_results if r["metrics"].get("mean_latent_sensitivity") is not None]
        max_sens = [r["metrics"]["max_latent_sensitivity"] for r in sp_results if r["metrics"].get("max_latent_sensitivity") is not None]
        sp_tdiag_summary = {
            "mode": mode_name,
            "split": split_name,
            "specimen": sp,
            "total_files": len(sp_results),
            "mean_channel_sensitivity": float(np.mean(mean_sens)) if mean_sens else None,
            "max_channel_sensitivity": float(np.max(max_sens)) if max_sens else None,
            "files": [
                {
                    "file_name": r["file_name"],
                    "mean_sensitivity": r["metrics"].get("mean_latent_sensitivity"),
                    "max_sensitivity": r["metrics"].get("max_latent_sensitivity"),
                    "top_channels": r.get("task4_latent_diagnostics", {}).get("top_active_channels"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(tdiag_sp_dir, "latent_diagnostics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_tdiag_summary, f, indent=2)

    # 6. Global Size Calibration Scatter Plot
    if all_size_pairs:
        try:
            valid_pairs = [p for p in all_size_pairs if len(p[0]) > 0 and len(p[1]) > 0]
            if valid_pairs:
                y_true_all_sz = np.concatenate([p[0] for p in valid_pairs])
                y_pred_all_sz = np.concatenate([p[1] for p in valid_pairs])
                if len(y_true_all_sz) >= 2:
                    global_scatter_dir = os.path.join(output_dir, "2b_Size_Regression")
                    os.makedirs(global_scatter_dir, exist_ok=True)
                    global_scatter_path = os.path.join(global_scatter_dir, "global_flaw_size_calibration_scatter.png")
                    r2_sz = float(r2_score(y_true_all_sz, y_pred_all_sz))
                    mae_sz = float(mean_absolute_error(y_true_all_sz, y_pred_all_sz))
                    plot_flaw_size_calibration_scatter(
                        true_sizes=y_true_all_sz,
                        pred_sizes=y_pred_all_sz,
                        save_path=global_scatter_path,
                        title=f"Global Flaw Size Calibration ({mode_name} | {split_name.upper()})",
                        r2=r2_sz,
                        mae=mae_sz,
                        unit="mm",
                    )
        except Exception as e:
            print(f"  [Size Scatter Warning] {e}", flush=True)

    # 7. Global Consolidated Evaluation Report & CSV
    all_cnrs = [r["metrics"]["contrast_ratio_cnr"] for r in file_results if r["metrics"].get("contrast_ratio_cnr") is not None]
    all_aucs = [r["metrics"]["linear_probe_auc_roc"] for r in file_results if r["metrics"].get("linear_probe_auc_roc") is not None]
    all_aps = [r["metrics"]["linear_probe_average_precision"] for r in file_results if r["metrics"].get("linear_probe_average_precision") is not None]
    all_f1s = [r["metrics"]["linear_probe_f1"] for r in file_results if r["metrics"].get("linear_probe_f1") is not None]
    all_r2s = [r["metrics"]["depth_r2"] for r in file_results if r["metrics"].get("depth_r2") is not None]
    all_maes = [r["metrics"]["depth_mae_mm"] for r in file_results if r["metrics"].get("depth_mae_mm") is not None]
    all_defect_r2s = [r["metrics"]["defect_only_r2"] for r in file_results if r["metrics"].get("defect_only_r2") is not None]
    all_hurdle_r2s = [r["metrics"]["hurdle_plate_r2"] for r in file_results if r["metrics"].get("hurdle_plate_r2") is not None]
    all_maha_aucs = [r["metrics"]["unsupervised_maha_auc"] for r in file_results if r["metrics"].get("unsupervised_maha_auc") is not None]
    all_maha_aps = [r["metrics"]["unsupervised_maha_ap"] for r in file_results if r["metrics"].get("unsupervised_maha_ap") is not None]
    all_ious = [r["metrics"]["defect_iou_jaccard"] for r in file_results if r["metrics"].get("defect_iou_jaccard") is not None]
    all_dices = [r["metrics"]["defect_dice_f1"] for r in file_results if r["metrics"].get("defect_dice_f1") is not None]
    all_size_defect_r2s = [r["metrics"]["flaw_size_defect_r2"] for r in file_results if r["metrics"].get("flaw_size_defect_r2") is not None]
    all_size_maes = [r["metrics"]["flaw_size_defect_mae_mm"] for r in file_results if r["metrics"].get("flaw_size_defect_mae_mm") is not None]
    all_size_plate_r2s = [r["metrics"]["flaw_size_plate_r2"] for r in file_results if r["metrics"].get("flaw_size_plate_r2") is not None]
    all_sens = [r["metrics"]["mean_latent_sensitivity"] for r in file_results if r["metrics"].get("mean_latent_sensitivity") is not None]

    per_specimen_summary = {}
    for sp_name, sp_res_list in specimen_groups.items():
        sp_aucs = [r["metrics"]["linear_probe_auc_roc"] for r in sp_res_list if r["metrics"].get("linear_probe_auc_roc") is not None]
        sp_aps = [r["metrics"]["linear_probe_average_precision"] for r in sp_res_list if r["metrics"].get("linear_probe_average_precision") is not None]
        sp_cnrs = [r["metrics"]["contrast_ratio_cnr"] for r in sp_res_list if r["metrics"].get("contrast_ratio_cnr") is not None]
        sp_maha_aucs = [r["metrics"]["unsupervised_maha_auc"] for r in sp_res_list if r["metrics"].get("unsupervised_maha_auc") is not None]
        sp_maha_aps = [r["metrics"]["unsupervised_maha_ap"] for r in sp_res_list if r["metrics"].get("unsupervised_maha_ap") is not None]
        sp_ious = [r["metrics"]["defect_iou_jaccard"] for r in sp_res_list if r["metrics"].get("defect_iou_jaccard") is not None]
        sp_dices = [r["metrics"]["defect_dice_f1"] for r in sp_res_list if r["metrics"].get("defect_dice_f1") is not None]
        sp_defect_r2s = [r["metrics"]["defect_only_r2"] for r in sp_res_list if r["metrics"].get("defect_only_r2") is not None]
        sp_maes = [r["metrics"]["depth_mae_mm"] for r in sp_res_list if r["metrics"].get("depth_mae_mm") is not None]
        sp_rmses = [r["metrics"]["depth_rmse_mm"] for r in sp_res_list if r["metrics"].get("depth_rmse_mm") is not None]
        sp_size_defect_r2s = [r["metrics"]["flaw_size_defect_r2"] for r in sp_res_list if r["metrics"].get("flaw_size_defect_r2") is not None]
        sp_size_maes = [r["metrics"]["flaw_size_defect_mae_mm"] for r in sp_res_list if r["metrics"].get("flaw_size_defect_mae_mm") is not None]

        per_specimen_summary[sp_name] = {
            "total_files": len(sp_res_list),
            "linear_probe_auc": float(np.mean(sp_aucs)) if sp_aucs else None,
            "linear_probe_ap": float(np.mean(sp_aps)) if sp_aps else None,
            "contrast_ratio_cnr": float(np.mean(sp_cnrs)) if sp_cnrs else None,
            "unsupervised_maha_auc": float(np.mean(sp_maha_aucs)) if sp_maha_aucs else None,
            "unsupervised_maha_ap": float(np.mean(sp_maha_aps)) if sp_maha_aps else None,
            "mean_defect_iou": float(np.mean(sp_ious)) if sp_ious else None,
            "mean_defect_dice": float(np.mean(sp_dices)) if sp_dices else None,
            "defect_only_r2": float(np.mean(sp_defect_r2s)) if sp_defect_r2s else None,
            "depth_mae_mm": float(np.mean(sp_maes)) if sp_maes else None,
            "depth_rmse_mm": float(np.mean(sp_rmses)) if sp_rmses else None,
            "flaw_size_defect_r2": float(np.mean(sp_size_defect_r2s)) if sp_size_defect_r2s else None,
            "flaw_size_defect_mae_mm": float(np.mean(sp_size_maes)) if sp_size_maes else None,
        }

    report = {
        "mode": mode_name,
        "split": split_name,
        "evaluation_protocol": protocol_name,
        "holdout_target": holdout_target,
        "model_architecture": {
            "tokenizer_type": getattr(model.config, "tokenizer_type", "unknown"),
            "encoder_type": getattr(model.config, "encoder_type", "standard"),
            "predictor_type": getattr(model.config, "predictor_type", "unknown"),
            "embed_dim": model.config.embed_dim,
            "crop_border": crop_border,
        },
        "aggregate_metrics": {
            "total_files_evaluated": len(file_results),
            "labeled_files_count": len(all_aucs),
            "task1_anomaly_detection": {
                "mean_linear_probe_auc_roc": float(np.mean(all_aucs)) if all_aucs else None,
                "std_linear_probe_auc_roc": float(np.std(all_aucs)) if all_aucs else None,
                "mean_linear_probe_average_precision": float(np.mean(all_aps)) if all_aps else None,
                "mean_linear_probe_f1": float(np.mean(all_f1s)) if all_f1s else None,
                "mean_contrast_ratio_cnr": float(np.mean(all_cnrs)) if all_cnrs else None,
                "mean_unsupervised_maha_auc": float(np.mean(all_maha_aucs)) if all_maha_aucs else None,
                "mean_unsupervised_maha_ap": float(np.mean(all_maha_aps)) if all_maha_aps else None,
                "mean_defect_iou_jaccard": float(np.mean(all_ious)) if all_ious else None,
                "mean_defect_dice_f1": float(np.mean(all_dices)) if all_dices else None,
            },
            "task2_depth_regression": {
                "mean_defect_only_r2": float(np.mean(all_defect_r2s)) if all_defect_r2s else None,
                "mean_depth_mae_mm": float(np.mean(all_maes)) if all_maes else None,
                "mean_depth_plate_r2": float(np.mean(all_r2s)) if all_r2s else None,
                "mean_hurdle_plate_r2": float(np.mean(all_hurdle_r2s)) if all_hurdle_r2s else None,
            },
            "task2b_flaw_size_sizing": {
                "mean_flaw_size_defect_r2": float(np.mean(all_size_defect_r2s)) if all_size_defect_r2s else None,
                "mean_flaw_size_defect_mae_mm": float(np.mean(all_size_maes)) if all_size_maes else None,
                "mean_flaw_size_plate_r2": float(np.mean(all_size_plate_r2s)) if all_size_plate_r2s else None,
            },
            "task4_latent_diagnostics": {
                "mean_channel_sensitivity": float(np.mean(all_sens)) if all_sens else None,
            },
        },
        "per_specimen_summary": per_specimen_summary,
        "per_file_results": file_results,
    }

    report_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "specimen", "sensor", "waveform", "liftoff",
            "task1_linear_auc", "task1_linear_ap", "task1_linear_f1",
            "task1_mlp_auc", "task1_delta_auc", "task1_cnr",
            "task1_defect_iou", "task1_defect_dice",
            "task2_depth_defect_r2", "task2_depth_mae_mm", "task2_depth_rmse_mm",
            "task2b_size_defect_r2", "task2b_size_mae_mm", "task2b_size_plate_r2",
            "task4_mean_channel_sensitivity", "task4_max_channel_sensitivity",
        ])
        for r in file_results:
            m = r.get("metrics", {})
            meta = r.get("metadata", {})
            writer.writerow([
                r.get("file_name"),
                r.get("specimen"),
                meta.get("sensor", ""),
                meta.get("waveform", ""),
                meta.get("liftoff", ""),
                f"{m.get('linear_probe_auc_roc', 0.0):.4f}" if m.get('linear_probe_auc_roc') is not None else "",
                f"{m.get('linear_probe_average_precision', 0.0):.4f}" if m.get('linear_probe_average_precision') is not None else "",
                f"{m.get('linear_probe_f1', 0.0):.4f}" if m.get('linear_probe_f1') is not None else "",
                f"{m.get('mlp_2layer_auc_roc', 0.0):.4f}" if m.get('mlp_2layer_auc_roc') is not None else "",
                f"{m.get('delta_auc', 0.0):.4f}" if m.get('delta_auc') is not None else "",
                f"{m.get('contrast_ratio_cnr', 0.0):.4f}" if m.get('contrast_ratio_cnr') is not None else "",
                f"{m.get('defect_iou_jaccard', 0.0):.4f}" if m.get('defect_iou_jaccard') is not None else "",
                f"{m.get('defect_dice_f1', 0.0):.4f}" if m.get('defect_dice_f1') is not None else "",
                f"{m.get('defect_only_r2', 0.0):.4f}" if m.get('defect_only_r2') is not None else "",
                f"{m.get('depth_mae_mm', 0.0):.4f}" if m.get('depth_mae_mm') is not None else "",
                f"{m.get('depth_rmse_mm', 0.0):.4f}" if m.get('depth_rmse_mm') is not None else "",
                f"{m.get('flaw_size_defect_r2', 0.0):.4f}" if m.get('flaw_size_defect_r2') is not None else "",
                f"{m.get('flaw_size_defect_mae_mm', 0.0):.4f}" if m.get('flaw_size_defect_mae_mm') is not None else "",
                f"{m.get('flaw_size_plate_r2', 0.0):.4f}" if m.get('flaw_size_plate_r2') is not None else "",
                f"{m.get('mean_latent_sensitivity', 0.0):.4f}" if m.get('mean_latent_sensitivity') is not None else "",
                f"{m.get('max_latent_sensitivity', 0.0):.4f}" if m.get('max_latent_sensitivity') is not None else "",
            ])

    return report


def main():
    args = build_arg_parser().parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, exp_name=args.exp_name)
    args.checkpoint = checkpoint_path

    if args.output_dir is None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(ckpt_dir)
        args.output_dir = os.path.join(parent_dir, "evaluation_results") if os.path.isdir(parent_dir) else "evaluation_results/5x5"
    args.output_dir = to_safe_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78)
    print("  5x5 SPATIOTEMPORAL PECT-JEPA COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 78)
    print(f"[Evaluation] Model Checkpoint: {checkpoint_path}")
    print(f"[Evaluation] Output Root Dir:  {args.output_dir}")
    print(f"[Evaluation] Target Splits:    {args.eval_splits}")
    print(f"[Evaluation] Target Modes:     {args.eval_modes}")

    model = load_model_from_checkpoint(checkpoint_path, device=args.device)
    crop_border = args.crop_border if args.crop_border is not None else getattr(model.config, "crop_border", 15)

    # 1. Resolve dataset partitions
    resolved_split_summary = resolve_split_summary_path(args.split_summary, checkpoint_path=checkpoint_path)
    train_files: List[str] = []
    val_files: List[str] = []
    test_files: List[str] = []
    protocol_name: str = "unknown"
    holdout_target: str = "none"

    if args.file and os.path.exists(args.file):
        splits_dict = {"single_file": [args.file]}
        protocol_name = "single_file"
        holdout_target = os.path.basename(args.file)
    elif resolved_split_summary and os.path.exists(resolved_split_summary):
        print(f"\n[Split] Loading partition sets from: {resolved_split_summary}")
        with open(resolved_split_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        train_files = summary.get("train_files", [])
        val_files = summary.get("val_files", [])
        test_files = summary.get("test_files", [])
        protocol_name = summary.get("protocol", "unknown")
        holdout_target = summary.get("holdout_target", "unknown")
        splits_dict = {
            "test": test_files,
            "val": val_files,
            "train": train_files,
        }
    else:
        all_files = find_all_tdms_files(args.data_dir)
        if not all_files:
            print(f"[Error] No TDMS files found in {args.data_dir}")
            sys.exit(1)
        train_files, val_files, test_files, summary = get_dataset_split(
            file_paths=all_files,
            protocol=args.split_protocol,
            holdout_target=args.holdout_target,
            holdout_liftoff=args.holdout_liftoff,
            holdout_sensor=args.holdout_sensor,
            holdout_waveform=args.holdout_waveform,
            val_ratio=0.1,
            seed=42,
        )
        protocol_name = args.split_protocol
        holdout_target = args.holdout_target
        splits_dict = {
            "test": test_files,
            "val": val_files,
            "train": train_files,
        }

    # Filter requested splits
    active_splits = [s for s in args.eval_splits if s in splits_dict and len(splits_dict[s]) > 0]
    if not active_splits:
        active_splits = list(splits_dict.keys())

    print(f"\nActive splits to benchmark: {active_splits}")
    for s in active_splits:
        print(f"  - Split '{s}': {len(splits_dict[s])} files")

    # 2. Fit base Zero-Shot readouts on train set pool
    feat_cache_dir = to_safe_path(os.path.join(args.output_dir, ".feat_cache"))
    os.makedirs(feat_cache_dir, exist_ok=True)

    base_readouts: Dict[str, Any] = {}
    train_features_cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    if train_files and len(train_files) > 0 and protocol_name != "single_file":
        base_readouts, train_features_cache = train_base_zero_shot_readouts(
            model=model,
            train_files=train_files,
            data_dir=args.data_dir,
            cache_dir=feat_cache_dir,
            batch_size=args.batch_size,
            device=args.device,
            crop_border=crop_border,
            eval_3d=args.eval_3d,
        )
    morphology_clf = base_readouts.get("morphology_clf")

    # 3. Master benchmark storage
    master_benchmark: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "protocol": protocol_name,
        "holdout_target": holdout_target,
        "evaluated_splits": active_splits,
        "within_file": {},
        "zero_shot": {},
        "comparison_table": [],
    }

    # 4. Loop across each partition
    for split_name in active_splits:
        file_list = splits_dict[split_name]
        if args.max_eval_files is not None and split_name == "test":
            file_list = file_list[:args.max_eval_files]

        print(f"\n{'=' * 78}")
        print(f"  PROCESSING PARTITION: {split_name.upper()} ({len(file_list)} files)")
        print(f"{'=' * 78}")

        wf_results: List[Dict[str, Any]] = []
        zs_results: List[Dict[str, Any]] = []
        wf_size_pairs: List[Any] = []
        zs_size_pairs: List[Any] = []

        split_wf_dir = os.path.join(args.output_dir, "Within_File", split_name)
        split_zs_dir = os.path.join(args.output_dir, "Zero_Shot", split_name)

        for idx, fp in enumerate(file_list):
            fname = os.path.basename(fp)
            print(f"\n[{split_name.upper()} {idx + 1}/{len(file_list)}] {fname}", flush=True)

            # Extract features once per file (or retrieve from train cache / disk cache)
            if fp in train_features_cache:
                feature_map, volume_3d = train_features_cache[fp]
            else:
                feature_map, volume_3d = get_or_extract_features(
                    file_path=fp,
                    model=model,
                    cache_dir=feat_cache_dir,
                    batch_size=args.batch_size,
                    device=args.device,
                    crop_border=crop_border,
                    eval_3d=args.eval_3d,
                )

            # Mode A: Within-File
            if "Within_File" in args.eval_modes:
                res_wf = evaluate_single_file_within_file(
                    file_path=fp,
                    feature_map=feature_map,
                    volume_3d=volume_3d,
                    output_dir=split_wf_dir,
                    model_config=model.config,
                    save_features=args.save_features,
                    crop_border=crop_border,
                    eval_3d=args.eval_3d,
                    eval_diagnostics=args.eval_diagnostics,
                    morphology_clf=morphology_clf,
                )
                m_wf = res_wf.get("metrics", {})
                auc_s = f"AUC: {m_wf['auc_roc']:.4f}" if m_wf.get("auc_roc") is not None else "AUC: N/A"
                cnr_s = f"CNR: {m_wf['contrast_ratio_cnr']:.2f}" if m_wf.get("contrast_ratio_cnr") is not None else ""
                r2_s = f"Depth-R²: {m_wf['defect_only_r2']:.3f}" if m_wf.get("defect_only_r2") is not None else ""
                sz_s = f"Size-R²: {m_wf['flaw_size_defect_r2']:.3f}" if m_wf.get("flaw_size_defect_r2") is not None else ""
                print(f"    [Within-File] {auc_s} | {cnr_s} | {r2_s} | {sz_s}", flush=True)

                if "size_pairs" in res_wf.get("task2b_flaw_size_sizing", {}):
                    wf_size_pairs.append(res_wf["task2b_flaw_size_sizing"]["size_pairs"])
                    del res_wf["task2b_flaw_size_sizing"]["size_pairs"]
                wf_results.append(res_wf)

            # Mode B: Zero-Shot
            if "Zero_Shot" in args.eval_modes and base_readouts:
                res_zs = evaluate_single_file_zeroshot(
                    file_path=fp,
                    feature_map=feature_map,
                    volume_3d=volume_3d,
                    base_readouts=base_readouts,
                    output_dir=split_zs_dir,
                    model_config=model.config,
                    save_features=args.save_features,
                    crop_border=crop_border,
                    eval_3d=args.eval_3d,
                    eval_diagnostics=args.eval_diagnostics,
                )
                m_zs = res_zs.get("metrics", {})
                auc_s = f"AUC: {m_zs['auc_roc']:.4f}" if m_zs.get("auc_roc") is not None else "AUC: N/A"
                cnr_s = f"CNR: {m_zs['contrast_ratio_cnr']:.2f}" if m_zs.get("contrast_ratio_cnr") is not None else ""
                r2_s = f"Depth-R²: {m_zs['defect_only_r2']:.3f}" if m_zs.get("defect_only_r2") is not None else ""
                sz_s = f"Size-R²: {m_zs['flaw_size_defect_r2']:.3f}" if m_zs.get("flaw_size_defect_r2") is not None else ""
                print(f"    [Zero-Shot]   {auc_s} | {cnr_s} | {r2_s} | {sz_s}", flush=True)

                if "size_pairs" in res_zs.get("task2b_flaw_size_sizing", {}):
                    zs_size_pairs.append(res_zs["task2b_flaw_size_sizing"]["size_pairs"])
                    del res_zs["task2b_flaw_size_sizing"]["size_pairs"]
                zs_results.append(res_zs)

            plt.close("all")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save partition summaries
        if "Within_File" in args.eval_modes and wf_results:
            report_wf = save_split_summaries(
                file_results=wf_results,
                output_dir=split_wf_dir,
                split_name=split_name,
                mode_name="Within_File",
                protocol_name=protocol_name,
                holdout_target=holdout_target,
                model=model,
                crop_border=crop_border,
                all_size_pairs=wf_size_pairs,
            )
            master_benchmark["within_file"][split_name] = report_wf["aggregate_metrics"]

        if "Zero_Shot" in args.eval_modes and zs_results:
            report_zs = save_split_summaries(
                file_results=zs_results,
                output_dir=split_zs_dir,
                split_name=split_name,
                mode_name="Zero_Shot",
                protocol_name=protocol_name,
                holdout_target=holdout_target,
                model=model,
                crop_border=crop_border,
                all_size_pairs=zs_size_pairs,
            )
            master_benchmark["zero_shot"][split_name] = report_zs["aggregate_metrics"]

        # Comparative row for master summary
        m_wf_agg = master_benchmark["within_file"].get(split_name, {})
        m_zs_agg = master_benchmark["zero_shot"].get(split_name, {})
        master_benchmark["comparison_table"].append({
            "split": split_name,
            "within_file": {
                "auc_roc": m_wf_agg.get("task1_anomaly_detection", {}).get("mean_linear_probe_auc_roc"),
                "ap": m_wf_agg.get("task1_anomaly_detection", {}).get("mean_linear_probe_average_precision"),
                "cnr": m_wf_agg.get("task1_anomaly_detection", {}).get("mean_contrast_ratio_cnr"),
                "defect_depth_r2": m_wf_agg.get("task2_depth_regression", {}).get("mean_defect_only_r2"),
                "depth_mae_mm": m_wf_agg.get("task2_depth_regression", {}).get("mean_depth_mae_mm"),
                "defect_size_r2": m_wf_agg.get("task2b_flaw_size_sizing", {}).get("mean_flaw_size_defect_r2"),
                "size_mae_mm": m_wf_agg.get("task2b_flaw_size_sizing", {}).get("mean_flaw_size_defect_mae_mm"),
                "defect_iou": m_wf_agg.get("task1_anomaly_detection", {}).get("mean_defect_iou_jaccard"),
            },
            "zero_shot": {
                "auc_roc": m_zs_agg.get("task1_anomaly_detection", {}).get("mean_linear_probe_auc_roc"),
                "ap": m_zs_agg.get("task1_anomaly_detection", {}).get("mean_linear_probe_average_precision"),
                "cnr": m_zs_agg.get("task1_anomaly_detection", {}).get("mean_contrast_ratio_cnr"),
                "defect_depth_r2": m_zs_agg.get("task2_depth_regression", {}).get("mean_defect_only_r2"),
                "depth_mae_mm": m_zs_agg.get("task2_depth_regression", {}).get("mean_depth_mae_mm"),
                "defect_size_r2": m_zs_agg.get("task2b_flaw_size_sizing", {}).get("mean_flaw_size_defect_r2"),
                "size_mae_mm": m_zs_agg.get("task2b_flaw_size_sizing", {}).get("mean_flaw_size_defect_mae_mm"),
                "defect_iou": m_zs_agg.get("task1_anomaly_detection", {}).get("mean_defect_iou_jaccard"),
            },
        })

    # 5. Save Root Master Benchmark Summaries
    master_json_path = os.path.join(args.output_dir, "master_benchmark_summary.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_benchmark, f, indent=2)

    master_csv_path = os.path.join(args.output_dir, "master_benchmark_summary.csv")
    with open(master_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "mode", "auc_roc", "ap", "cnr",
            "defect_depth_r2", "depth_mae_mm",
            "defect_size_r2", "size_mae_mm", "defect_iou",
        ])
        for row in master_benchmark["comparison_table"]:
            sp = row["split"]
            wf = row["within_file"]
            zs = row["zero_shot"]
            writer.writerow([
                sp, "Within_File",
                f"{wf.get('auc_roc', 0.0):.4f}" if wf.get('auc_roc') is not None else "",
                f"{wf.get('ap', 0.0):.4f}" if wf.get('ap') is not None else "",
                f"{wf.get('cnr', 0.0):.2f}" if wf.get('cnr') is not None else "",
                f"{wf.get('defect_depth_r2', 0.0):.4f}" if wf.get('defect_depth_r2') is not None else "",
                f"{wf.get('depth_mae_mm', 0.0):.4f}" if wf.get('depth_mae_mm') is not None else "",
                f"{wf.get('defect_size_r2', 0.0):.4f}" if wf.get('defect_size_r2') is not None else "",
                f"{wf.get('size_mae_mm', 0.0):.4f}" if wf.get('size_mae_mm') is not None else "",
                f"{wf.get('defect_iou', 0.0):.4f}" if wf.get('defect_iou') is not None else "",
            ])
            writer.writerow([
                sp, "Zero_Shot",
                f"{zs.get('auc_roc', 0.0):.4f}" if zs.get('auc_roc') is not None else "",
                f"{zs.get('ap', 0.0):.4f}" if zs.get('ap') is not None else "",
                f"{zs.get('cnr', 0.0):.2f}" if zs.get('cnr') is not None else "",
                f"{zs.get('defect_depth_r2', 0.0):.4f}" if zs.get('defect_depth_r2') is not None else "",
                f"{zs.get('depth_mae_mm', 0.0):.4f}" if zs.get('depth_mae_mm') is not None else "",
                f"{zs.get('defect_size_r2', 0.0):.4f}" if zs.get('defect_size_r2') is not None else "",
                f"{zs.get('size_mae_mm', 0.0):.4f}" if zs.get('size_mae_mm') is not None else "",
                f"{zs.get('defect_iou', 0.0):.4f}" if zs.get('defect_iou') is not None else "",
            ])

    # Backward compatibility: copy Within_File/test to root if available
    wf_test_json = os.path.join(args.output_dir, "Within_File", "test", "evaluation_summary.json")
    wf_test_csv = os.path.join(args.output_dir, "Within_File", "test", "evaluation_summary.csv")
    if os.path.isfile(wf_test_json):
        shutil.copyfile(wf_test_json, os.path.join(args.output_dir, "evaluation_summary.json"))
    if os.path.isfile(wf_test_csv):
        shutil.copyfile(wf_test_csv, os.path.join(args.output_dir, "evaluation_summary.csv"))

    # Print Master Summary Table
    print("\n" + "=" * 90)
    print("  MASTER BENCHMARK COMPARISON TABLE: WITHIN-FILE vs ZERO-SHOT ACROSS SPLITS")
    print("=" * 90)
    print(f"{'Split':<8} | {'Mode':<12} | {'AUC-ROC':<8} | {'AP':<8} | {'CNR':<6} | {'Depth R²':<9} | {'Depth MAE':<10} | {'Size R²':<8} | {'IoU':<6}")
    print("-" * 90)
    for row in master_benchmark["comparison_table"]:
        sp = row["split"]
        wf = row["within_file"]
        zs = row["zero_shot"]
        print(f"{sp:<8} | {'Within-File':<12} | "
              f"{fmt_val(wf.get('auc_roc')):<8} | "
              f"{fmt_val(wf.get('ap')):<8} | "
              f"{fmt_val(wf.get('cnr'), 2):<6} | "
              f"{fmt_val(wf.get('defect_depth_r2')):<9} | "
              f"{fmt_val(wf.get('depth_mae_mm')):<10} | "
              f"{fmt_val(wf.get('defect_size_r2')):<8} | "
              f"{fmt_val(wf.get('defect_iou')):<6}")
        print(f"{sp:<8} | {'Zero-Shot':<12} | "
              f"{fmt_val(zs.get('auc_roc')):<8} | "
              f"{fmt_val(zs.get('ap')):<8} | "
              f"{fmt_val(zs.get('cnr'), 2):<6} | "
              f"{fmt_val(zs.get('defect_depth_r2')):<9} | "
              f"{fmt_val(zs.get('depth_mae_mm')):<10} | "
              f"{fmt_val(zs.get('defect_size_r2')):<8} | "
              f"{fmt_val(zs.get('defect_iou')):<6}")
        print("-" * 90)

    print(f"\nSaved Master Benchmark JSON: {master_json_path}")
    print(f"Saved Master Benchmark CSV:  {master_csv_path}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
