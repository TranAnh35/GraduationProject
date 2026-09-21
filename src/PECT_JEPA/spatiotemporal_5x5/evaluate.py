"""
CLI Evaluation and Downstream Probing Suite for 5x5 Spatiotemporal PECT-JEPA.

Modular Downstream Benchmark Architecture:
- Task 1: Anomaly Detection (1_Anomaly_Detection/<specimen>/)
  Defect probability heatmaps P(Y=1|z), ROC & Precision-Recall curves,
  Linear Probe vs MLP 2-Layer benchmarks, True CNR, peak CNR, Representation Gap.
- Task 2: Quantitative Depth Regression (2_Depth_Regression/<specimen>/)
  Physical depth sizing in mm, 2D predicted depth maps, calibration scatter plots,
  Linear Ridge vs MLP Regressor (R^2, MAE mm, RMSE mm).
- Task 3: Defect Severity Classification (3_Severity_Classification/<specimen>/)
  4-class depth binning (Sound, Shallow <=0.2mm, Medium 0.3-0.6mm, Severe >=0.7mm),
  normalized confusion matrices, Macro F1 and Accuracy.
- Task 4: Lift-off Invariance (4_Liftoff_Invariance/)
  Linear CKA and Cosine Similarity across paired lift-offs (z1 vs z2 vs z3),
  CKA invariance matrix heatmap and correlation analysis.
- Task 5: Representation Geometry (5_Representation_Geometry/<specimen>/)
  Latent PCA-RGB projection, Hypersphere Angular Distance (1 - cos theta),
  top-3 PCs variance explained, angular CNR and AUC.
- Root:
  Consolidated evaluation_summary.json and tabular evaluation_summary.csv.

Usage:
    # Evaluate held-out domain from split summary:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --checkpoint checkpoints/pect_jepa_5x5/best_model_5x5.pt \\
        --split_summary checkpoints/pect_jepa_5x5/pect_jepa_5x5_base_split_summary.json

    # Dynamic Compound OOD:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --checkpoint checkpoints/pect_jepa_5x5/best_model_5x5.pt \\
        --split_protocol compound_ood

    # Evaluate single TDMS file:
    python -m src.PECT_JEPA.spatiotemporal_5x5.evaluate \\
        --checkpoint checkpoints/pect_jepa_5x5/best_model_5x5.pt \\
        --file data/TMR/Corrosion/Square/tmr_corosion_frontside_square_300x300_z1_20260126_190655.tdms
"""

import argparse
import csv
import json
import os
import sys
import types
from typing import List, Dict, Any, Optional, Tuple

# Defensive safeguard for HPC clusters where torch._dynamo has broken imports or NumPy 2.x conflicts
try:
    import torch._dynamo
except Exception:
    fake_dynamo = types.ModuleType("torch._dynamo")
    fake_dynamo.disable = lambda fn=None, *args, **kwargs: (fn if fn is not None else (lambda f: f))
    sys.modules["torch._dynamo"] = fake_dynamo

import warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import StratifiedKFold, KFold
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
    plot_anomaly_heatmap_5x5,
    plot_latent_representation_quality,
    compute_anomaly_metrics,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.linear_probe import LinearProbeEvaluator
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.downstream_benchmarks import DownstreamBenchmarkSuite
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.liftoff_invariance import (
    compute_linear_cka,
    compute_feature_similarity_matrix,
)
from src.PECT_JEPA.spatiotemporal_5x5.evaluation.visualizations import (
    plot_probability_heatmap,
    plot_roc_pr_curves,
    plot_depth_regression_maps,
    plot_depth_calibration_scatter,
    plot_severity_confusion_matrix,
    plot_liftoff_cka_heatmap,
)


def find_ground_truth_mask(file_path: str, data_dir: str = "data") -> Optional[np.ndarray]:
    """
    Finds and loads the authoritative CAD ground-truth mask corresponding to a TDMS file's specimen.
    Utilizes GroundTruthManager to ensure mathematical alignment with specimen_mask_features.json.
    """
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
    p.add_argument("--eval_liftoff", action="store_true", default=True, help="Compute Linear CKA across lift-off variations")
    p.add_argument("--no_eval_liftoff", dest="eval_liftoff", action="store_false", help="Skip lift-off invariance analysis")
    p.add_argument("--save_features", action="store_true", default=False, help="Save extracted .npy feature maps to disk")
    p.add_argument("--crop_border", type=int, default=15, help="Outer boundary pixels cropped to remove air/edge effect")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for sliding window feature extraction")
    p.add_argument("--device", type=str, default="cuda", help="Target device: 'cuda' or 'cpu'")
    p.add_argument("--max_eval_files", type=int, default=None, help="Optional limit on number of test files to evaluate")
    return p


def resolve_checkpoint_path(checkpoint_path: Optional[str] = None, exp_name: Optional[str] = None) -> str:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    search_base = "experiments/5x5"
    target_key = (checkpoint_path or "best").strip().lower()

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

    # Fallback to standard locations
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
        # Check adjacent directory for config_5x5.json or config.json
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

    # If tokenizer_type was not specified in checkpoint config, infer from state_dict
    if not (isinstance(cfg_dict, dict) and "tokenizer_type" in cfg_dict):
        if "tokenizer.cross_domain_attn.in_proj_weight" in state_dict or "tokenizer.fuse_proj.weight" in state_dict:
            config.tokenizer_type = "dual_domain_attention"
        elif "tokenizer.time_proj.weight" in state_dict:
            config.tokenizer_type = "dual_domain"
        elif "tokenizer.proj.weight" in state_dict:
            config.tokenizer_type = "time_only"

    # If predictor_type was not specified in checkpoint config, infer from state_dict
    if not (isinstance(cfg_dict, dict) and "predictor_type" in cfg_dict):
        if "predictor.op_embedding.default_op" in state_dict:
            config.predictor_type = "operator_diffusion"
        else:
            config.predictor_type = "standard"

    # Infer embed_dim from encoder positional embedding if needed
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
    print(f"  Model config: tokenizer_type={config.tokenizer_type}, predictor_type={config.predictor_type}, embed_dim={config.embed_dim}, C={config.in_channels}")
    return model


def evaluate_single_file(
    file_path: str,
    model: PECT_JEPA_5x5,
    output_dir: str,
    batch_size: int = 512,
    device: str = "cuda",
    save_features: bool = False,
    crop_border: int = 15,
) -> Dict[str, Any]:
    """
    Runs full C-scan feature extraction and executes the 4 single-file downstream benchmark tasks:
    - Task 1: Anomaly Detection (1_Anomaly_Detection/<specimen>/)
    - Task 2: Quantitative Depth Regression (2_Depth_Regression/<specimen>/)
    - Task 3: Defect Severity Classification (3_Severity_Classification/<specimen>/)
    - Task 5: Representation Geometry (5_Representation_Geometry/<specimen>/)
    """
    fname_base = os.path.splitext(os.path.basename(file_path))[0]
    meta = extract_file_metadata(file_path)
    gt_mgr = get_ground_truth_manager(data_dir=getattr(model.config, "data_dir", "data"))
    specimen_key = gt_mgr.canonical_specimen_key(file_path)

    # 1. Modular Directory Paths
    task1_dir = os.path.join(output_dir, "1_Anomaly_Detection", specimen_key)
    task2_dir = os.path.join(output_dir, "2_Depth_Regression", specimen_key)
    task3_dir = os.path.join(output_dir, "3_Severity_Classification", specimen_key)
    task5_dir = os.path.join(output_dir, "5_Representation_Geometry", specimen_key)
    for d in [task1_dir, task2_dir, task3_dir, task5_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n--- Extracting C-Scan Features: {os.path.basename(file_path)} ---")
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

    feature_map = extract_full_cscan_map(
        model=model,
        full_cscan_3d=grid_3d,
        batch_size=batch_size,
        device=device,
        show_pbar=False,
    )

    # Load Ground Truth representations
    gt_mask = gt_mgr.get_ground_truth_mask_for_file(file_path, aligned_scan=True)
    depth_map_gt = gt_mgr.generate_depth_map(specimen_key)
    severity_mask_gt = gt_mgr.generate_severity_mask(specimen_key)

    has_gt = gt_mask is not None
    min_Y = min(feature_map.shape[0], gt_mask.shape[0]) if has_gt else feature_map.shape[0]
    min_X = min(feature_map.shape[1], gt_mask.shape[1]) if has_gt else feature_map.shape[1]

    sub_feat = feature_map[:min_Y, :min_X]
    sub_gt = gt_mask[:min_Y, :min_X] if has_gt else None
    sub_depth = depth_map_gt[:min_Y, :min_X] if depth_map_gt is not None else None
    sub_sev = severity_mask_gt[:min_Y, :min_X] if severity_mask_gt is not None else None

    # =========================================================================
    # Task 1: Anomaly Detection (Linear Probe + MLP 2-Layer + Probability Heatmap)
    # =========================================================================
    task1_res: Dict[str, Any] = {}
    prob_heatmap_path = None
    roc_pr_path = None

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

            # 1. Defect Probability Heatmap
            prob_heatmap_path = os.path.join(task1_dir, f"{fname_base}_prob_heatmap.png")
            plot_probability_heatmap(
                prob_map=prob_map,
                save_path=prob_heatmap_path,
                title=f"Defect Probability Map | {meta.get('specimen', '')} - {meta.get('sensor', '')}\n"
                      f"Lift-off: {meta.get('liftoff', '')} | Waveform: {meta.get('waveform', '')} | AUC: {lp_auc:.4f} | CNR: {cnr:.2f}",
            )

            # 2. ROC & PR Curves
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
            task1_res["prob_heatmap_path"] = prob_heatmap_path
            task1_res["roc_pr_path"] = roc_pr_path

        except Exception as e:
            print(f"    [Task 1 Warning] Anomaly detection failed: {e}")
            task1_res = {"error": str(e)}

    # =========================================================================
    # Task 2: Quantitative Depth Regression (Linear Ridge vs MLP Regressor)
    # =========================================================================
    task2_res: Dict[str, Any] = {}
    pred_depth_map_path = None
    depth_scatter_path = None

    if has_gt and sub_depth is not None:
        try:
            bench = DownstreamBenchmarkSuite(n_splits=5, random_state=42)
            reg_benchmark = bench.benchmark_depth_regression(sub_feat, sub_depth, focus_defects_only=False)

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

            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_fit_s, y_fit)

            pred_depth_flat = ridge.predict(scaler.transform(flat_feats))
            pred_depth_map = np.clip(pred_depth_flat.reshape(min_Y, min_X), 0.0, None)

            lp_reg = reg_benchmark.get("linear_probe", {})
            mlp_reg = reg_benchmark.get("mlp_2layer", {})
            r2_val = lp_reg.get("r2_score", 0.0)
            mae_val = lp_reg.get("mae_mm", 0.0)
            rmse_val = lp_reg.get("rmse_mm", 0.0)

            pred_depth_map_path = os.path.join(task2_dir, f"{fname_base}_predicted_depth_map.png")
            depth_scatter_path = os.path.join(task2_dir, f"{fname_base}_depth_scatter.png")

            plot_depth_regression_maps(
                true_depth_map=sub_depth,
                pred_depth_map=pred_depth_map,
                save_path=pred_depth_map_path,
                title=f"Quantitative Depth Sizing | {meta.get('specimen', '')} - {meta.get('sensor', '')}\n"
                      f"Waveform: {meta.get('waveform', '')} | Lift-off: {meta.get('liftoff', '')}",
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
                "pred_depth_map_path": pred_depth_map_path,
                "depth_scatter_path": depth_scatter_path,
            }
        except Exception as e:
            print(f"    [Task 2 Warning] Depth regression failed: {e}")
            task2_res = {"error": str(e)}

    # =========================================================================
    # Task 3: Defect Severity Classification (4-Class Depth Bins)
    # =========================================================================
    task3_res: Dict[str, Any] = {}
    cm_path = None

    if has_gt and sub_sev is not None:
        try:
            bench = DownstreamBenchmarkSuite(n_splits=5, random_state=42)
            sev_benchmark = bench.benchmark_severity_classification(sub_feat, sub_sev)

            valid_idx = np.where(sub_sev.reshape(-1) >= 0)[0]
            X_v = sub_feat.reshape(-1, sub_feat.shape[-1])[valid_idx].astype(np.float32)
            y_v = sub_sev.reshape(-1)[valid_idx].astype(np.int64)

            c0 = np.where(y_v == 0)[0]
            c_other = np.where(y_v > 0)[0]
            if len(c0) > 8000:
                rng = np.random.RandomState(42)
                sub_c0 = rng.choice(c0, size=8000, replace=False)
                eval_idx = np.concatenate([c_other, sub_c0])
                X_v = X_v[eval_idx]
                y_v = y_v[eval_idx]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_true_all, y_pred_all = [], []
            for tr, te in skf.split(X_v, y_v):
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_v[tr])
                X_te_s = sc.transform(X_v[te])
                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, tol=1e-3, random_state=42, solver="lbfgs")
                lr.fit(X_tr_s, y_v[tr])
                preds = lr.predict(X_te_s)
                y_true_all.append(y_v[te])
                y_pred_all.append(preds)

            cm_y_true = np.concatenate(y_true_all)
            cm_y_pred = np.concatenate(y_pred_all)

            cm_path = os.path.join(task3_dir, f"{fname_base}_confusion_matrix.png")
            lp_sev = sev_benchmark.get("linear_probe", {})
            mlp_sev = sev_benchmark.get("mlp_2layer", {})
            lp_f1 = lp_sev.get("macro_f1", 0.0)
            lp_acc = lp_sev.get("accuracy", 0.0)

            plot_severity_confusion_matrix(
                y_true=cm_y_true,
                y_pred=cm_y_pred,
                save_path=cm_path,
                title=f"Severity Classification | {meta.get('specimen', '')} - {meta.get('sensor', '')}",
                macro_f1=lp_f1,
                accuracy=lp_acc,
            )

            task3_res = {
                "linear_probe": lp_sev,
                "mlp_2layer": mlp_sev,
                "representation_gap": sev_benchmark.get("representation_gap", {}),
                "confusion_matrix_path": cm_path,
            }
        except Exception as e:
            print(f"    [Task 3 Warning] Severity classification failed: {e}")
            task3_res = {"error": str(e)}

    # =========================================================================
    # Task 5: Representation Geometry (PCA-RGB + Angular Distance + CAD Overlay)
    # =========================================================================
    latent_geom_path = os.path.join(task5_dir, f"{fname_base}_latent_geometry.png")
    lq_dict = plot_latent_representation_quality(
        feature_map=feature_map,
        gt_mask=gt_mask,
        save_path=latent_geom_path,
        title_prefix=f"{meta.get('specimen', '')} - {meta.get('sensor', '')} ({meta.get('liftoff', '')})",
        close_fig=True,
    )

    if save_features:
        feat_path = os.path.join(output_dir, f"{fname_base}_features_5x5.npy")
        np.save(feat_path, feature_map)

    # Consolidated flat metrics for summary tables & CSV
    t1_lp = task1_res.get("linear_probe", {})
    t1_mlp = task1_res.get("mlp_2layer", {})
    t2_lp = task2_res.get("linear_probe", {})
    t3_lp = task3_res.get("linear_probe", {})

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
        "depth_mae_mm": t2_lp.get("mae_mm"),
        "depth_rmse_mm": t2_lp.get("rmse_mm"),
        "severity_macro_f1": t3_lp.get("macro_f1"),
        "severity_accuracy": t3_lp.get("accuracy"),
        "knn_5_accuracy": task1_res.get("knn_5_accuracy"),
    }

    result = {
        "file": file_path,
        "file_name": os.path.basename(file_path),
        "specimen": specimen_key,
        "metadata": meta,
        "metrics": metrics_flat,
        "task1_anomaly_detection": task1_res,
        "task2_depth_regression": task2_res,
        "task3_severity_classification": task3_res,
        "task5_representation_geometry": {
            "latent_quality": lq_dict,
            "latent_geometry_path": latent_geom_path,
        },
    }

    auc_str = f" | AUC: {metrics_flat['auc_roc']:.4f} | AP: {metrics_flat['average_precision']:.4f}" if metrics_flat.get("auc_roc") is not None else ""
    cnr_str = f" | CNR: {metrics_flat['contrast_ratio_cnr']:.2f}" if metrics_flat.get("contrast_ratio_cnr") is not None else ""
    r2_str = f" | R²: {metrics_flat['depth_r2']:.3f}" if metrics_flat.get("depth_r2") is not None else ""
    f1_str = f" | Sev-F1: {metrics_flat['severity_macro_f1']:.3f}" if metrics_flat.get("severity_macro_f1") is not None else ""
    print(f"  [Result]{cnr_str}{auc_str}{r2_str}{f1_str}")
    return result


def evaluate_liftoff_invariance(
    file_paths: List[str],
    model: PECT_JEPA_5x5,
    output_dir: str,
    batch_size: int = 512,
    device: str = "cuda",
    crop_border: int = 15,
) -> Dict[str, Any]:
    """
    Computes Linear CKA and Cosine Similarity across lift-off variations (z1 vs z2 vs z3)
    and plots the lift-off CKA matrix in 4_Liftoff_Invariance/.
    """
    print("\n" + "=" * 70)
    print("  RUNNING MULTI-LIFT-OFF INVARIANCE ANALYSIS (Linear CKA & Cosine Sim)")
    print("=" * 70)

    task4_dir = os.path.join(output_dir, "4_Liftoff_Invariance")
    os.makedirs(task4_dir, exist_ok=True)

    meta_by_fp = {fp: extract_file_metadata(fp) for fp in file_paths}
    groups: Dict[tuple, Dict[str, str]] = {}
    for fp, m in meta_by_fp.items():
        key = (m.get("specimen"), m.get("sensor"), m.get("waveform"))
        lo = m.get("liftoff")
        if lo:
            groups.setdefault(key, {})[lo] = fp

    liftoff_results = []
    pairwise_ckas: Dict[Tuple[str, str], List[float]] = {}

    # Select representative triplets (1 per specimen: corrosion, rivet_v1, rivet_v2) for fast & balanced evaluation
    selected_groups = {}
    for key, lo_files in groups.items():
        if len(lo_files) >= 2:
            sp = key[0] or "unknown"
            if sp not in selected_groups:
                selected_groups[sp] = (key, lo_files)
            if len(selected_groups) >= 3:
                break

    for sp, (key, lo_files) in selected_groups.items():
        liftoff_keys = sorted(lo_files.keys())
        if len(liftoff_keys) >= 2:
            specimen, sensor, waveform = key
            print(f"\nEvaluating Lift-off Invariance: Specimen={specimen}, Sensor={sensor}, Waveform={waveform}", flush=True)
            print(f"  Available lift-off levels: {liftoff_keys}", flush=True)

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

            for i in range(len(liftoff_keys)):
                for j in range(i + 1, len(liftoff_keys)):
                    lo_a = liftoff_keys[i]
                    lo_b = liftoff_keys[j]
                    fa = features_by_lo[lo_a]
                    fb = features_by_lo[lo_b]

                    n_sub = min(10000, fa.shape[0])
                    sub_idx = np.random.RandomState(42).choice(fa.shape[0], size=n_sub, replace=False)

                    cka = float(compute_linear_cka(fa[sub_idx], fb[sub_idx]))
                    cos_sim = float(compute_feature_similarity_matrix(fa[sub_idx], fb[sub_idx]))

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
                    pair_tuple = tuple(sorted([lo_a, lo_b]))
                    pairwise_ckas.setdefault(pair_tuple, []).append(cka)

    # 3x3 Lift-off Invariance Matrix Heatmap (z1, z2, z3)
    lo_levels = ["z1", "z2", "z3"]
    cka_matrix = np.eye(len(lo_levels), dtype=np.float32)
    for i, la in enumerate(lo_levels):
        for j, lb in enumerate(lo_levels):
            if i != j:
                pt = tuple(sorted([la, lb]))
                if pt in pairwise_ckas and pairwise_ckas[pt]:
                    cka_matrix[i, j] = float(np.mean(pairwise_ckas[pt]))

    cka_matrix_plot_path = os.path.join(task4_dir, "liftoff_cka_matrix.png")
    plot_liftoff_cka_heatmap(
        cka_matrix=cka_matrix,
        labels=lo_levels,
        save_path=cka_matrix_plot_path,
        title="5x5 PECT-JEPA Multi-Lift-Off Linear CKA Invariance",
    )

    mean_cka = float(np.mean([r["linear_cka"] for r in liftoff_results])) if liftoff_results else None
    mean_cos = float(np.mean([r["cosine_similarity"] for r in liftoff_results])) if liftoff_results else None

    summary = {
        "total_pairs_evaluated": len(liftoff_results),
        "mean_linear_cka": mean_cka,
        "mean_cosine_similarity": mean_cos,
        "liftoff_levels": lo_levels,
        "cka_matrix": cka_matrix.tolist(),
        "cka_matrix_plot_path": cka_matrix_plot_path,
        "pairwise_results": liftoff_results,
    }

    lo_json_path = os.path.join(task4_dir, "liftoff_invariance_summary.json")
    with open(lo_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved lift-off invariance artifacts to: {task4_dir}")
    return summary


def main():
    args = build_arg_parser().parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, exp_name=args.exp_name)
    args.checkpoint = checkpoint_path

    if args.output_dir is None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(ckpt_dir)
        args.output_dir = os.path.join(parent_dir, "evaluation_results") if os.path.isdir(parent_dir) else "evaluation_results/5x5"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  5x5 SPATIOTEMPORAL PECT-JEPA MODULAR EVALUATION SUITE")
    print("=" * 70)
    print(f"[Evaluation] Model Checkpoint: {checkpoint_path}")
    print(f"[Evaluation] Output Directory: {args.output_dir}")

    model = load_model_from_checkpoint(checkpoint_path, device=args.device)

    # 1. Determine evaluation test file set
    test_files: List[str] = []
    test_slices: Dict[str, List[str]] = {}
    protocol_name: str = "custom"
    holdout_target: str = "none"

    resolved_split_summary = resolve_split_summary_path(args.split_summary, checkpoint_path=checkpoint_path)

    if args.file and os.path.exists(args.file):
        test_files = [args.file]
        protocol_name = "single_file"
        holdout_target = os.path.basename(args.file)
        print(f"Single file evaluation mode: {args.file}")

    elif resolved_split_summary and os.path.exists(resolved_split_summary):
        print(f"Loading evaluation test partition from split summary: {resolved_split_summary}")
        with open(resolved_split_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        test_files = summary.get("test_files", [])
        test_slices = summary.get("test_slices", {})
        protocol_name = summary.get("protocol", "unknown")
        holdout_target = summary.get("holdout_target", "unknown")
        print(f"  Loaded {len(test_files)} held-out test files for protocol '{protocol_name}' ({holdout_target})")

    else:
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

    crop_border = args.crop_border if args.crop_border is not None else getattr(model.config, "crop_border", 15)

    # 2. Evaluate each test file for Tasks 1, 2, 3, 5
    file_results = []
    print(f"\n--- Evaluating {len(test_files)} Test Scans across 5 Benchmark Tasks ---")
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

    # 3. Task 4: Lift-off Invariance Analysis (run by default if multiple lift-offs exist)
    liftoff_summary = {}
    if args.eval_liftoff:
        all_pool = find_all_tdms_files(args.data_dir)
        try:
            liftoff_summary = evaluate_liftoff_invariance(
                file_paths=all_pool,
                model=model,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                device=args.device,
                crop_border=crop_border,
            )
        except Exception as e:
            print(f"  [Task 4 Warning] Lift-off invariance analysis failed: {e}")

    # =========================================================================
    # 4. Generate Dedicated Per-Task Summaries (1_, 2_, 3_, 5_)
    # =========================================================================
    specimen_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in file_results:
        sp = r.get("specimen", "unknown")
        specimen_groups.setdefault(sp, []).append(r)

    # Task 1: Anomaly Detection per specimen
    for sp, sp_results in specimen_groups.items():
        t1_sp_dir = os.path.join(args.output_dir, "1_Anomaly_Detection", sp)
        aucs = [r["metrics"]["linear_probe_auc_roc"] for r in sp_results if r["metrics"].get("linear_probe_auc_roc") is not None]
        aps = [r["metrics"]["linear_probe_average_precision"] for r in sp_results if r["metrics"].get("linear_probe_average_precision") is not None]
        f1s = [r["metrics"]["linear_probe_f1"] for r in sp_results if r["metrics"].get("linear_probe_f1") is not None]
        cnrs = [r["metrics"]["contrast_ratio_cnr"] for r in sp_results if r["metrics"].get("contrast_ratio_cnr") is not None]
        mlp_aucs = [r["metrics"]["mlp_2layer_auc_roc"] for r in sp_results if r["metrics"].get("mlp_2layer_auc_roc") is not None]
        delta_aucs = [r["metrics"]["delta_auc"] for r in sp_results if r["metrics"].get("delta_auc") is not None]

        sp_t1_summary = {
            "specimen": sp,
            "total_files": len(sp_results),
            "labeled_files": len(aucs),
            "linear_probe": {
                "mean_auc_roc": float(np.mean(aucs)) if aucs else None,
                "std_auc_roc": float(np.std(aucs)) if aucs else None,
                "mean_average_precision": float(np.mean(aps)) if aps else None,
                "mean_f1": float(np.mean(f1s)) if f1s else None,
            },
            "mlp_2layer": {
                "mean_auc_roc": float(np.mean(mlp_aucs)) if mlp_aucs else None,
            },
            "representation_gap": {
                "mean_delta_auc": float(np.mean(delta_aucs)) if delta_aucs else None,
            },
            "mean_contrast_ratio_cnr": float(np.mean(cnrs)) if cnrs else None,
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
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t1_sp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t1_summary, f, indent=2)

    # Task 2: Depth Regression per specimen
    for sp, sp_results in specimen_groups.items():
        t2_sp_dir = os.path.join(args.output_dir, "2_Depth_Regression", sp)
        r2s = [r["metrics"]["depth_r2"] for r in sp_results if r["metrics"].get("depth_r2") is not None]
        maes = [r["metrics"]["depth_mae_mm"] for r in sp_results if r["metrics"].get("depth_mae_mm") is not None]
        rmses = [r["metrics"]["depth_rmse_mm"] for r in sp_results if r["metrics"].get("depth_rmse_mm") is not None]

        sp_t2_summary = {
            "specimen": sp,
            "total_files": len(sp_results),
            "evaluated_files": len(r2s),
            "linear_probe": {
                "mean_r2_score": float(np.mean(r2s)) if r2s else None,
                "mean_mae_mm": float(np.mean(maes)) if maes else None,
                "mean_rmse_mm": float(np.mean(rmses)) if rmses else None,
            },
            "files": [
                {
                    "file_name": r["file_name"],
                    "sensor": r["metadata"].get("sensor"),
                    "waveform": r["metadata"].get("waveform"),
                    "liftoff": r["metadata"].get("liftoff"),
                    "r2_score": r["metrics"].get("depth_r2"),
                    "mae_mm": r["metrics"].get("depth_mae_mm"),
                    "rmse_mm": r["metrics"].get("depth_rmse_mm"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t2_sp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t2_summary, f, indent=2)

    # Task 3: Severity Classification per specimen
    for sp, sp_results in specimen_groups.items():
        t3_sp_dir = os.path.join(args.output_dir, "3_Severity_Classification", sp)
        f1s = [r["metrics"]["severity_macro_f1"] for r in sp_results if r["metrics"].get("severity_macro_f1") is not None]
        accs = [r["metrics"]["severity_accuracy"] for r in sp_results if r["metrics"].get("severity_accuracy") is not None]

        sp_t3_summary = {
            "specimen": sp,
            "total_files": len(sp_results),
            "evaluated_files": len(f1s),
            "linear_probe": {
                "mean_macro_f1": float(np.mean(f1s)) if f1s else None,
                "mean_accuracy": float(np.mean(accs)) if accs else None,
            },
            "files": [
                {
                    "file_name": r["file_name"],
                    "sensor": r["metadata"].get("sensor"),
                    "waveform": r["metadata"].get("waveform"),
                    "liftoff": r["metadata"].get("liftoff"),
                    "macro_f1": r["metrics"].get("severity_macro_f1"),
                    "accuracy": r["metrics"].get("severity_accuracy"),
                }
                for r in sp_results
            ],
        }
        with open(os.path.join(t3_sp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sp_t3_summary, f, indent=2)

    # Task 5: Representation Geometry per specimen & overall
    geom_data = []
    for r in file_results:
        lq = r.get("task5_representation_geometry", {}).get("latent_quality", {})
        if lq:
            geom_data.append({
                "file": r.get("file_name"),
                "specimen": r.get("specimen"),
                "pca_variance_explained": lq.get("pca_variance_explained"),
                "total_3pc_variance": lq.get("total_3pc_variance"),
                "angular_cnr": lq.get("angular_cnr"),
                "angular_auc": lq.get("angular_auc"),
                "angular_ap": lq.get("angular_ap"),
            })
    geom_dir = os.path.join(args.output_dir, "5_Representation_Geometry")
    geom_summary = {
        "files_count": len(geom_data),
        "mean_total_3pc_variance": float(np.mean([g["total_3pc_variance"] for g in geom_data if g.get("total_3pc_variance") is not None])) if any(g.get("total_3pc_variance") is not None for g in geom_data) else None,
        "mean_angular_cnr": float(np.mean([g["angular_cnr"] for g in geom_data if g.get("angular_cnr") is not None and not np.isnan(g["angular_cnr"])])) if any(g.get("angular_cnr") is not None for g in geom_data) else None,
        "mean_angular_auc": float(np.mean([g["angular_auc"] for g in geom_data if g.get("angular_auc") is not None and not np.isnan(g["angular_auc"])])) if any(g.get("angular_auc") is not None for g in geom_data) else None,
        "per_file": geom_data,
    }
    with open(os.path.join(geom_dir, "geometry_summary.json"), "w", encoding="utf-8") as f:
        json.dump(geom_summary, f, indent=2)

    # =========================================================================
    # 5. Global Consolidated Evaluation Report & CSV
    # =========================================================================
    all_cnrs = [r["metrics"]["contrast_ratio_cnr"] for r in file_results if r["metrics"].get("contrast_ratio_cnr") is not None]
    all_aucs = [r["metrics"]["linear_probe_auc_roc"] for r in file_results if r["metrics"].get("linear_probe_auc_roc") is not None]
    all_aps = [r["metrics"]["linear_probe_average_precision"] for r in file_results if r["metrics"].get("linear_probe_average_precision") is not None]
    all_f1s = [r["metrics"]["linear_probe_f1"] for r in file_results if r["metrics"].get("linear_probe_f1") is not None]
    all_r2s = [r["metrics"]["depth_r2"] for r in file_results if r["metrics"].get("depth_r2") is not None]
    all_maes = [r["metrics"]["depth_mae_mm"] for r in file_results if r["metrics"].get("depth_mae_mm") is not None]
    all_sev_f1s = [r["metrics"]["severity_macro_f1"] for r in file_results if r["metrics"].get("severity_macro_f1") is not None]

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
            "crop_border": crop_border,
        },
        "aggregate_metrics": {
            "total_test_files_evaluated": len(file_results),
            "labeled_files_count": len(all_aucs),
            "task1_anomaly_detection": {
                "mean_linear_probe_auc_roc": float(np.mean(all_aucs)) if all_aucs else None,
                "std_linear_probe_auc_roc": float(np.std(all_aucs)) if all_aucs else None,
                "mean_linear_probe_average_precision": float(np.mean(all_aps)) if all_aps else None,
                "mean_linear_probe_f1": float(np.mean(all_f1s)) if all_f1s else None,
                "mean_contrast_ratio_cnr": float(np.mean(all_cnrs)) if all_cnrs else None,
            },
            "task2_depth_regression": {
                "mean_depth_r2": float(np.mean(all_r2s)) if all_r2s else None,
                "mean_depth_mae_mm": float(np.mean(all_maes)) if all_maes else None,
            },
            "task3_severity_classification": {
                "mean_severity_macro_f1": float(np.mean(all_sev_f1s)) if all_sev_f1s else None,
            },
            "task4_liftoff_invariance": {
                "mean_linear_cka": liftoff_summary.get("mean_linear_cka"),
                "mean_cosine_similarity": liftoff_summary.get("mean_cosine_similarity"),
            },
            "task5_representation_geometry": {
                "mean_total_3pc_variance": geom_summary.get("mean_total_3pc_variance"),
                "mean_angular_cnr": geom_summary.get("mean_angular_cnr"),
                "mean_angular_auc": geom_summary.get("mean_angular_auc"),
            },
        },
        "per_file_results": file_results,
        "liftoff_invariance": liftoff_summary,
    }

    report_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # evaluation_summary.csv
    csv_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "specimen", "sensor", "waveform", "liftoff",
            "task1_linear_auc", "task1_linear_ap", "task1_linear_f1",
            "task1_mlp_auc", "task1_delta_auc", "task1_cnr",
            "task2_depth_r2", "task2_depth_mae_mm", "task2_depth_rmse_mm",
            "task3_severity_macro_f1", "task3_severity_accuracy",
            "task5_3pc_variance", "task5_angular_cnr", "task5_angular_auc"
        ])
        for r in file_results:
            m = r.get("metrics", {})
            meta = r.get("metadata", {})
            lq = r.get("task5_representation_geometry", {}).get("latent_quality", {})
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
                f"{m.get('depth_r2', 0.0):.4f}" if m.get('depth_r2') is not None else "",
                f"{m.get('depth_mae_mm', 0.0):.4f}" if m.get('depth_mae_mm') is not None else "",
                f"{m.get('depth_rmse_mm', 0.0):.4f}" if m.get('depth_rmse_mm') is not None else "",
                f"{m.get('severity_macro_f1', 0.0):.4f}" if m.get('severity_macro_f1') is not None else "",
                f"{m.get('severity_accuracy', 0.0):.4f}" if m.get('severity_accuracy') is not None else "",
                f"{lq.get('total_3pc_variance', 0.0):.4f}" if lq.get('total_3pc_variance') is not None else "",
                f"{lq.get('angular_cnr', 0.0):.4f}" if lq.get('angular_cnr') is not None else "",
                f"{lq.get('angular_auc', 0.0):.4f}" if lq.get('angular_auc') is not None else "",
            ])

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"Protocol: {protocol_name.upper()} | Holdout Target: {holdout_target}")
    print(f"Evaluated Test Files: {len(file_results)} ({len(all_aucs)} with Ground Truth labels)")
    if all_aucs:
        print(f"Task 1 (Linear Probe Defect Detection): Mean AUC = {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f} | Mean AP = {np.mean(all_aps):.4f}")
        print(f"Task 1 (Defect Contrast Ratio CNR):     Mean CNR = {np.mean(all_cnrs):.2f}")
    if all_r2s:
        print(f"Task 2 (Depth Regression R²):           Mean R²  = {np.mean(all_r2s):.4f} | Mean MAE = {np.mean(all_maes):.4f} mm")
    if all_sev_f1s:
        print(f"Task 3 (Severity Classification):       Mean F1  = {np.mean(all_sev_f1s):.4f}")
    if liftoff_summary.get("mean_linear_cka") is not None:
        print(f"Task 4 (Lift-off Invariance CKA):       Mean CKA = {liftoff_summary['mean_linear_cka']:.4f}")
    if geom_summary.get("mean_total_3pc_variance") is not None:
        print(f"Task 5 (Representation Geometry):       Top-3 PCs Explained Variance = {geom_summary['mean_total_3pc_variance']:.1%}")

    print(f"\nArtifacts organized into 5 modular task folders:")
    print(f"  - 1_Anomaly_Detection:       {os.path.join(args.output_dir, '1_Anomaly_Detection')}")
    print(f"  - 2_Depth_Regression:        {os.path.join(args.output_dir, '2_Depth_Regression')}")
    print(f"  - 3_Severity_Classification: {os.path.join(args.output_dir, '3_Severity_Classification')}")
    print(f"  - 4_Liftoff_Invariance:      {os.path.join(args.output_dir, '4_Liftoff_Invariance')}")
    print(f"  - 5_Representation_Geometry: {os.path.join(args.output_dir, '5_Representation_Geometry')}")
    print(f"  - Summary JSON:              {report_path}")
    print(f"  - Summary CSV:               {csv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
