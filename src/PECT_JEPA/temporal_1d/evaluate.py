"""
Downstream Evaluation Script for 1D Temporal PECT-JEPA (TS-JEPA).
Evaluates representations on 1D waveforms with rich tqdm progress tracking:
1. Cross-Sensor Generalization
2. Cross-Waveform Generalization
3. Cross-Lift-off Generalization
100% self-contained within temporal_1d.
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.PECT_JEPA.temporal_1d.models.jepa_1d import PECT_JEPA_1D
from src.PECT_JEPA.temporal_1d.configs.config import get_default_config_1d, Temporal1DConfig
from src.PECT_JEPA.temporal_1d.data.dataset import PECT1DDataset, collate_1d_batch
from src.PECT_JEPA.temporal_1d.data.preprocessing import find_all_tdms_files
from src.PECT_JEPA.temporal_1d.data.split import split_cross_sensor, split_cross_waveform, split_cross_liftoff
from src.PECT_JEPA.temporal_1d.evaluation.anomaly_1d import WaveformAnomalyDetector
from src.PECT_JEPA.temporal_1d.evaluation.visualization_1d import plot_waveform_anomaly_map_1d


def run_split_evaluation(
    model: PECT_JEPA_1D,
    train_files: List[str],
    test_files: List[str],
    task_name: str,
    target_domain: str,
    device: str = "cuda",
    batch_size: int = 512,
    save_plots: bool = True,
    use_memmap: bool = True
) -> Dict[str, Any]:
    """Generic evaluation runner for a single train/test split."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    ev_pad_to = (
        model.config.raw_padded_length
        if getattr(model.config, "tokenizer_mode", "resampled") == "raw"
        else None
    )

    train_ds = PECT1DDataset(
        file_paths=train_files, time_samples=model.config.time_samples,
        log_time_samples=model.config.log_time_samples,
        normalization=model.config.normalization,
        use_memmap=use_memmap, cache_dir=model.config.cache_dir,
        pad_to=ev_pad_to, pad_mode=model.config.pad_mode
    )
    test_ds = PECT1DDataset(
        file_paths=test_files, time_samples=model.config.time_samples,
        log_time_samples=model.config.log_time_samples,
        normalization=model.config.normalization,
        use_memmap=use_memmap, cache_dir=model.config.cache_dir,
        pad_to=ev_pad_to, pad_mode=model.config.pad_mode
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        return {"error": f"Empty dataset for {target_domain}"}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_1d_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_1d_batch, num_workers=0)

    # 1. Extract and fit clustering prototype bank on train features
    train_features = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f"  [Train Feature Bank: {target_domain[:18]}]", dynamic_ncols=True, leave=False):
            x = batch["data"].to(dev)
            feats = model.extract_features(x, pool=True)
            train_features.append(feats.cpu())

    if len(train_features) == 0:
        return {"error": "Failed to extract train features"}

    train_tensor = torch.cat(train_features, dim=0)
    detector = WaveformAnomalyDetector(method="clustering", n_clusters=2)
    detector.fit(train_tensor)

    # 2. Score test domain waveforms
    test_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  [Scoring Target Domain: {target_domain[:18]}]", dynamic_ncols=True, leave=False):
            x = batch["data"].to(dev)
            feats = model.extract_features(x, pool=True)
            scores = detector.score(feats)
            test_scores.append(scores)

    all_scores = np.concatenate(test_scores)

    if save_plots and len(all_scores) >= 90000:
        os.makedirs(f"results/plots/1d/{task_name}", exist_ok=True)
        first_scan_scores = all_scores[:90000]
        plot_path = f"results/plots/1d/{task_name}/{target_domain}_scan_1.png"
        try:
            plot_waveform_anomaly_map_1d(
                anomaly_scores=first_scan_scores,
                grid_shape=(300, 300),
                save_path=plot_path,
                title=f"1D JEPA ({task_name}: {target_domain}) - 300x300 Anomaly Heatmap"
            )
        except Exception:
            pass

    return {
        "task": task_name,
        "target": target_domain,
        "num_train": len(train_ds),
        "num_test": len(test_ds),
        "mean_anomaly_score": float(np.mean(all_scores)),
        "std_anomaly_score": float(np.std(all_scores)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 1D Temporal PECT-JEPA representations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to 1D model checkpoint .pt")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    parser.add_argument("--task", type=str, default="all", choices=["all", "sensor", "waveform", "liftoff"], help="Evaluation task")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--save_plots", type=lambda v: v.lower() == "true", default=True, help="Save C-Scan heatmaps")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading 1D checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config_dict = checkpoint.get("config", {})
    if config_dict:
        config = Temporal1DConfig.from_dict(config_dict)
    else:
        config = get_default_config_1d()

    model = PECT_JEPA_1D(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_files = find_all_tdms_files(args.data_dir)
    print(f"Found {len(all_files)} total TDMS files in {args.data_dir}")
    if len(all_files) == 0:
        print("ERROR: No TDMS files found.")
        return

    # Index files once to parse metadata
    sample_dataset = PECT1DDataset(
        file_paths=all_files,
        time_samples=model.config.time_samples,
        use_memmap=True,
        cache_dir=model.config.cache_dir,
    )
    meta_list = sample_dataset.metadata_list

    # ---------------- 1. Cross-Sensor Evaluation ----------------
    if args.task in ["all", "sensor"]:
        print("\n" + "=" * 65)
        print("RUNNING 1D CROSS-SENSOR GENERALIZATION EVALUATION")
        print("=" * 65)
        sensors = sorted(list({m["sensor"] for m in meta_list if m["sensor"] != "Unknown"}))
        for target in tqdm(sensors, desc="Cross-Sensor Benchmarks", dynamic_ncols=True):
            train_f, test_f = split_cross_sensor(meta_list, test_sensor=target)
            res = run_split_evaluation(model, train_f, test_f, "cross_sensor", target,
                                       device=args.device, batch_size=args.batch_size, save_plots=args.save_plots)
            print(f"  [Sensor: {target:<22}] Score: {res.get('mean_anomaly_score', 'N/A'):.5f} | Train: {res.get('num_train', 0)} | Test: {res.get('num_test', 0)}")

    # ---------------- 2. Cross-Waveform Evaluation ----------------
    if args.task in ["all", "waveform"]:
        print("\n" + "=" * 65)
        print("RUNNING 1D CROSS-WAVEFORM GENERALIZATION EVALUATION")
        print("=" * 65)
        waveforms = sorted(list({m["waveform"] for m in meta_list if m["waveform"] != "Unknown"}))
        for target in tqdm(waveforms, desc="Cross-Waveform Benchmarks", dynamic_ncols=True):
            train_f, test_f = split_cross_waveform(meta_list, test_waveform=target)
            res = run_split_evaluation(model, train_f, test_f, "cross_waveform", target,
                                       device=args.device, batch_size=args.batch_size, save_plots=args.save_plots)
            print(f"  [Waveform: {target:<20}] Score: {res.get('mean_anomaly_score', 'N/A'):.5f} | Train: {res.get('num_train', 0)} | Test: {res.get('num_test', 0)}")

    # ---------------- 3. Cross-Lift-off Evaluation ----------------
    if args.task in ["all", "liftoff"]:
        print("\n" + "=" * 65)
        print("RUNNING 1D CROSS-LIFT-OFF GENERALIZATION EVALUATION")
        print("=" * 65)
        liftoffs = sorted(list({m["liftoff"] for m in meta_list if m["liftoff"] != "Unknown"}))
        for target in tqdm(liftoffs, desc="Cross-Liftoff Benchmarks", dynamic_ncols=True):
            train_f, test_f = split_cross_liftoff(meta_list, test_liftoff=target)
            res = run_split_evaluation(model, train_f, test_f, "cross_liftoff", target,
                                       device=args.device, batch_size=args.batch_size, save_plots=args.save_plots)
            print(f"  [Lift-off: {target:<20}] Score: {res.get('mean_anomaly_score', 'N/A'):.5f} | Train: {res.get('num_train', 0)} | Test: {res.get('num_test', 0)}")

    print("\n--- 1D Downstream Evaluation Finished ---")


if __name__ == "__main__":
    main()
