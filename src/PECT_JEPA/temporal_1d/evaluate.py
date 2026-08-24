"""
Downstream Evaluation Script for 1D Temporal PECT-JEPA (TS-JEPA).
Evaluates representations on 1D waveforms with tqdm progress tracking:
1. Anomaly Detection / Feature space separation
2. Cross-Sensor evaluation
3. Cross-Wave evaluation
4. Cross-Lift-off evaluation
100% self-contained within temporal_1d.
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any

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
from src.PECT_JEPA.temporal_1d.evaluation.visualization_1d import plot_waveform_anomaly_map_1d, plot_latent_tsne_1d


def evaluate_cross_sensor_1d(
    model: PECT_JEPA_1D,
    all_file_paths: List[str],
    test_sensor: str = "TMR",
    device: str = "cuda",
    batch_size: int = 512,
    save_plots: bool = True
) -> Dict[str, Any]:
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    sample_dataset = PECT1DDataset(file_paths=all_file_paths, time_samples=model.config.time_samples, use_memmap=False)
    train_files, test_files = split_cross_sensor(sample_dataset.metadata_list, test_sensor=test_sensor)

    print(f"\n[1D Cross-Sensor Evaluation: Target={test_sensor}] {len(train_files)} train files, {len(test_files)} test files")

    train_ds = PECT1DDataset(file_paths=train_files, time_samples=model.config.time_samples, use_memmap=False)
    test_ds = PECT1DDataset(file_paths=test_files, time_samples=model.config.time_samples, use_memmap=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_1d_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_1d_batch)

    # 1. Fit clustering normal bank on source sensors
    train_features = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f"Fitting Clustering Normal Bank ({test_sensor})", dynamic_ncols=True, leave=False):
            x = batch["data"].to(dev)
            feats = model.extract_features(x, pool=True)  # [B, D]
            train_features.append(feats.cpu())

    if len(train_features) > 0:
        train_tensor = torch.cat(train_features, dim=0)
        # Unsupervised Latent Clustering
        detector = WaveformAnomalyDetector(method="clustering", n_clusters=2)
        detector.fit(train_tensor)

        test_scores = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Scoring Test Sensor ({test_sensor})", dynamic_ncols=True, leave=False):
                x = batch["data"].to(dev)
                feats = model.extract_features(x, pool=True)
                scores = detector.score(feats)
                test_scores.append(scores)

        all_scores = np.concatenate(test_scores)

        if save_plots and len(all_scores) >= 90000:
            # First 90,000 points represent the first 300x300 scan
            first_scan_scores = all_scores[:90000]
            plot_path = f"results/plots/1d/cross_sensor/{test_sensor}_scan_1.png"
            plot_waveform_anomaly_map_1d(
                anomaly_scores=first_scan_scores,
                grid_shape=(300, 300),
                save_path=plot_path,
                title=f"1D Temporal JEPA ({test_sensor}) - Native Point-Wise 300x300 Heatmap"
            )

        return {
            "test_sensor": test_sensor,
            "num_train_waveforms": len(train_ds),
            "num_test_waveforms": len(test_ds),
            "mean_anomaly_score": float(np.mean(all_scores))
        }

    return {"error": "No train features extracted"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 1D Temporal PECT-JEPA representations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to 1D model checkpoint .pt")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing TDMS files")
    parser.add_argument("--task", type=str, default="all", choices=["all", "sensor"], help="Evaluation task")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
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

    if args.task in ["all", "sensor"]:
        print("\n" + "=" * 55)
        print("RUNNING 1D CROSS-SENSOR EVALUATION")
        print("=" * 55)
        sensors = ["TMR", "Hall_Pot_Core", "Differential_Pot_Core", "Hall_Air_Core"]
        for target_sensor in tqdm(sensors, desc="1D Cross-Sensor Tasks", dynamic_ncols=True):
            res = evaluate_cross_sensor_1d(model, all_files, test_sensor=target_sensor, device=args.device, batch_size=args.batch_size)
            print(f"Target Sensor: {target_sensor} | Mean Anomaly Score: {res.get('mean_anomaly_score', 'N/A')}")

    print("\n--- 1D Evaluation Finished ---")


if __name__ == "__main__":
    main()
