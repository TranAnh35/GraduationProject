"""
Cross-Sensor Generalization Evaluation for PECT-JEPA (Section 18.2) with tqdm progress tracking.
Assesses whether learned representations transfer across different sensor types:
- Differential_Pot_Core
- Hall_Air_Core
- Hall_Pot_Core
- TMR
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.jepa import PECT_JEPA
from ..data.dataset import PECTDataset, collate_pect_batch
from ..data.split import split_cross_sensor
from .anomaly import RepresentationAnomalyDetector
from .visualization import plot_anomaly_heatmap_3d, plot_latent_tsne


def evaluate_cross_sensor(
    model: PECT_JEPA,
    all_file_paths: List[str],
    test_sensor: str = "TMR",
    device: str = "cuda",
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Run Cross-Sensor evaluation with Unsupervised Latent Clustering
    and Dense [300x300] Anomaly Map Generation (stride=1).
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    # Split dataset
    sample_dataset = PECTDataset(file_paths=all_file_paths, time_samples=model.config.temporal_encoder.raw_samples)
    train_files, test_files = split_cross_sensor(sample_dataset.metadata_list, test_sensor=test_sensor)

    print(f"\n[Cross-Sensor Evaluation: Target={test_sensor}] {len(train_files)} train files, {len(test_files)} test files")

    train_ds = PECTDataset(file_paths=train_files, time_samples=model.config.temporal_encoder.raw_samples)
    test_ds = PECTDataset(file_paths=test_files, time_samples=model.config.temporal_encoder.raw_samples)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_pect_batch)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_pect_batch)

    # 1. Extract representations from source sensors
    train_features = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f"Fitting Clustering Normal Bank ({test_sensor})", dynamic_ncols=True, leave=False):
            x = batch["data"].to(dev)
            feats = model.extract_features(x, pool_temporal=True)  # [1, H_t, W_t, D]
            train_features.append(feats.squeeze(0).cpu())

    if len(train_features) > 0:
        train_tensor = torch.cat([f.reshape(-1, f.shape[-1]) for f in train_features], dim=0)
        # Unsupervised Latent Clustering to find dominant healthy cluster
        detector = RepresentationAnomalyDetector(method="clustering", n_clusters=2)
        detector.fit(train_tensor)

        test_maps = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc=f"Scoring Test Sensor ({test_sensor})", dynamic_ncols=True, leave=False)):
                x = batch["data"].to(dev)
                # Dense stride=1 feature extraction for true [300, 300] resolution
                feats_dense = model.extract_dense_features(x, pool_temporal=True)  # [H, W, D] (300, 300, 128)
                amap_dense = detector.score(feats_dense)  # [300, 300]
                test_maps.append(amap_dense)

                if save_plots and idx < 3:
                    raw_np = x.squeeze(0).cpu().numpy()
                    plot_path = f"results/plots/3d/cross_sensor/{test_sensor}_sample_{idx+1}.png"
                    plot_anomaly_heatmap_3d(
                        raw_scan=raw_np,
                        anomaly_map=amap_dense,
                        save_path=plot_path,
                        title=f"Cross-Sensor Transfer ({test_sensor}) - Dense 300x300 Heatmap"
                    )

        return {
            "test_sensor": test_sensor,
            "num_train_files": len(train_files),
            "num_test_files": len(test_files),
            "test_anomaly_maps": test_maps
        }

    return {"error": "No train features extracted"}
