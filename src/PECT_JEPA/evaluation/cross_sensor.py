"""
Cross-Sensor Generalization Evaluation for PECT-JEPA (Section 18.2).
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

from ..models.jepa import PECT_JEPA
from ..data.dataset import PECTDataset, collate_pect_batch
from ..data.split import split_cross_sensor
from .anomaly import RepresentationAnomalyDetector


def evaluate_cross_sensor(
    model: PECT_JEPA,
    all_file_paths: List[str],
    test_sensor: str = "TMR",
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run Cross-Sensor evaluation: fit normal representation on source sensors,
    and evaluate anomaly/representation stability on target sensor.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    # Split dataset
    sample_dataset = PECTDataset(file_paths=all_file_paths, time_samples=model.config.temporal_encoder.raw_samples)
    train_files, test_files = split_cross_sensor(sample_dataset.metadata_list, test_sensor=test_sensor)

    print(f"Cross-Sensor Eval [{test_sensor}]: {len(train_files)} train files, {len(test_files)} test files")

    train_ds = PECTDataset(file_paths=train_files, time_samples=model.config.temporal_encoder.raw_samples)
    test_ds = PECTDataset(file_paths=test_files, time_samples=model.config.temporal_encoder.raw_samples)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_pect_batch)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_pect_batch)

    # Extract representations from train files
    train_features = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch["data"].to(dev)
            feats = model.extract_features(x, pool_temporal=True)  # [1, H_t, W_t, D]
            train_features.append(feats.squeeze(0).cpu())

    if len(train_features) > 0:
        train_tensor = torch.cat([f.reshape(-1, f.shape[-1]) for f in train_features], dim=0)
        detector = RepresentationAnomalyDetector(method="prototype")
        detector.fit(train_tensor)

        test_maps = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["data"].to(dev)
                feats = model.extract_features(x, pool_temporal=True)
                amap = detector.score(feats.squeeze(0))
                test_maps.append(amap)

        return {
            "test_sensor": test_sensor,
            "num_train_files": len(train_files),
            "num_test_files": len(test_files),
            "test_anomaly_maps": test_maps
        }

    return {"error": "No train features extracted"}
