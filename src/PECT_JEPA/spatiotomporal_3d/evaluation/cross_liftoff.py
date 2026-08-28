"""
Cross-Lift-off Generalization Evaluation for PECT-JEPA (Section 18.4) with tqdm progress tracking.
Assesses representation robustness across lift-off heights:
- 1 mm (z1)
- 2 mm (z2)
- 3 mm (z3)
"""

import torch
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.jepa import PECT_JEPA
from ..data.dataset import PECTDataset, collate_pect_batch
from ..data.split import split_cross_liftoff
from .anomaly import RepresentationAnomalyDetector


def evaluate_cross_liftoff(
    model: PECT_JEPA,
    all_file_paths: List[str],
    test_liftoff: str = "z3",
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run Cross-Lift-off evaluation across lift-off distances.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    sample_dataset = PECTDataset(file_paths=all_file_paths, time_samples=model.config.temporal_encoder.raw_samples)
    train_files, test_files = split_cross_liftoff(sample_dataset.metadata_list, test_liftoff=test_liftoff)

    print(f"\n[Cross-Liftoff Evaluation: Target={test_liftoff}] {len(train_files)} train files, {len(test_files)} test files")

    train_ds = PECTDataset(file_paths=train_files, time_samples=model.config.temporal_encoder.raw_samples)
    test_ds = PECTDataset(file_paths=test_files, time_samples=model.config.temporal_encoder.raw_samples)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_pect_batch)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_pect_batch)

    train_features = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=f"Fitting Normal Bank ({test_liftoff})", dynamic_ncols=True, leave=False):
            x = batch["data"].to(dev)
            feats = model.extract_features(x, pool_temporal=True)
            train_features.append(feats.squeeze(0).cpu())

    if len(train_features) > 0:
        train_tensor = torch.cat([f.reshape(-1, f.shape[-1]) for f in train_features], dim=0)
        detector = RepresentationAnomalyDetector(method="prototype")
        detector.fit(train_tensor)

        test_maps = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Scoring Test Lift-off ({test_liftoff})", dynamic_ncols=True, leave=False):
                x = batch["data"].to(dev)
                feats = model.extract_features(x, pool_temporal=True)
                amap = detector.score(feats.squeeze(0))
                test_maps.append(amap)

        return {
            "test_liftoff": test_liftoff,
            "num_train_files": len(train_files),
            "num_test_files": len(test_files),
            "test_anomaly_maps": test_maps
        }

    return {"error": "No train features extracted"}
