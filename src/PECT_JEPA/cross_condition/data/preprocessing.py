"""
Data Preprocessing and Canonical Geometric Transformation for Cross-Condition PECT.
Applies authoritative rotation, flipping, and border cropping from raw_tdms_rotate_crop.csv
so that all C-scans [270, 270, C] strictly align with CAD ground-truth masks.
"""

import os
import csv
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np

from src.PECT_JEPA.temporal_1d.data.preprocessing import (
    read_tdms_1d_waveforms,
    apply_lowpass_filter,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import (
    linear_time_resample,
    normalize_waveforms_linear,
)


class CanonicalCoordinateTransformer:
    """
    Manages authoritative geometric transformations (rotation_deg, flip_lr, flip_ud, crop)
    from raw_tdms_rotate_crop.csv.
    """
    def __init__(self, csv_path: str = "data/ground_truth/raw_tdms_rotate_crop.csv"):
        self.transforms: Dict[str, Dict[str, int]] = {}
        self.default_crop = {"crop_top": 15, "crop_bottom": 15, "crop_left": 15, "crop_right": 15}

        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = row.get("source", "").strip()
                    if not src:
                        continue
                    if src == "__global__":
                        self.default_crop["crop_top"] = int(row["crop_top"]) if row.get("crop_top") else 15
                        self.default_crop["crop_bottom"] = int(row["crop_bottom"]) if row.get("crop_bottom") else 15
                        self.default_crop["crop_left"] = int(row["crop_left"]) if row.get("crop_left") else 15
                        self.default_crop["crop_right"] = int(row["crop_right"]) if row.get("crop_right") else 15
                        continue

                    rot = int(row["rotation_deg"]) if row.get("rotation_deg") else 0
                    flr = int(row["flip_lr"]) if row.get("flip_lr") else 0
                    fud = int(row["flip_ud"]) if row.get("flip_ud") else 0
                    ct = int(row["crop_top"]) if row.get("crop_top") else self.default_crop["crop_top"]
                    cb = int(row["crop_bottom"]) if row.get("crop_bottom") else self.default_crop["crop_bottom"]
                    cl = int(row["crop_left"]) if row.get("crop_left") else self.default_crop["crop_left"]
                    cr = int(row["crop_right"]) if row.get("crop_right") else self.default_crop["crop_right"]

                    self.transforms[src] = {
                        "rotation_deg": rot,
                        "flip_lr": flr,
                        "flip_ud": fud,
                        "crop_top": ct,
                        "crop_bottom": cb,
                        "crop_left": cl,
                        "crop_right": cr,
                    }

    def get_transform(self, file_path: str) -> Dict[str, int]:
        norm_path = file_path.replace("\\", "/")
        base_name = os.path.basename(file_path)

        # 1. Substring match
        for src_key, t in self.transforms.items():
            if src_key.replace("\\", "/") in norm_path:
                return t

        # 2. Basename match
        for src_key, t in self.transforms.items():
            if os.path.basename(src_key) == base_name:
                return t

        # 3. Fallback
        return {
            "rotation_deg": 0,
            "flip_lr": 0,
            "flip_ud": 0,
            **self.default_crop,
        }

    def transform_cscan(self, cscan: np.ndarray, file_path: str) -> np.ndarray:
        """
        Transforms raw [300, 300, C] C-scan into canonical [270, 270, C] coordinate space.
        """
        t = self.get_transform(file_path)
        img = cscan.copy()

        # Rotate (k * 90 deg counter-clockwise)
        rot_deg = t["rotation_deg"]
        k = (rot_deg // 90) % 4
        if k > 0:
            img = np.rot90(img, k=k, axes=(0, 1))

        # Horizontal flip
        if t["flip_lr"]:
            img = np.flip(img, axis=1)

        # Vertical flip
        if t["flip_ud"]:
            img = np.flip(img, axis=0)

        # Crop to canonical 270x270
        ct, cb = t["crop_top"], t["crop_bottom"]
        cl, cr = t["crop_left"], t["crop_right"]
        sY, sX = img.shape[:2]

        cropped = img[ct : sY - cb, cl : sX - cr]
        return np.ascontiguousarray(cropped, dtype=np.float32)


# Global Transformer instance
_GLOBAL_TRANSFORMER: Optional[CanonicalCoordinateTransformer] = None


def get_canonical_transformer() -> CanonicalCoordinateTransformer:
    global _GLOBAL_TRANSFORMER
    if _GLOBAL_TRANSFORMER is None:
        _GLOBAL_TRANSFORMER = CanonicalCoordinateTransformer()
    return _GLOBAL_TRANSFORMER


def load_canonical_cscan_from_tdms(
    file_path: str,
    time_samples: int = 500,
    temporal_samples: int = 128,
    lowpass_cutoff: float = 2500.0,
    normalization: str = "global_peak",
    raster_correction: bool = True,
    apply_transforms: bool = True,
) -> np.ndarray:
    """
    Loads TDMS, resamples to 128 continuous temporal samples, applies lowpass,
    normalizes, and applies authoritative geometric rotation/flip/crop from raw_tdms_rotate_crop.csv.
    Returns:
        canonical_cscan: [270, 270, 128] float32 array aligned with CAD ground truth.
    """
    raw = read_tdms_1d_waveforms(
        file_path=file_path,
        target_time_samples=time_samples,
        normalization="none",
        raster_correction=raster_correction,
    )
    resampled = linear_time_resample(raw, n_out=temporal_samples)
    lowpassed = apply_lowpass_filter(resampled, cutoff_hz=lowpass_cutoff, fs=25600.0, order=4)
    normed = normalize_waveforms_linear(lowpassed, normalization=normalization)

    # Reshape to nominal [300, 300, C]
    grid_300 = normed[:300 * 300].reshape(300, 300, temporal_samples)

    if apply_transforms:
        transformer = get_canonical_transformer()
        canonical_grid = transformer.transform_cscan(grid_300, file_path)
        return canonical_grid
    return grid_300.astype(np.float32)
