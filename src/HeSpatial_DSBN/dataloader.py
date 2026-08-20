"""
Astrous / Star Spatial Context Dataloader for HeSpatial-DSBN.
Supports multi-sensor (Hall, Coil, Diffensor) and multi-material (Aluminum, Steel) datasets.
Standalone module - no external dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from nptdms import TdmsFile

from . import config
from .processing import norm_zscore

Scale = Tuple[int, int]  # (radius, dilation)


# =============================================================================
# TDMS & RASTER LOADING
# =============================================================================

def load_tdms_data(file_path: str) -> Tuple[np.ndarray, int, int, int]:
    """
    Load TDMS file -> Returns (data_3d, sX, sY, samples).
    data_3d shape: (N, T, 1)
    """
    tdms_file = TdmsFile.read(file_path)
    group1 = tdms_file["Freq_Sampling_SizeX_SizeY"]
    info = group1.channels()[0]
    f = info[:][0]
    sampling = info[:][1]
    sX = int(info[:][2])
    sY = int(info[:][3])
    samples = int(sampling / f)

    group2 = tdms_file["Waveform"]
    data = group2.channels()[0][:]
    num_pts = sX * sY
    X_scan = np.reshape(data[:num_pts * samples], (num_pts, samples)).astype(np.float32)

    # Downsample if samples > config.TIME_SAMPLES
    if samples > config.TIME_SAMPLES:
        step = max(1, samples // config.TIME_SAMPLES)
        idx = np.arange(0, samples, step)[:config.TIME_SAMPLES]
        X_scan = X_scan[:, idx]
        samples = X_scan.shape[1]

    return X_scan[:, :, None], sX, sY, samples


def reshape_raster(data_2d: np.ndarray, sX: int) -> np.ndarray:
    """
    Reshape 2D scan sequence (N, T) -> (sY, sX, T) handling serpentine raster scanning.
    """
    sY = data_2d.shape[0] // sX
    T = data_2d.shape[1]
    r = data_2d[: sY * sX].reshape(sY, sX, T).copy()
    r[1::2] = r[1::2, ::-1]  # Flip odd rows for serpentine scan reversal
    return r


def load_tdms_raster(file_path: str, target_t: int = config.TIME_SAMPLES) -> Tuple[np.ndarray, int, int, int]:
    """
    Load TDMS file directly into a 3D raster array of shape (sY, sX, T) with baseline cleaning.
    """
    tdms_file = TdmsFile.read(file_path)
    group1 = tdms_file["Freq_Sampling_SizeX_SizeY"]
    info = group1.channels()[0]
    f = info[:][0]
    sampling = info[:][1]
    sX = int(info[:][2])
    sY = int(info[:][3])
    samples = int(sampling / f)

    group2 = tdms_file["Waveform"]
    data = group2.channels()[0][:]
    num_pts = sX * sY
    X_scan = np.reshape(data[:num_pts * samples], (num_pts, samples)).astype(np.float32)

    # Downsample to target_t if necessary
    if samples > target_t:
        step = max(1, samples // target_t)
        idx = np.arange(0, samples, step)[:target_t]
        X_scan = X_scan[:, idx]
        samples = X_scan.shape[1]

    raster = reshape_raster(X_scan, sX)
    # Baseline DC offset subtraction
    raster = raster - np.mean(raster, axis=-1, keepdims=True)
    return raster, sY, sX, samples


def get_tdms_file_path(sensor: str, material: str, crack_size: str = "1mm", data_dir: str = config.DATA_DIR) -> Optional[str]:
    """Find path to TDMS file given sensor, material and crack size."""
    sensor_dir = "diff" if sensor.lower().startswith("diff") else sensor.lower()
    mat_dir = "al" if material.lower().startswith("al") else "steel"
    mat_label = "AL" if mat_dir == "al" else "steel"
    
    filename = f"{sensor_dir}_crack_{mat_label}_{crack_size}.tdms"
    path = os.path.join(data_dir, sensor_dir, mat_dir, filename)
    if os.path.exists(path):
        return path
    
    # Fallback search
    pattern = os.path.join(data_dir, sensor_dir, mat_dir, f"*{crack_size}*.tdms")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def get_default_normal_ranges(material_name: str) -> List[Tuple[int, int]]:
    """Get defect-free normal scanning ranges for normal background sampling."""
    mat = material_name.lower()
    if mat in ["al", "aluminum"]:
        return [(0, 15), (65, 140), (185, 255), (300, 320)]
    elif mat in ["steel", "carbon_steel"]:
        return [(10, 45), (70, 90), (120, 140), (175, 206)]
    else:
        return [(0, 20)]


def get_normal_centers(raster: np.ndarray, material_name: str = "al", mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract centers of normal (defect-free) regions for training normal reconstruction.
    """
    sY, sX = raster.shape[0], raster.shape[1]
    if mask is not None:
        normal_mask = ~mask.astype(bool)
        centers = np.argwhere(normal_mask).astype(np.int32)
        if len(centers) > 0:
            return centers

    # Fallback to default normal range bands
    ranges = get_default_normal_ranges(material_name)
    normal_mask = np.zeros((sY, sX), dtype=bool)
    for y_start, y_end in ranges:
        ys = min(y_start, sY)
        ye = min(y_end, sY)
        if ye > ys:
            normal_mask[ys:ye, :] = True

    if not np.any(normal_mask):
        # If ranges fall outside sY, take first 25% and last 25% rows
        q = max(1, sY // 4)
        normal_mask[:q, :] = True
        normal_mask[-q:, :] = True

    return np.argwhere(normal_mask).astype(np.int32)


def generate_synthetic_pect_raster(
    sY: int = 40, sX: int = 40, T: int = config.TIME_SAMPLES, num_defects: int = 4, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic PECT raster scan (sY, sX, T) and Ground-Truth defect mask for testing.
    """
    if seed is not None:
        np.random.seed(seed)

    raster = np.zeros((sY, sX, T), dtype=np.float32)
    mask = np.zeros((sY, sX), dtype=bool)

    t = np.linspace(0, 1, T, dtype=np.float32)
    base_wave = np.sin(2 * np.pi * 5 * t) * np.exp(-3 * t)

    for y in range(sY):
        for x in range(sX):
            noise = np.random.normal(0, 0.05, size=(T,)).astype(np.float32)
            raster[y, x, :] = base_wave + noise

    # Add artificial defects
    for d in range(num_defects):
        cy = np.random.randint(5, max(6, sY - 5))
        cx = np.random.randint(5, max(6, sX - 5))
        rad = np.random.randint(2, 4)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dy**2 + dx**2 <= rad**2:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < sY and 0 <= nx < sX:
                        mask[ny, nx] = True
                        defect_signal = 0.5 * np.exp(-5 * t) * np.cos(2 * np.pi * 12 * t)
                        raster[ny, nx, :] += defect_signal

    return raster, mask


def load_dataset_raster(
    sensor: str,
    material: str,
    crack_size: str = "1mm",
    use_synthetic: bool = False,
    sY: int = 40,
    sX: int = 40,
    T: int = config.TIME_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real TDMS raster or generate synthetic raster if real data is unavailable/requested.
    Returns: (raster (sY, sX, T), mask (sY, sX))
    """
    if not use_synthetic:
        path = get_tdms_file_path(sensor, material, crack_size)
        if path and os.path.exists(path):
            try:
                raster, r_sY, r_sX, r_T = load_tdms_raster(path, target_t=T)
                # Create defect mask by finding high-energy/prominent anomaly regions
                diff_feat = np.max(np.abs(np.diff(raster, axis=-1)), axis=-1)
                thresh = np.percentile(diff_feat, 95)
                mask = diff_feat > thresh
                return raster, mask
            except Exception as e:
                print(f"  ⚠️ Failed to load TDMS file {path}: {e}. Falling back to synthetic raster.")
    
    return generate_synthetic_pect_raster(sY, sX, T)


# =============================================================================
# STAR NEIGHBORHOOD SAMPLING
# =============================================================================

def _normalize_scales(scales: Optional[Sequence[Scale]], radius: Optional[int], dilation: Optional[int]) -> List[Scale]:
    if scales is not None and len(scales) > 0:
        return [(int(r), int(d)) for r, d in scales]
    r = int(radius if radius is not None else config.RADIUS)
    d = int(dilation if dilation is not None else config.DILATION)
    return [(r, d)]


def _offsets(radius: int, dilation: int) -> np.ndarray:
    offs = np.arange(-radius, radius + 1, dtype=np.int32) * int(dilation)
    return offs[offs != 0]  # Exclude center offset 0


def num_star_channels(scales: Sequence[Scale]) -> int:
    """Total context channels extracted (8 directions x radius)."""
    return sum(8 * int(r) for r, _ in scales)


@dataclass
class StarContextSampler:
    """
    Extracts 4 lines (8 principal directions: V, H, Main Diag, Anti Diag) 
    around center point (y, x), excluding the center point itself.
    """
    raster: np.ndarray  # (sY, sX, T)
    scales: List[Scale]
    pad_mode: str = "reflect"

    def __post_init__(self):
        self.scales = _normalize_scales(self.scales, None, None)
        self.pad = max(r * d for r, d in self.scales) if self.scales else 0
        if self.pad > 0:
            self.raster_pad = np.pad(self.raster, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=self.pad_mode)
        else:
            self.raster_pad = self.raster

    def batch(self, centers_yx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        centers_yx: (B, 2) centers in original raster coordinates.
        Returns:
            X: (B, T, 8*r) Context tensor (excluding center)
            y: (B, T, 1) Target center waveform
        """
        centers_yx = np.asarray(centers_yx, dtype=np.int32)
        B = centers_yx.shape[0]
        T = self.raster.shape[2]
        
        if B == 0:
            C = num_star_channels(self.scales)
            return np.zeros((0, T, C), dtype=np.float32), np.zeros((0, T, 1), dtype=np.float32)

        y0 = centers_yx[:, 0]
        x0 = centers_yx[:, 1]

        y_target = self.raster[y0, x0, :][:, :, None]  # Target center trace (B, T, 1)

        yp = y0 + self.pad
        xp = x0 + self.pad

        ctx_list = []
        for r, d in self.scales:
            offs = _offsets(r, d)
            # Line 1: Vertical (Up & Down)
            for off in offs:
                ctx_list.append(self.raster_pad[yp + off, xp, :])
            # Line 2: Horizontal (Left & Right)
            for off in offs:
                ctx_list.append(self.raster_pad[yp, xp + off, :])
            # Line 3: Main Diagonal (Top-Left to Bottom-Right)
            for off in offs:
                ctx_list.append(self.raster_pad[yp + off, xp + off, :])
            # Line 4: Anti Diagonal (Top-Right to Bottom-Left)
            for off in offs:
                ctx_list.append(self.raster_pad[yp + off, xp - off, :])

        X_ctx = np.stack(ctx_list, axis=-1).astype(np.float32)  # (B, T, 8r)
        return X_ctx, y_target.astype(np.float32)


# =============================================================================
# MULTI-SENSOR MULTI-DOMAIN BATCH GENERATOR
# =============================================================================

class MultiDomainPECTDataset:
    """
    Manages multi-domain PECT datasets across 3 Sensors (Hall, Coil, Diffensor)
    and 2 Materials (Aluminum, Steel) for Cross-Sensor Adaptation.
    """
    def __init__(
        self,
        rasters_dict: Dict[Tuple[str, str], np.ndarray],
        batch_size: int = config.BATCH_SIZE,
        scales: Optional[List[Scale]] = None,
    ):
        """
        rasters_dict: Dict mapping (sensor_name, material_name) -> raster array (sY, sX, T)
        """
        self.rasters_dict = rasters_dict
        self.batch_size = batch_size
        self.scales = _normalize_scales(scales, None, None)
        self.samplers: Dict[Tuple[str, str], StarContextSampler] = {}

        for key, raster in rasters_dict.items():
            self.samplers[key] = StarContextSampler(raster=raster, scales=self.scales)

    def sample_domain_batch(
        self,
        sensor_name: str,
        material_name: str,
        centers: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch for a specific domain key.
        Returns:
            X: (B, T, 8r) Context tensor
            y: (B, T, 1) Target center waveform
            domain_ids: (B,) Integer Domain ID (0 to 5)
        """
        key = (sensor_name.lower(), material_name.lower())
        if key not in self.samplers:
            raise KeyError(f"Domain key {key} not found in dataset!")

        bs = batch_size or self.batch_size
        idx = np.random.choice(len(centers), size=min(bs, len(centers)), replace=False)
        batch_centers = centers[idx]

        X_b, y_b = self.samplers[key].batch(batch_centers)
        d_id = config.DOMAIN_MAP.get(key, 0)
        domain_ids = np.full((len(X_b),), d_id, dtype=np.int32)

        return X_b, y_b, domain_ids
