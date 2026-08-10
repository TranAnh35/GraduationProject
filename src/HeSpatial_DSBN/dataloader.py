"""
Astrous / Star Spatial Context Dataloader for HeSpatial-DSBN.
Supports multi-sensor (Hall, Coil, Diffensor) and multi-material (Aluminum, Steel) datasets.
Standalone module - no external dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass
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
    X_scan = np.reshape(data, (-1, samples)).astype(np.float32)

    # Downsample if samples > 500
    if samples > 500:
        idx = np.arange(0, X_scan.shape[1], 4)[:config.TIME_SAMPLES]
        X_scan = X_scan[:, idx]
        samples = X_scan.shape[1]

    return X_scan[:-1, :, None], sX, sY, samples


def reshape_raster(data_2d: np.ndarray, sX: int) -> np.ndarray:
    """
    Reshape 2D scan sequence (N, T) -> (sY, sX, T) handling serpentine raster scanning.
    """
    sY = data_2d.shape[0] // sX
    T = data_2d.shape[1]
    r = data_2d[: sY * sX].reshape(sY, sX, T)
    r[1::2] = r[1::2, ::-1]  # Flip odd rows for serpentine scan reversal
    return r


def get_default_normal_ranges(material_name: str) -> List[Tuple[int, int]]:
    """Get defect-free normal scanning ranges for pseudo-normal initialization."""
    mat = material_name.lower()
    if mat == "al" or mat == "aluminum":
        return [(0, 15), (65, 140), (185, 255), (300, 320)]
    elif mat == "steel" or mat == "carbon_steel":
        return [(10, 45), (70, 90), (120, 140), (175, 206)]
    else:
        return [(0, 20)]


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
