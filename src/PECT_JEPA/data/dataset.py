"""
PECT Dataset and TDMS File Loading.
Reads TDMS scan files following reference dataloader implementation and wraps into PyTorch Dataset.
"""

import os
import glob
import re
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nptdms import TdmsFile

from .preprocessing import reshape_raster, normalize_per_file


def parse_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Parse experiment metadata (sensor, waveform, defect type, liftoff) from file path and filename.
    Metadata is used for grouping, split, and downstream evaluation, NOT forward pass input (Section 3).
    """
    norm_path = os.path.normpath(file_path).replace("\\", "/")
    parts = norm_path.split("/")
    filename = os.path.basename(norm_path).lower()

    sensor = "unknown"
    specimen = "unknown"
    waveform = "unknown"
    liftoff = "unknown"

    # Search in directory path components
    for p in parts:
        p_lower = p.lower()
        if "hall_air" in p_lower or "air_core" in p_lower:
            sensor = "Hall_Air_Core"
        elif "hall_pot" in p_lower:
            sensor = "Hall_Pot_Core"
        elif "diferential" in p_lower or "differential" in p_lower:
            sensor = "Differential_Pot_Core"
        elif "tmr" in p_lower:
            sensor = "TMR"

        if "corrosion" in p_lower or "corosion" in p_lower:
            specimen = "Corrosion"
        elif "rivet_v1" in p_lower:
            specimen = "Rivet_v1"
        elif "rivet_v2" in p_lower or "mixed" in p_lower:
            specimen = "Rivet_v2"

        if "chirp" in p_lower:
            waveform = "Chirp"
        elif "gauss" in p_lower:
            waveform = "Gaussian"
        elif "square" in p_lower:
            waveform = "Square"

    # Search liftoff in filename (e.g. z1, z2, z3, lf_1mm, etc.)
    if "z1" in filename or "1mm" in filename:
        liftoff = "1mm"
    elif "z2" in filename or "2mm" in filename:
        liftoff = "2mm"
    elif "z3" in filename or "3mm" in filename:
        liftoff = "3mm"

    return {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "sensor": sensor,
        "specimen": specimen,
        "waveform": waveform,
        "liftoff": liftoff,
    }


def read_tdms_scan(
    file_path: str,
    target_time_samples: int = 500,
    raster: bool = True,
    target_spatial_size: Optional[Tuple[int, int]] = (300, 300),
    normalization: str = "min_max",
    eps: float = 1e-8
) -> np.ndarray:
    """
    Read a PECT TDMS scan file, perform raster mapping, downsample time axis to target_time_samples,
    and apply per-file normalization.

    Args:
        file_path: Path to .tdms file
        target_time_samples: Number of temporal points per response (default: 500)
        raster: Whether to reconstruct 2D spatial grid from raster scan
        target_spatial_size: Target (H, W) spatial dimensions to slice/pad
        normalization: 'min_max', 'standard', or 'none'
        eps: Small constant for numerical stability

    Returns:
        3D numpy array of shape [H, W, target_time_samples] (float32)
    """
    with TdmsFile.read(file_path) as tdms_file:
        group1 = tdms_file['Freq_Sampling_SizeX_SizeY']
        infor = group1.channels()[0]
        f = float(infor[:][0])
        sampling = float(infor[:][1])
        sX = int(infor[:][2])
        sY = int(infor[:][3])
        samples = int(sampling / f)

        group2 = tdms_file['Waveform']
        raw_data = group2.channels()[0][:]

    # Reshape continuous points to (-1, samples)
    data = np.reshape(raw_data, (-1, samples))

    # Downsample or resample temporal dimension to target_time_samples (default 500)
    if samples > target_time_samples:
        step = samples // target_time_samples
        idx = np.arange(0, step * target_time_samples, step)
        data = data[:, idx]
    elif samples < target_time_samples:
        # Interpolate if fewer samples
        orig_t = np.linspace(0, 1, samples)
        new_t = np.linspace(0, 1, target_time_samples)
        data = np.apply_along_axis(lambda row: np.interp(new_t, orig_t, row), axis=1, arr=data)

    if raster:
        data_2d = reshape_raster(data, sX)  # shape: [sY, sX, target_time_samples]
    else:
        num_rows = data.shape[0] // sX
        data_2d = data.reshape(num_rows, sX, target_time_samples)

    # Adjust spatial dimensions if target_spatial_size is specified
    if target_spatial_size is not None:
        target_h, target_w = target_spatial_size
        curr_h, curr_w = data_2d.shape[0], data_2d.shape[1]

        # Crop if larger
        crop_h = min(curr_h, target_h)
        crop_w = min(curr_w, target_w)
        cropped = data_2d[:crop_h, :crop_w, :]

        # Pad if smaller
        if crop_h < target_h or crop_w < target_w:
            pad_h = target_h - crop_h
            pad_w = target_w - crop_w
            cropped = np.pad(
                cropped,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='edge'
            )
        data_2d = cropped

    data_2d = data_2d.astype(np.float32)

    # Apply per-file normalization
    if normalization != "none":
        data_2d = normalize_per_file(data_2d, method=normalization, eps=eps)

    return data_2d


class PECTDataset(Dataset):
    """
    PyTorch Dataset for PECT acquisition files.
    Each sample is a full acquisition [H, W, 500].
    """

    def __init__(
        self,
        file_paths: List[str],
        time_samples: int = 500,
        raster_correction: bool = True,
        spatial_size: Optional[Tuple[int, int]] = (300, 300),
        normalization: str = "min_max",
        cache_in_memory: bool = False,
        eps: float = 1e-8
    ):
        super().__init__()
        self.file_paths = file_paths
        self.time_samples = time_samples
        self.raster_correction = raster_correction
        self.spatial_size = spatial_size
        self.normalization = normalization
        self.cache_in_memory = cache_in_memory
        self.eps = eps

        self.metadata_list = [parse_metadata_from_path(fp) for fp in self.file_paths]
        self._cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx], self.metadata_list[idx]

        file_path = self.file_paths[idx]
        data_np = read_tdms_scan(
            file_path=file_path,
            target_time_samples=self.time_samples,
            raster=self.raster_correction,
            target_spatial_size=self.spatial_size,
            normalization=self.normalization,
            eps=self.eps
        )
        tensor_data = torch.from_numpy(data_np).float()

        if self.cache_in_memory:
            self._cache[idx] = tensor_data

        return tensor_data, self.metadata_list[idx]


def collate_pect_batch(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Custom collate function for batching PECT acquisitions.
    Returns:
        {
            'data': torch.Tensor [B, H, W, 500],
            'metadata': List[Dict[str, Any]]
        }
    """
    tensors = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    stacked = torch.stack(tensors, dim=0)
    return {
        "data": stacked,
        "metadata": metas
    }


def find_all_tdms_files(data_dir: str) -> List[str]:
    """Find all .tdms files recursively in data_dir."""
    pattern = os.path.join(data_dir, "**", "*.tdms")
    return sorted(glob.glob(pattern, recursive=True))
