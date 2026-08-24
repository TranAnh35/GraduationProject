"""
PECT Dataset and TDMS File Loading with Memory Mapping & Resolution-Agnostic Support (Section 2, implement.md).
Reads TDMS scan files, applies raster scan mapping and Min-Max per-file normalization.
Supports Clip-as-Sample extraction with np.memmap / Lazy Loading for <500MB RAM consumption.
"""

import os
import glob
import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nptdms import TdmsFile

from .preprocessing import reshape_raster, normalize_per_file


def parse_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Parse experiment metadata (sensor, waveform, defect type, liftoff) from file path and filename.
    Metadata is used for grouping, split, and downstream evaluation, NOT forward pass input (Section 1.2).
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
    target_spatial_size: Optional[Tuple[int, int]] = None,
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
        target_spatial_size: Optional target (H, W) spatial dimensions to slice/pad (None keeps native size)
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

    # Apply per-file normalization (Section 2.1)
    if normalization != "none":
        data_2d = normalize_per_file(data_2d, method=normalization, eps=eps)

    return data_2d


class PECTDataset(Dataset):
    """
    PyTorch Dataset for full PECT acquisition files.
    Each sample is a full acquisition [H, W, 500].
    """

    def __init__(
        self,
        file_paths: List[str],
        time_samples: int = 500,
        raster_correction: bool = True,
        spatial_size: Optional[Tuple[int, int]] = None,
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


class PECTClipDataset(Dataset):
    """
    PyTorch Dataset for Clip-as-Sample training (Section 2.1 & 2.2, implement.md).
    Slices each acquisition into temporal clips of shape [H, W, T_c] (T_c = 16).
    Supports On-the-fly Random Spatial Cropping [H_crop, W_crop, T_c] (e.g. [64, 64, 16])
    and Memory Mapping (np.memmap) to maintain RAM usage < 500MB (Section 2.3).
    """
    def __init__(
        self,
        file_paths: List[str],
        clip_length: int = 16,
        clip_stride: int = 8,
        time_samples: int = 500,
        raster_correction: bool = True,
        spatial_size: Optional[Tuple[int, int]] = None,
        spatial_crop_size: Optional[Tuple[int, int]] = (64, 64),
        train: bool = True,
        normalization: str = "min_max",
        use_memmap: bool = True,
        cache_dir: Optional[str] = ".cache/pect_mmap",
        eps: float = 1e-8
    ):
        super().__init__()
        self.file_paths = file_paths
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.time_samples = time_samples
        self.raster_correction = raster_correction
        self.spatial_size = spatial_size
        self.spatial_crop_size = spatial_crop_size
        self.train = train
        self.normalization = normalization
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir
        self.eps = eps

        if self.use_memmap and self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.metadata_list = [parse_metadata_from_path(fp) for fp in self.file_paths]
        self.clip_index: List[Tuple[int, int]] = []
        self._mmap_files: Dict[int, Any] = {}
        self._file_shapes: Dict[int, Tuple[int, int, int]] = {}

        # Build index and optional memory-mapped binary cache
        for f_idx, fp in enumerate(self.file_paths):
            num_clips = max(1, (self.time_samples - self.clip_length) // self.clip_stride + 1)
            for c_idx in range(num_clips):
                start_t = c_idx * self.clip_stride
                self.clip_index.append((f_idx, start_t))

            if self.use_memmap and self.cache_dir is not None:
                h_name = hashlib.md5(fp.encode('utf-8')).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{h_name}.dat")
                meta_path = os.path.join(self.cache_dir, f"{h_name}.meta")

                if not os.path.exists(cache_path) or not os.path.exists(meta_path):
                    data_np = read_tdms_scan(
                        file_path=fp,
                        target_time_samples=self.time_samples,
                        raster=self.raster_correction,
                        target_spatial_size=self.spatial_size,
                        normalization=self.normalization,
                        eps=self.eps
                    )
                    shape = data_np.shape
                    with open(meta_path, "w") as f:
                        f.write(f"{shape[0]},{shape[1]},{shape[2]}")
                    fp_mmap = np.memmap(cache_path, dtype='float32', mode='w+', shape=shape)
                    fp_mmap[:] = data_np[:]
                    fp_mmap.flush()
                    del fp_mmap
                
                with open(meta_path, "r") as f:
                    shape_parts = [int(x) for x in f.read().strip().split(",")]
                    self._file_shapes[f_idx] = (shape_parts[0], shape_parts[1], shape_parts[2])

    def __len__(self) -> int:
        return len(self.clip_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        file_idx, start_t = self.clip_index[idx]
        fp = self.file_paths[file_idx]

        if self.use_memmap and self.cache_dir is not None and file_idx in self._file_shapes:
            h_name = hashlib.md5(fp.encode('utf-8')).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{h_name}.dat")
            shape = self._file_shapes[file_idx]

            # Read slice directly via memory mapping without loading full tensor to RAM
            mmap_arr = np.memmap(cache_path, dtype='float32', mode='r', shape=shape)
            end_t = min(start_t + self.clip_length, shape[-1])
            clip_np = np.array(mmap_arr[:, :, start_t:end_t], copy=True)
            clip_data = torch.from_numpy(clip_np).float()
        else:
            data_np = read_tdms_scan(
                file_path=fp,
                target_time_samples=self.time_samples,
                raster=self.raster_correction,
                target_spatial_size=self.spatial_size,
                normalization=self.normalization,
                eps=self.eps
            )
            end_t = min(start_t + self.clip_length, data_np.shape[-1])
            clip_data = torch.from_numpy(data_np[:, :, start_t:end_t]).float()

        # Replicate pad if temporal points are fewer than clip_length
        if clip_data.shape[-1] < self.clip_length:
            pad_len = self.clip_length - clip_data.shape[-1]
            clip_data = torch.nn.functional.pad(clip_data, (0, pad_len), mode='replicate')

        meta = dict(self.metadata_list[file_idx])
        meta["clip_start_t"] = start_t

        # On-the-fly Random Spatial Cropping (Section 2.2)
        if self.spatial_crop_size is not None:
            crop_h, crop_w = self.spatial_crop_size
            curr_h, curr_w = clip_data.shape[0], clip_data.shape[1]

            if curr_h >= crop_h and curr_w >= crop_w:
                if self.train:
                    y0 = int(np.random.randint(0, curr_h - crop_h + 1))
                    x0 = int(np.random.randint(0, curr_w - crop_w + 1))
                else:
                    y0 = (curr_h - crop_h) // 2
                    x0 = (curr_w - crop_w) // 2
                clip_data = clip_data[y0 : y0 + crop_h, x0 : x0 + crop_w, :]
                meta["crop_y0"] = y0
                meta["crop_x0"] = x0
            else:
                # Pad if dimensions smaller than crop size
                pad_h = max(0, crop_h - curr_h)
                pad_w = max(0, crop_w - curr_w)
                clip_data = torch.nn.functional.pad(clip_data, (0, 0, 0, pad_w, 0, pad_h), mode='replicate')
                meta["crop_y0"] = 0
                meta["crop_x0"] = 0

        return clip_data, meta


def collate_pect_batch(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Custom collate function for batching PECT clips or acquisitions.
    Returns:
        {
            'data': torch.Tensor [B, H, W, T_c],
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
