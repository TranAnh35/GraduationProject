"""
Dataset for 1D Temporal PECT-JEPA (TS-JEPA) (implement.md, Section 4.2).
Loads TDMS waveforms, normalizes per-file, and indexes into a massive 1D waveform collection [N_total, 500].
Supports Memory Mapping (np.memmap) for instant zero-copy loading.
100% self-contained within temporal_1d.
"""

import os
import glob
import hashlib
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import parse_metadata_from_path, read_tdms_1d_waveforms


class PECT1DDataset(Dataset):
    """
    PyTorch Dataset for 1D Transient Waveform JEPA.
    Each sample is a 1D electromagnetic decay signal x in R^500.
    """
    def __init__(
        self,
        file_paths: List[str],
        time_samples: int = 500,
        normalization: str = "min_max",
        raster_correction: bool = True,
        use_memmap: bool = True,
        cache_dir: Optional[str] = ".cache/pect_1d_mmap",
        eps: float = 1e-8
    ):
        super().__init__()
        self.file_paths = file_paths
        self.time_samples = time_samples
        self.normalization = normalization
        self.raster_correction = raster_correction
        self.use_memmap = use_memmap
        self.cache_dir = cache_dir
        self.eps = eps

        if self.use_memmap and self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.metadata_list = [parse_metadata_from_path(fp) for fp in self.file_paths]
        self.sample_index: List[Tuple[int, int]] = []
        self._file_shapes: Dict[int, Tuple[int, int]] = {}  # file_idx -> (n_points, 500)

        # Index points per file and build memory-mapped binary cache
        for f_idx, fp in enumerate(self.file_paths):
            if self.use_memmap and self.cache_dir is not None:
                h_name = hashlib.md5(f"{fp}_1d_{time_samples}_{normalization}".encode('utf-8')).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{h_name}.dat")
                meta_path = os.path.join(self.cache_dir, f"{h_name}.meta")

                if not os.path.exists(cache_path) or not os.path.exists(meta_path):
                    flat_data = read_tdms_1d_waveforms(
                        file_path=fp,
                        target_time_samples=self.time_samples,
                        normalization=self.normalization,
                        raster_correction=self.raster_correction,
                        eps=self.eps
                    )  # [n_points, 500]
                    n_points, T = flat_data.shape

                    with open(meta_path, "w") as f:
                        f.write(f"{n_points},{T}")
                    fp_mmap = np.memmap(cache_path, dtype='float32', mode='w+', shape=(n_points, T))
                    fp_mmap[:] = flat_data[:]
                    fp_mmap.flush()
                    del fp_mmap
                
                with open(meta_path, "r") as f:
                    parts = [int(x) for x in f.read().strip().split(",")]
                    n_points, T = parts[0], parts[1]
                    self._file_shapes[f_idx] = (n_points, T)
            else:
                flat_data = read_tdms_1d_waveforms(
                    file_path=fp,
                    target_time_samples=self.time_samples,
                    normalization=self.normalization,
                    raster_correction=self.raster_correction,
                    eps=self.eps
                )
                n_points = flat_data.shape[0]
                self._file_shapes[f_idx] = (n_points, flat_data.shape[-1])

            # Build global index: (file_idx, point_idx)
            for p_idx in range(n_points):
                self.sample_index.append((f_idx, p_idx))

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        file_idx, point_idx = self.sample_index[idx]
        fp = self.file_paths[file_idx]

        if self.use_memmap and self.cache_dir is not None and file_idx in self._file_shapes:
            h_name = hashlib.md5(f"{fp}_1d_{self.time_samples}_{self.normalization}".encode('utf-8')).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{h_name}.dat")
            shape = self._file_shapes[file_idx]

            mmap_arr = np.memmap(cache_path, dtype='float32', mode='r', shape=shape)
            waveform_np = np.array(mmap_arr[point_idx, :], copy=True)
            waveform = torch.from_numpy(waveform_np).float()
        else:
            flat_data = read_tdms_1d_waveforms(
                file_path=fp,
                target_time_samples=self.time_samples,
                normalization=self.normalization,
                raster_correction=self.raster_correction,
                eps=self.eps
            )
            waveform = torch.from_numpy(flat_data[point_idx]).float()

        meta = dict(self.metadata_list[file_idx])
        meta["point_idx"] = point_idx
        return waveform, meta


def collate_1d_batch(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Collate function for batching 1D waveforms.
    Returns:
        {
            'data': torch.Tensor [B, 500],
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
