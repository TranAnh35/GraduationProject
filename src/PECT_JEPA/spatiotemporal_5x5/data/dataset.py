"""
Dataset and File-Balanced Sampler for 5x5 Spatiotemporal PECT-JEPA.

Produces local 5x5 C-scan grid patches [5, 5, C=256] from TDMS files.
"""

import os
import hashlib
import random
from typing import List, Dict, Optional, Tuple, Any, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from .preprocessing import (
    build_two_channel_input,
    read_tdms_1d_waveforms,
    parse_metadata_from_path,
    linear_time_resample,
    normalize_waveforms_linear,
    apply_lowpass_filter,
)
from .topologies import get_spatial_topology_offsets


class PECT5x5Dataset(Dataset):
    """
    Spatiotemporal 5x5 PECT C-scan Dataset.
    Supports standard dense 5x5 grid or multi-scale Concentric Star topology.
    Output item is always [5, 5, C] tensor (25 points).
    """

    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        arrays: Optional[np.ndarray] = None,  # [N_samples, 5, 5, C] for tests/synth
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        spatial_topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        sX: int = 300,
        sY: int = 300,
        time_samples: int = 500,
        temporal_samples: int = 128,
        in_channels: int = 128,
        resample_mode: str = "linear",
        log_time_samples: int = 128,
        t_start_frac: float = 0.02,
        normalization: str = "global_peak",
        early_window_frac: float = 0.10,
        raster_correction: bool = True,
        crop_border: int = 10,
        apply_lowpass: bool = True,
        lowpass_cutoff: float = 2500.0,
        lowpass_order: int = 4,
        use_memmap: bool = True,
        cache_dir: Optional[str] = ".cache/pect_5x5_mmap",
        eps: float = 1e-8,
        preload_ram: bool = True,
        return_meta: bool = True,
    ):
        super().__init__()
        self.spatial_topology = spatial_topology
        self.star_radii = star_radii
        self.offsets = get_spatial_topology_offsets(
            topology=spatial_topology,
            star_radii=star_radii,
            grid_size=grid_size,
        )  # [25, 2]
        self.offsets_r = self.offsets[:, 0]
        self.offsets_c = self.offsets[:, 1]
        self.max_offset = int(np.max(np.abs(self.offsets)))
        self.pad = max(grid_size // 2, self.max_offset)
        self.return_meta = return_meta
        self.grid_size = grid_size
        self.sX = sX
        self.sY = sY
        self.crop_border = max(0, int(crop_border))
        self.eff_sX = max(1, sX - 2 * self.crop_border)
        self.eff_sY = max(1, sY - 2 * self.crop_border)
        self.time_samples = time_samples
        self.temporal_samples = temporal_samples
        self.resample_mode = resample_mode
        self.log_time_samples = log_time_samples
        self.in_channels = temporal_samples if resample_mode == "linear" else 2 * log_time_samples
        self.t_start_frac = t_start_frac
        self.normalization = normalization
        self.early_window_frac = early_window_frac
        self.raster_correction = raster_correction
        self.apply_lowpass = apply_lowpass
        self.lowpass_cutoff = float(lowpass_cutoff)
        self.lowpass_order = int(lowpass_order)
        self.use_memmap = use_memmap and cache_dir is not None
        self.cache_dir = cache_dir
        self.eps = eps
        self.preload_ram = preload_ram

        self.metadata_list: List[Dict[str, Any]] = []
        self._cache_paths: Dict[int, Tuple[str, tuple]] = {}
        self._file_eff_dims: Dict[int, Tuple[int, int]] = {}
        self.sample_index: List[Tuple[int, int]] = []  # (file_idx, point_idx)
        self.file_point_counts: List[int] = []
        self._data: Optional[np.ndarray] = None
        self._in_memory_files: Dict[int, np.ndarray] = {}

        # Direct in-memory arrays (for tests)
        if arrays is not None:
            self._data = np.asarray(arrays, dtype=np.float32)
            n_samples = self._data.shape[0]
            self.metadata_list = [dict(m) for m in metadata_list] if metadata_list else [{} for _ in range(n_samples)]
            self.file_point_counts = [n_samples]
            for p in range(n_samples):
                self.sample_index.append((0, p))
            self.file_paths = []
            return

        if not file_paths:
            raise ValueError("PECT5x5Dataset requires either `arrays` or `file_paths`.")

        self.file_paths = list(file_paths)
        if self.use_memmap:
            os.makedirs(self.cache_dir, exist_ok=True)

        pbar = tqdm(
            enumerate(self.file_paths),
            total=len(self.file_paths),
            desc="[Caching/Loading PECT 5x5 Files]",
            dynamic_ncols=True,
            leave=False
        )
        for f_idx, fp in pbar:
            self.metadata_list.append(parse_metadata_from_path(fp))
            fname = os.path.basename(fp)
            pbar.set_postfix({"file": fname[:22]})

            if self.use_memmap:
                radii_str = "_".join(str(r) for r in self.star_radii)
                h = hashlib.md5(
                    f"{fp}_topo_{self.spatial_topology}_radii{radii_str}_pad{self.pad}_{self.resample_mode}_{self.in_channels}_{self.normalization}_crop{self.crop_border}_lp{self.apply_lowpass}_{self.lowpass_cutoff}_{self.lowpass_order}".encode("utf-8")
                ).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{h}_padded.dat")
                meta_path = os.path.join(self.cache_dir, f"{h}_padded.meta")

                if os.path.exists(cache_path) and os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        shape = tuple(int(v) for v in f.read().strip().split(","))
                    self._cache_paths[f_idx] = (cache_path, shape)
                    eff_sY = shape[0] - 2 * self.pad
                    eff_sX = shape[1] - 2 * self.pad
                    self._file_eff_dims[f_idx] = (eff_sY, eff_sX)
                    n_points = eff_sY * eff_sX
                else:
                    padded_grid = self._load_and_pad_file(fp)
                    with open(meta_path, "w") as f:
                        f.write(",".join(str(v) for v in padded_grid.shape))
                    mm = np.memmap(cache_path, dtype="float32", mode="w+", shape=padded_grid.shape)
                    mm[:] = padded_grid[:]
                    mm.flush()
                    del mm
                    self._cache_paths[f_idx] = (cache_path, tuple(padded_grid.shape))
                    eff_sY = padded_grid.shape[0] - 2 * self.pad
                    eff_sX = padded_grid.shape[1] - 2 * self.pad
                    self._file_eff_dims[f_idx] = (eff_sY, eff_sX)
                    n_points = eff_sY * eff_sX
            else:
                padded_grid = self._load_and_pad_file(fp)
                self._in_memory_files[f_idx] = padded_grid
                eff_sY = padded_grid.shape[0] - 2 * self.pad
                eff_sX = padded_grid.shape[1] - 2 * self.pad
                self._file_eff_dims[f_idx] = (eff_sY, eff_sX)
                n_points = eff_sY * eff_sX

            self.file_point_counts.append(n_points)
            for p in range(n_points):
                self.sample_index.append((f_idx, p))

        if self.use_memmap and self.preload_ram:
            for f_idx, (c_path, shape) in self._cache_paths.items():
                self._in_memory_files[f_idx] = np.fromfile(c_path, dtype=np.float32).reshape(shape)
            self._cache_paths.clear()

    def _load_and_pad_file(self, fp: str) -> np.ndarray:
        raw = read_tdms_1d_waveforms(
            file_path=fp,
            target_time_samples=self.time_samples,
            normalization="none",
            raster_correction=self.raster_correction,
            eps=self.eps,
        )
        if self.resample_mode == "linear":
            x_resampled = linear_time_resample(raw, n_out=self.temporal_samples)
            if self.apply_lowpass:
                x_resampled = apply_lowpass_filter(
                    x_resampled, cutoff_hz=self.lowpass_cutoff, fs=25600.0, order=self.lowpass_order
                )
            flat_c = normalize_waveforms_linear(x_resampled, normalization=self.normalization, eps=self.eps)
        else:
            two_ch = build_two_channel_input(
                raw,
                log_time_samples=self.log_time_samples,
                t_start_frac=self.t_start_frac,
                normalization=self.normalization,
                early_window_frac=self.early_window_frac,
                eps=self.eps,
            )  # [N, 2, 128]
            N, channels, T_prime = two_ch.shape
            flat_c = two_ch.reshape(N, channels * T_prime)  # [N, 256]

        sX = self.sX
        sY = self.sY if self.sY > 0 else (flat_c.shape[0] // sX if sX > 0 else 300)
        grid = flat_c[:sY * sX].reshape(sY, sX, self.in_channels)  # [sY, sX, in_channels]

        # Crop outer boundary pixels to remove air/edge effect
        if self.crop_border > 0:
            cb = self.crop_border
            grid = grid[cb: sY - cb, cb: sX - cb, :]

        # Edge padding by pad width 2 on rows and columns
        padded = np.pad(grid, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")
        return padded.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        file_idx, point_idx = self.sample_index[idx]

        if self._data is not None:
            patch = self._data[point_idx]
            meta = dict(self.metadata_list[file_idx])
            meta["point_idx"] = point_idx
            return torch.from_numpy(patch).float(), meta

        # Map 1D point_idx back to 2D (row, col) in cropped interior grid
        eff_sX = self._file_eff_dims[file_idx][1] if file_idx in self._file_eff_dims else self.eff_sX
        row = point_idx // eff_sX
        col = point_idx % eff_sX

        if file_idx in self._cache_paths:
            if not hasattr(self, "_mm_handles"):
                self._mm_handles = {}
            if file_idx not in self._mm_handles:
                cache_path, shape = self._cache_paths[file_idx]
                self._mm_handles[file_idx] = np.memmap(cache_path, dtype="float32", mode="r", shape=shape)
            padded = self._mm_handles[file_idx]
        else:
            padded = self._in_memory_files[file_idx]

        # Extract 25 points via vectorized topology offsets around center (row + pad, col + pad)
        center_r = row + self.pad
        center_c = col + self.pad
        sample_rows = center_r + self.offsets_r
        sample_cols = center_c + self.offsets_c
        patch = padded[sample_rows, sample_cols, :].reshape(
            self.grid_size, self.grid_size, self.in_channels
        )

        if not self.return_meta:
            return torch.from_numpy(patch)

        meta = dict(self.metadata_list[file_idx])
        meta["point_idx"] = point_idx
        meta["file_idx"] = file_idx
        meta["row"] = row
        meta["col"] = col
        meta["orig_row"] = row + self.crop_border
        meta["orig_col"] = col + self.crop_border
        return torch.from_numpy(patch), meta


class FileBalancedBatchSampler5x5(Sampler[List[int]]):
    """
    File-balanced batch sampler:
    Each batch samples k_per_file points from many distinct files uniformly.
    """

    def __init__(
        self,
        dataset: PECT5x5Dataset,
        batch_size: int = 256,
        k_per_file: int = 8,
        shuffle: bool = True,
        seed: int = 42,
        steps_per_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.k_per_file = max(1, k_per_file)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.steps_per_epoch = steps_per_epoch

        # Group sample indices by file
        self.file_to_indices: Dict[int, List[int]] = {}
        for global_idx, (f_idx, _) in enumerate(dataset.sample_index):
            if f_idx not in self.file_to_indices:
                self.file_to_indices[f_idx] = []
            self.file_to_indices[f_idx].append(global_idx)

        self.num_files = len(self.file_to_indices)
        self.files_per_batch = max(1, batch_size // self.k_per_file)
        total_samples = len(dataset)
        self.num_batches = steps_per_epoch if steps_per_epoch is not None else max(1, total_samples // batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch * 10007)
        file_pools = {
            f: list(indices) for f, indices in self.file_to_indices.items()
        }
        if self.shuffle:
            for f in file_pools:
                rng.shuffle(file_pools[f])

        file_ptrs = {f: 0 for f in file_pools}
        active_files = list(file_pools.keys())

        for _ in range(self.num_batches):
            batch = []
            if self.shuffle:
                chosen_files = rng.sample(active_files, min(self.files_per_batch, len(active_files)))
            else:
                chosen_files = active_files[:min(self.files_per_batch, len(active_files))]

            for f in chosen_files:
                pool = file_pools[f]
                ptr = file_ptrs[f]
                k = self.k_per_file
                if ptr + k > len(pool):
                    if self.shuffle:
                        rng.shuffle(pool)
                    ptr = 0
                batch.extend(pool[ptr:ptr + k])
                file_ptrs[f] = ptr + k

            if len(batch) < self.batch_size and active_files:
                fill_k = self.batch_size - len(batch)
                extra = rng.choices([idx for f in active_files for idx in file_pools[f][:500]], k=fill_k)
                batch.extend(extra)

            yield batch[:self.batch_size]


def collate_5x5_batch(batch: List[Any]) -> Dict[str, Any]:
    """Collate list of (patch [5, 5, C], meta) or pure patches into a batch dictionary."""
    if isinstance(batch[0], tuple):
        data = torch.stack([item[0] for item in batch], dim=0)
        meta = [item[1] for item in batch]
        return {"data": data, "meta": meta}
    elif isinstance(batch[0], torch.Tensor):
        return {"data": torch.stack(batch, dim=0)}
    else:
        return {"data": torch.from_numpy(np.stack(batch, axis=0)).float()}
