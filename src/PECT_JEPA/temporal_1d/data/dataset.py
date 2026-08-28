"""
Dataset for 1D Temporal PECT-JEPA (Stage A).

Produces the two-channel log-time model input [N, 2, T'=128] from either
in-memory raw arrays (tests / synthetic) or TDMS files (memmap-cached).
Includes the Stage A5 file-balanced batch sampler.
"""

import os
import hashlib
import random
from typing import List, Dict, Optional, Tuple, Any, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocessing import (
    build_two_channel_input,
    read_tdms_1d_waveforms,
    parse_metadata_from_path,
)


class PECT1DDataset(Dataset):
    """
    Two-channel PECT waveform dataset [N, 2, T'].

    Sources, in priority order:
      1. `arrays`:  in-memory raw waveforms [N, T_raw] (+ `metadata_list`) —
         used for tests / synthetic experiments, no TDMS dependency.
      2. `file_paths`: TDMS files read through `read_tdms_1d_waveforms`
         (metadata parsed from paths), optionally memory-mapped cached.
    """

    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        arrays: Optional[np.ndarray] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        time_samples: int = 500,
        log_time_samples: int = 128,
        t_start_frac: float = 0.02,
        normalization: str = "peak_early",
        early_window_frac: float = 0.10,
        raster_correction: bool = True,
        use_memmap: bool = True,
        cache_dir: Optional[str] = ".cache/pect_1d_mmap",
        eps: float = 1e-8,
        pad_to: Optional[int] = None,
        pad_mode: str = "edge"
    ):
        super().__init__()
        self.time_samples = time_samples
        self.log_time_samples = log_time_samples
        self.t_start_frac = t_start_frac
        self.normalization = normalization
        self.early_window_frac = early_window_frac
        self.raster_correction = raster_correction
        self.use_memmap = use_memmap and cache_dir is not None
        self.cache_dir = cache_dir
        self.eps = eps
        self.pad_to = pad_to          # None -> log-time resample; int -> B0 raw pad
        self.pad_mode = pad_mode

        self.metadata_list: List[Dict[str, Any]] = []
        self._cache_paths: Dict[int, Tuple[str, tuple]] = {}
        self.sample_index: List[Tuple[int, int]] = []  # (file_idx, point_idx)
        self.file_point_counts: List[int] = []
        self._data: Optional[np.ndarray] = None
        self._in_memory_files: Dict[int, np.ndarray] = {}

        if arrays is not None:
            data = build_two_channel_input(
                arrays, log_time_samples, t_start_frac,
                normalization, early_window_frac, eps=eps,
                pad_to=self.pad_to, pad_mode=self.pad_mode
            )
            self._data = data  # [N, 2, T']
            n_points = data.shape[0]
            if metadata_list is not None:
                self.metadata_list = [dict(m) for m in metadata_list]
            else:
                self.metadata_list = [{} for _ in range(n_points)]
            self.file_point_counts = [n_points]
            for p in range(n_points):
                self.sample_index.append((0, p))
            self.file_paths = []
            return

        if not file_paths:
            raise ValueError("PECT1DDataset requires either `arrays` or `file_paths`.")

        self.file_paths = list(file_paths)
        if self.use_memmap:
            os.makedirs(self.cache_dir, exist_ok=True)

        pbar = tqdm(
            enumerate(self.file_paths),
            total=len(self.file_paths),
            desc="[Caching/Loading PECT 1D Files]",
            dynamic_ncols=True,
            leave=False
        )
        for f_idx, fp in pbar:
            self.metadata_list.append(parse_metadata_from_path(fp))
            fname = os.path.basename(fp)
            pbar.set_postfix({"file": fname[:22]})

            if self.use_memmap:
                h = hashlib.md5(
                    f"{fp}_2ch_{log_time_samples}_{normalization}_{t_start_frac}"
                    f"_pad{self.pad_to}_{self.pad_mode}".encode("utf-8")
                ).hexdigest()
                cache_path = os.path.join(self.cache_dir, f"{h}.dat")
                meta_path = os.path.join(self.cache_dir, f"{h}.meta")

                if os.path.exists(cache_path) and os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        shape = tuple(int(v) for v in f.read().strip().split(","))
                    self._cache_paths[f_idx] = (cache_path, shape)
                    n_points = shape[0]
                else:
                    arr = self._load_file(fp)  # [n_points, 2, T']
                    n_points = arr.shape[0]
                    with open(meta_path, "w") as f:
                        f.write(",".join(str(v) for v in arr.shape))
                    mm = np.memmap(cache_path, dtype="float32", mode="w+", shape=arr.shape)
                    mm[:] = arr[:]
                    mm.flush()
                    del mm
                    self._cache_paths[f_idx] = (cache_path, tuple(arr.shape))
            else:
                arr = self._load_file(fp)
                n_points = arr.shape[0]
                self._in_memory_files[f_idx] = arr

            self.file_point_counts.append(n_points)
            for p in range(n_points):
                self.sample_index.append((f_idx, p))

    def _load_file(self, fp: str) -> np.ndarray:
        raw = read_tdms_1d_waveforms(
            file_path=fp,
            target_time_samples=self.time_samples,
            normalization="none",   # Stage A normalization applied in build_two_channel_input
            raster_correction=self.raster_correction,
            eps=self.eps,
        )
        return build_two_channel_input(
            raw,
            log_time_samples=self.log_time_samples,
            t_start_frac=self.t_start_frac,
            normalization=self.normalization,
            early_window_frac=self.early_window_frac,
            eps=self.eps,
            pad_to=self.pad_to,
            pad_mode=self.pad_mode,
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        file_idx, point_idx = self.sample_index[idx]

        if file_idx in self._cache_paths:
            cache_path, shape = self._cache_paths[file_idx]
            mm = np.memmap(cache_path, dtype="float32", mode="r", shape=shape)
            item = np.array(mm[point_idx], copy=True)
        elif self._data is not None:
            item = self._data[point_idx]
        else:
            item = self._in_memory_files[file_idx][point_idx]

        meta = dict(self.metadata_list[file_idx])
        meta["point_idx"] = point_idx
        meta["file_idx"] = file_idx
        return torch.from_numpy(np.ascontiguousarray(item)).float(), meta


class FileBalancedBatchSampler:
    """
    Stage A5: file-balanced batch sampling.

    Each batch draws points from many distinct files (uniform over files)
    instead of uniformly over the ~90k near-duplicate points of whichever
    file comes first. Prevents the in-batch file shortcut and exposes the
    model to many acquisition conditions per optimization step.
    """

    def __init__(
        self,
        file_point_counts: List[int],
        batch_size: int = 256,
        k_per_file: int = 8,
        seed: int = 42,
        shuffle: bool = True
    ):
        if batch_size < k_per_file:
            raise ValueError("batch_size must be >= k_per_file")
        self.file_point_counts = list(file_point_counts)
        self.batch_size = batch_size
        self.k_per_file = k_per_file
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._offsets = np.cumsum([0] + self.file_point_counts[:-1]).tolist()
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        total = sum(self.file_point_counts)
        return max(1, total // self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self._rng.randint(0, 2 ** 31 - 1) + self._epoch)
        n_batches = len(self)
        n_files = len(self.file_point_counts)
        for _ in range(n_batches):
            batch: List[int] = []
            n_draw = max(1, self.batch_size // self.k_per_file)
            file_ids = list(range(n_files))
            if self.shuffle:
                rng.shuffle(file_ids)
            chosen = file_ids[:min(n_draw, n_files)]
            per_file = self.batch_size // len(chosen)
            rem = self.batch_size - per_file * len(chosen)
            for j, f in enumerate(chosen):
                k = per_file + (1 if j < rem else 0)
                n_pts = self.file_point_counts[f]
                k = min(k, n_pts)
                pts = rng.sample(range(n_pts), k) if n_pts > k else list(range(n_pts))
                batch.extend(self._offsets[f] + p for p in pts)
            # top-up if some chosen files had fewer points than requested
            in_batch = set(batch)
            attempts = 0
            while len(batch) < self.batch_size and attempts < 5000:
                f = rng.randrange(n_files)
                p = rng.randrange(self.file_point_counts[f])
                idx = self._offsets[f] + p
                if idx not in in_batch:
                    in_batch.add(idx)
                    batch.append(idx)
                attempts += 1
            while len(batch) < self.batch_size:
                f = rng.randrange(n_files)
                p = rng.randrange(self.file_point_counts[f])
                batch.append(self._offsets[f] + p)
            yield batch


def collate_1d_batch(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Collate two-channel items into {data: [B, 2, T'], metadata: [...]}.
    """
    tensors = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    return {"data": torch.stack(tensors, dim=0), "metadata": metas}
