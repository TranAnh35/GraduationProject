"""
Exhaustive Pair-Balanced Cross-Condition Dataset for PECT-JEPA.

Features:
- Exhaustive Spatial Grid Traversal: Visits EVERY physical coordinate (r, c) across ALL directed condition pairs (A -> B != B -> A).
- Zero Arbitrary Dropping: Completely eliminates random subsampling and arbitrary truncation so 100% of scan pixels and defect clusters are learned.
- Deterministic Bijective Mapping: Maps index `idx` uniquely to (pair_idx, coord_idx).
- Shuffled Multi-Condition Learning: When combined with DataLoader(shuffle=True), batches are uniformly mixed across all condition pairs and spatial locations.
"""

import os
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .split import extract_file_metadata
from .topologies import get_spatial_topology_offsets
from .preprocessing import load_canonical_cscan_from_tdms


class PairedPECTDataset(Dataset):
    """
    In-memory Exhaustive Cross-Condition PECT Dataset.
    Loads canonical aligned C-scans into RAM and provides coordinate-aligned patch pairs (x_A, x_B)
    covering the entire finite space of directed condition permutations (A -> B != B -> A) and spatial coordinates.
    """
    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        grid_stride: int = 1,
        samples_per_epoch: Optional[int] = None,
        topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        temporal_samples: int = 128,
        transform_csv: str = "data/ground_truth/raw_tdms_rotate_crop.csv",
        seed: int = 42,
    ):
        super().__init__()
        import glob
        from .split import filter_base_domain_files

        if file_paths is None:
            if data_dir is None:
                data_dir = "data"
            all_files = glob.glob(os.path.join(data_dir, "**", "*.tdms"), recursive=True)
            all_files = [f for f in all_files if not f.endswith("_index.tdms")]
            file_paths = filter_base_domain_files(all_files)

        self.grid_size = grid_size
        self.temporal_samples = temporal_samples
        self.grid_stride = max(1, int(grid_stride))
        self.rng = random.Random(seed)

        self.offsets = get_spatial_topology_offsets(topology, star_radii, grid_size)
        self.max_off = int(np.max(np.abs(self.offsets)))
        self.pad = max(grid_size // 2, self.max_off)

        # Load scans and group by specimen
        self.scans_by_specimen: Dict[str, List[Dict]] = {}
        print(f"\n[Loading & Standardizing {len(file_paths)} Training C-scans into RAM...]")

        for fpath in tqdm(file_paths, desc="Loading canonical C-scans", dynamic_ncols=True):
            meta = extract_file_metadata(fpath)
            spec = meta["specimen"]
            if spec not in self.scans_by_specimen:
                self.scans_by_specimen[spec] = []

            # Load canonical aligned C-scan (already rotated, flipped, and cropped to 270x270x128)
            cscan = load_canonical_cscan_from_tdms(
                file_path=fpath,
                temporal_samples=temporal_samples,
                apply_transforms=True,
            )
            sY, sX, C = cscan.shape

            padded = np.pad(cscan, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")
            ref_patch = cscan[10:15, 10:15, :].copy()

            self.scans_by_specimen[spec].append({
                "file_path": fpath,
                "meta": meta,
                "sY": sY,
                "sX": sX,
                "C": C,
                "padded": padded,
                "ref_patch": ref_patch,
            })

        # Construct ALL directed pairs (A, B) with A != B on each specimen
        self.directed_pairs: List[Tuple[Dict, Dict]] = []
        for spec, scan_list in self.scans_by_specimen.items():
            n = len(scan_list)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.directed_pairs.append((scan_list[i], scan_list[j]))

        num_pairs = len(self.directed_pairs)
        print(f"Loaded {len(self.scans_by_specimen)} specimens with total {num_pairs} directed condition pairs (A -> B != B -> A).")
        for spec in self.scans_by_specimen:
            n_scans = len(self.scans_by_specimen[spec])
            print(f"  Specimen '{spec}': {n_scans} condition scans -> {n_scans * (n_scans - 1)} directed pairs.")

        # Determine spatial coordinate grid (e.g. 270x270 with stride)
        # Find minimum common shape
        min_sY = min(min(s["sY"] for s in scans) for scans in self.scans_by_specimen.values())
        min_sX = min(min(s["sX"] for s in scans) for scans in self.scans_by_specimen.values())

        # Build coordinate list
        coord_list = []
        for r in range(0, min_sY, self.grid_stride):
            for c in range(0, min_sX, self.grid_stride):
                coord_list.append((r, c))

        self.coords = np.array(coord_list, dtype=np.int32)
        self.coords_per_pair = len(self.coords)

        # Total exhaustive universe of samples = num_pairs * coords_per_pair
        self.total_exhaustive_samples = num_pairs * self.coords_per_pair

        if samples_per_epoch is None:
            self.samples_per_epoch = self.total_exhaustive_samples
            print(f"Dataset configured: 100% EXHAUSTIVE COVERAGE -> {self.samples_per_epoch:,} patch pairs per epoch ({self.coords_per_pair:,} pixels x {num_pairs} condition pairs).")
        else:
            self.samples_per_epoch = int(samples_per_epoch)
            print(f"Dataset configured: Capped to {self.samples_per_epoch:,} samples/epoch (Exhaustive total available: {self.total_exhaustive_samples:,}).")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        # Bijective mapping: map idx to (pair_idx, coord_idx)
        # If capped or expanded beyond total_exhaustive_samples, modulo ensures cycling without out-of-bounds
        eff_idx = idx % self.total_exhaustive_samples
        pair_idx = eff_idx // self.coords_per_pair
        coord_idx = eff_idx % self.coords_per_pair

        scan_a, scan_b = self.directed_pairs[pair_idx]
        r, c = self.coords[coord_idx]

        # Extract star topology patches at identical coordinate (r, c)
        r_pad = r + self.pad
        c_pad = c + self.pad

        sample_r = r_pad + self.offsets[:, 0]
        sample_c = c_pad + self.offsets[:, 1]

        patch_a = scan_a["padded"][sample_r, sample_c, :].reshape(self.grid_size, self.grid_size, -1)
        patch_b = scan_b["padded"][sample_r, sample_c, :].reshape(self.grid_size, self.grid_size, -1)

        return {
            "x_a": torch.from_numpy(patch_a).float(),
            "x_b": torch.from_numpy(patch_b).float(),
            "ref_b": torch.from_numpy(scan_b["ref_patch"]).float(),
            "coord": torch.tensor([r, c], dtype=torch.long),
        }
