"""
Pair-Balanced Cross-Condition Dataset for PECT-JEPA.
Samples spatial patches (x_A, x_B) at IDENTICAL physical coordinates (r, c)
from two distinct measurement conditions on the SAME specimen, covering all 112 directed permutations (A -> B and B -> A).
All C-scans are pre-rotated and cropped using raw_tdms_rotate_crop.csv into the canonical coordinate frame.
"""

import os
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .split import extract_file_metadata
from .topologies import get_spatial_topology_offsets
from .preprocessing import load_canonical_cscan_from_tdms


class PairedPECTDataset(Dataset):
    """
    In-memory Pair-Balanced PECT Dataset.
    Loads canonical aligned C-scans into RAM and provides coordinate-aligned patch pairs (x_A, x_B)
    sampled across all 112 directed condition permutations (A -> B != B -> A).
    """
    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        samples_per_epoch: Optional[int] = None,
        points_per_pair: int = 4000,
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
                apply_transforms=True
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

        # If samples_per_epoch is not explicitly specified, calculate realistically based on pairs
        if samples_per_epoch is None:
            self.samples_per_epoch = num_pairs * points_per_pair  # e.g. 112 * 4,000 = 448,000 samples/epoch
        else:
            self.samples_per_epoch = samples_per_epoch

        print(f"Dataset configured: {self.samples_per_epoch:,} patch pairs per epoch (~{self.samples_per_epoch // num_pairs:,} samples/pair).")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        # 1. Deterministically / uniformly cycle through or pick a directed pair (A, B)
        pair_idx = idx % len(self.directed_pairs)
        scan_a, scan_b = self.directed_pairs[pair_idx]

        sY = min(scan_a["sY"], scan_b["sY"])
        sX = min(scan_a["sX"], scan_b["sX"])

        # 2. Pick identical physical coordinates (r, c)
        r = self.rng.randint(0, sY - 1)
        c = self.rng.randint(0, sX - 1)

        # 3. Extract star patches
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
