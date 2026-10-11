"""
Paired Cross-Condition Dataset for PECT-JEPA.
Samples spatial patches (x_A, x_B) at IDENTICAL physical coordinates (r, c)
from two distinct measurement conditions (Sensor, Waveform, Liftoff) on the SAME physical specimen.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .split import extract_file_metadata
from .topologies import get_spatial_topology_offsets
from ..evaluation.cscan_extractor import load_cscan_from_tdms


SENSOR_MAP = {"hall_air_core": 0, "hall_pot_core": 1, "tmr": 2}
WAVEFORM_MAP = {"square": 0, "gaussian": 1, "chirp": 2}
LIFTOFF_MAP = {"z1": 0, "z2": 1, "z3": 2}


class PairedPECTDataset(Dataset):
    """
    In-memory Paired PECT Dataset.
    Loads base training C-scans into RAM and provides infinite or epoch-based
    cross-condition coordinate-aligned patch pairs.
    """
    def __init__(
        self,
        file_paths: List[str],
        samples_per_epoch: int = 20480,
        topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        temporal_samples: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.grid_size = grid_size
        self.temporal_samples = temporal_samples
        self.rng = random.Random(seed)

        # Offsets
        self.offsets = get_spatial_topology_offsets(topology, star_radii, grid_size) # [25, 2]
        self.max_off = int(np.max(np.abs(self.offsets)))
        self.pad = max(grid_size // 2, self.max_off)

        # Load and group scans by specimen
        self.scans_by_specimen: Dict[str, List[Dict]] = {}
        print(f"Loading {len(file_paths)} paired training C-scans into RAM...")

        for fpath in file_paths:
            meta = extract_file_metadata(fpath)
            spec = meta["specimen"]
            if spec not in self.scans_by_specimen:
                self.scans_by_specimen[spec] = []

            cscan = load_cscan_from_tdms(fpath) # [sY, sX, C]
            sY, sX, C = cscan.shape

            # Pre-pad C-scan for ultra-fast slicing
            padded = np.pad(cscan, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")

            # Extract a 5x5 reference patch from outer sound metal margin (e.g., [10:15, 10:15, :])
            # which is guaranteed sound metal (no defect present in margins)
            ref_patch = cscan[10:15, 10:15, :].copy()

            s_id = SENSOR_MAP.get(meta["sensor"].lower(), 0)
            w_id = WAVEFORM_MAP.get(meta["waveform"].lower(), 0)
            l_id = LIFTOFF_MAP.get(meta["liftoff"].lower(), 0)

            self.scans_by_specimen[spec].append({
                "file_path": fpath,
                "meta": meta,
                "sY": sY,
                "sX": sX,
                "C": C,
                "padded": padded,
                "ref_patch": ref_patch,
                "cond": np.array([s_id, w_id, l_id], dtype=np.int64),
            })

        self.specimen_keys = list(self.scans_by_specimen.keys())
        print(f"Loaded specimens: {self.specimen_keys}")
        for k in self.specimen_keys:
            print(f"  Specimen '{k}': {len(self.scans_by_specimen[k])} condition scans.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        # 1. Pick a random specimen
        spec = self.rng.choice(self.specimen_keys)
        scan_list = self.scans_by_specimen[spec]

        # 2. Pick two distinct condition scans A and B
        if len(scan_list) >= 2:
            idx_a, idx_b = self.rng.sample(range(len(scan_list)), 2)
        else:
            idx_a, idx_b = 0, 0

        scan_a = scan_list[idx_a]
        scan_b = scan_list[idx_b]

        sY = min(scan_a["sY"], scan_b["sY"])
        sX = min(scan_a["sX"], scan_b["sX"])

        # 3. Pick random physical coordinates (r, c)
        r = self.rng.randint(0, sY - 1)
        c = self.rng.randint(0, sX - 1)

        # 4. Extract star patches
        r_pad = r + self.pad
        c_pad = c + self.pad

        sample_r = r_pad + self.offsets[:, 0]
        sample_c = c_pad + self.offsets[:, 1]

        # Shape [25, C] reshaped to [5, 5, C]
        patch_a = scan_a["padded"][sample_r, sample_c, :].reshape(self.grid_size, self.grid_size, -1)
        patch_b = scan_b["padded"][sample_r, sample_c, :].reshape(self.grid_size, self.grid_size, -1)

        return {
            "x_a": torch.from_numpy(patch_a).float(),
            "x_b": torch.from_numpy(patch_b).float(),
            "ref_b": torch.from_numpy(scan_b["ref_patch"]).float(),
            "cond_a": torch.from_numpy(scan_a["cond"]),
            "cond_b": torch.from_numpy(scan_b["cond"]),
            "coord": torch.tensor([r, c], dtype=torch.long),
        }
