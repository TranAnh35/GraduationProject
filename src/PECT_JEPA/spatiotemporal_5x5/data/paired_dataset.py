"""
Paired Cross-Waveform Dataset for PECT-JEPA v2-A.
Extracts coordinate-matched patches (x_A, x_B) across different excitation waveforms (Square, Gauss, Chirp)
at identical spatial probe coordinates on the specimen.

Zero external metadata labels injected; preserves 100% self-supervised data pipeline.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from src.PECT_JEPA.spatiotemporal_5x5.data.dataset import PECT5x5Dataset
from src.PECT_JEPA.spatiotemporal_5x5.data.split import extract_file_metadata


class PairedWaveformPECT5x5Dataset(Dataset):
    """
    Pairs coordinate-matched 5x5 patches from two different waveforms (e.g. Square and Chirp)
    at the exact same spatial grid locations (row, col).
    """
    def __init__(
        self,
        base_dataset: PECT5x5Dataset,
        waveform_pair: Tuple[str, str] = ("Square", "Chirp"),
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.wf_A, self.wf_B = waveform_pair

        # Group sample indices by (specimen, liftoff, row, col) to find coordinate matches
        print(f"Indexing paired samples for {self.wf_A} <-> {self.wf_B}...")
        self.pairs: List[Tuple[int, int]] = []
        self._build_coordinate_pairs()
        print(f"Found {len(self.pairs)} exactly matched spatial pairs out of {len(self.base_dataset)} samples.")

    def _build_coordinate_pairs(self):
        index_map = {}
        for idx in range(len(self.base_dataset)):
            # Extract metadata and spatial coordinate
            file_path = self.base_dataset.file_paths[self.base_dataset.file_indices[idx]]
            meta = extract_file_metadata(file_path)
            wf = meta["waveform"]
            if wf not in (self.wf_A, self.wf_B):
                continue

            r = self.base_dataset.row_indices[idx]
            c = self.base_dataset.col_indices[idx]
            sp = meta["specimen"]
            lo = meta["liftoff"]
            key = (sp, lo, r, c)

            if key not in index_map:
                index_map[key] = {}
            index_map[key][wf] = idx

        for key, wfs in index_map.items():
            if self.wf_A in wfs and self.wf_B in wfs:
                self.pairs.append((wfs[self.wf_A], wfs[self.wf_B]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx_A, idx_B = self.pairs[idx]
        sample_A = self.base_dataset[idx_A]
        sample_B = self.base_dataset[idx_B]

        return {
            "data_A": sample_A["data"],          # [5, 5, C]
            "data_B": sample_B["data"],          # [5, 5, C]
            "file_id_A": sample_A.get("file_id", 0),
            "file_id_B": sample_B.get("file_id", 0),
        }


def collate_paired_5x5_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    data_A = torch.stack([item["data_A"] for item in batch], dim=0)
    data_B = torch.stack([item["data_B"] for item in batch], dim=0)
    file_ids_A = torch.tensor([item["file_id_A"] for item in batch], dtype=torch.long)
    file_ids_B = torch.tensor([item["file_id_B"] for item in batch], dtype=torch.long)

    return {
        "data_A": data_A,
        "data_B": data_B,
        "file_ids_A": file_ids_A,
        "file_ids_B": file_ids_B,
    }
