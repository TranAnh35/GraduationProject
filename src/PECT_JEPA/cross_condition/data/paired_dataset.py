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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .split import extract_file_metadata
from .topologies import get_spatial_topology_offsets
from .preprocessing import load_canonical_cscan_from_tdms, get_canonical_transformer


def context_sound_mask(gt_mask: np.ndarray, r: int, c: int,
                       offsets: np.ndarray, n_context: int = 16) -> np.ndarray:
    """Oracle sound mask for the N outer-ring context probes of a patch.

    Context probes are the LAST n_context of the 25 star offsets (idx
    9..24: core 0..8 are targets). Returns bool [n_context]: True where GT
    says sound-metal (gt == 0). Defect (1) AND ignore (<0) tokens are
    excluded. Coordinates clipped to mask bounds. Pure function (no TDMS)
    so Stage-0 mask geometry is directly unit-testable.
    """
    gt = np.asarray(gt_mask)
    H, W = gt.shape[:2]
    ctx_off = np.asarray(offsets)[-n_context:]
    rr = np.clip(int(r) + ctx_off[:, 0].astype(int), 0, H - 1)
    cc = np.clip(int(c) + ctx_off[:, 1].astype(int), 0, W - 1)
    return (gt[rr, cc] == 0)


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
        use_full_dataset: bool = False,
        grid_stride: int = 4,
        samples_per_epoch: Optional[int] = None,
        num_partitions: int = 10,
        topology: str = "concentric_star",
        star_radii: Tuple[int, int, int] = (1, 3, 7),
        grid_size: int = 5,
        temporal_samples: int = 128,
        transform_csv: str = "data/ground_truth/raw_tdms_rotate_crop.csv",
        seed: int = 42,
        include_sound_mask: bool = False,
        dynamic_condition_token: bool = False,
        ref_patch_mode: str = "legacy",
    ):
        super().__init__()
        self.num_partitions = max(1, int(num_partitions))
        self.seed = seed
        self.use_full_dataset = bool(use_full_dataset)
        import glob
        from .split import filter_base_domain_files

        if file_paths is None:
            if data_dir is None or not os.path.isdir(data_dir):
                for cand in [data_dir, "data", os.path.join("..", "..", "..", "data"), os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")]:
                    if cand and os.path.isdir(cand):
                        data_dir = cand
                        break
            if data_dir is None:
                data_dir = "data"
            all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.tdms"), recursive=True))
            all_files = [f for f in all_files if not f.endswith("_index.tdms") and "checkpoint" not in f]
            if self.use_full_dataset:
                file_paths = all_files
            else:
                file_paths = filter_base_domain_files(all_files)

        self.grid_size = grid_size
        self.temporal_samples = temporal_samples
        self.grid_stride = max(1, int(grid_stride))
        self.rng = random.Random(seed)

        # Stage-0 oracle sound masks (selective inv): per-specimen GT loaded
        # once here (2 x 270x270, trivial RAM). Only consumed when the model
        # flag inv_oracle_mask is on; otherwise __getitem__ skips the lookup.
        self.include_sound_mask = bool(include_sound_mask)
        self.dynamic_condition_token = bool(dynamic_condition_token)
        self.ref_patch_mode = str(ref_patch_mode)
        _gt_base = data_dir if data_dir is not None else "data"
        self.gt_by_specimen: Dict[str, np.ndarray] = {}
        if self.include_sound_mask:
            for _spec, _rel in (
                ("Corrosion", os.path.join("ground_truth", "corrosion", "corrosion_gt_mask.npy")),
                ("Rivet_v1", os.path.join("ground_truth", "rivet_v1", "rivet_v1_gt_mask.npy")),
                ("Rivet_v2", os.path.join("ground_truth", "rivet_v2", "rivet_v2_gt_mask.npy")),
            ):
                _p = os.path.join(_gt_base, _rel)
                if os.path.exists(_p):
                    self.gt_by_specimen[_spec] = np.load(_p)
            if len(self.gt_by_specimen) < 2:
                print(f"[WARN] include_sound_mask=True but only {len(self.gt_by_specimen)} GT masks found; "
                      f"samples without GT fall back to all-sound (blind inv).")

        self.offsets = get_spatial_topology_offsets(topology, star_radii, grid_size)
        self.max_off = int(np.max(np.abs(self.offsets)))
        self.pad = max(grid_size // 2, self.max_off)

        # Load scans and group by specimen (parallel TDMS decode: each file is
        # independent; order is preserved by mapping back onto file_paths, so
        # the resulting dataset is bit-identical to sequential loading)
        self.scans_by_specimen: Dict[str, List[Dict]] = {}
        ds_mode = "Full Cross-Condition Dataset (all 81 scans)" if self.use_full_dataset else "Base Domain"
        print(f"\n[Loading & Standardizing {len(file_paths)} {ds_mode} C-scans into RAM...]")

        # Pre-build the canonical transformer in the main thread (lazy global)
        get_canonical_transformer()
        _temporal = temporal_samples
        _pad = self.pad

        def _load_one(fpath: str):
            meta = extract_file_metadata(fpath)
            # Load canonical aligned C-scan (already rotated, flipped, and cropped to 270x270x128)
            cscan = load_canonical_cscan_from_tdms(
                file_path=fpath,
                temporal_samples=_temporal,
                apply_transforms=True,
            )
            sY, sX, C = cscan.shape
            padded = np.pad(cscan, ((_pad, _pad), (_pad, _pad), (0, 0)), mode="edge")
            # Background "sound metal" reference patch consumed by the condition token.
            # 'legacy': top-left 5x5, which sits inside the first defect site (15,15):
            #   measured 96% GT defect pixels on Corrosion (audit_ref_patch.py).
            # 'corner_median': pixel-wise median of the 4 corner patches -- 0% defect
            #   pixels on all three specimens, >=8.5 px from the nearest defect, and
            #   robust to the canonical rotation/flip of each file.
            if ref_patch_mode == "corner_median":
                _corners = (cscan[0:5, 0:5, :], cscan[0:5, -5:, :],
                            cscan[-5:, 0:5, :], cscan[-5:, -5:, :])
                ref_patch = np.median(np.stack(_corners, axis=0), axis=0).astype(cscan.dtype)
            else:
                ref_patch = cscan[10:15, 10:15, :].copy()
            return fpath, meta, {
                "file_path": fpath,
                "meta": meta,
                "sY": sY,
                "sX": sX,
                "C": C,
                "padded": padded,
                "ref_patch": ref_patch,
            }

        max_workers = min(4, max(1, len(file_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            loaded = list(tqdm(pool.map(_load_one, file_paths),
                               total=len(file_paths),
                               desc="Loading canonical C-scans",
                               dynamic_ncols=True))

        for fpath, meta, entry in loaded:
            spec = meta["specimen"]
            if spec not in self.scans_by_specimen:
                self.scans_by_specimen[spec] = []
            self.scans_by_specimen[spec].append(entry)

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

        if samples_per_epoch is not None:
            self.samples_per_epoch = int(samples_per_epoch)
            self.use_partitioning = False
            print(f"Dataset configured: Explicitly capped to {self.samples_per_epoch:,} samples/epoch (Universe: {self.total_exhaustive_samples:,}).")
        else:
            self.samples_per_epoch = self.total_exhaustive_samples // self.num_partitions
            self.use_partitioning = (self.num_partitions > 1)
            if self.use_partitioning:
                print(f"Dataset configured: MACRO-CYCLE PARTITIONING (K={self.num_partitions}) -> {self.samples_per_epoch:,} patch pairs per sub-epoch.")
                print(f"  Complete 100% traversal of {self.total_exhaustive_samples:,} universe achieved every {self.num_partitions} sub-epochs without duplicates or omissions.")
            else:
                print(f"Dataset configured: 100% EXHAUSTIVE COVERAGE -> {self.samples_per_epoch:,} patch pairs per epoch ({self.coords_per_pair:,} pixels x {num_pairs} condition pairs).")

        self.current_cycle_idx = -1
        self.current_cycle_perm = None
        self.active_indices = None
        self.set_epoch(1)

    def set_epoch(self, epoch: int):
        """
        Sets the active epoch to slice the exact partition of the current macro-cycle.
        Guarantees that across K consecutive epochs, 100% of the Cartesian universe is traversed exactly once.
        """
        if not getattr(self, "use_partitioning", False):
            return

        self.epoch = epoch
        cycle_idx = (epoch - 1) // self.num_partitions
        sub_epoch_idx = (epoch - 1) % self.num_partitions

        if cycle_idx != self.current_cycle_idx or self.current_cycle_perm is None:
            self.current_cycle_idx = cycle_idx
            # Fast reproducible permutation for this cycle
            rng = np.random.RandomState(self.seed + cycle_idx * 10007)
            self.current_cycle_perm = rng.permutation(self.total_exhaustive_samples)

        start = sub_epoch_idx * self.samples_per_epoch
        end = start + self.samples_per_epoch if sub_epoch_idx < (self.num_partitions - 1) else self.total_exhaustive_samples
        self.active_indices = self.current_cycle_perm[start:end]

    def __len__(self):
        if getattr(self, "use_partitioning", False) and self.active_indices is not None:
            return len(self.active_indices)
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        if getattr(self, "use_partitioning", False) and self.active_indices is not None:
            eff_idx = int(self.active_indices[idx])
        else:
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

        if self.dynamic_condition_token:
            # Dynamically sample a 5x5 patch from the margin [10:35, 10:35].
            # NOTE: for Corrosion the first defect site is (15,15) with a radius that
            # reaches rows/cols 10..14, so this window is NOT defect-free either
            # (audit_ref_patch.py). Prefer ref_patch_mode='corner_median' instead.
            # Identical physical coordinates for both Condition A and Condition B:
            ref_r = np.random.randint(10, 35) + self.pad
            ref_c = np.random.randint(10, 35) + self.pad
            ref_patch_a = scan_a["padded"][ref_r : ref_r + self.grid_size, ref_c : ref_c + self.grid_size, :]
            ref_patch_b = scan_b["padded"][ref_r : ref_r + self.grid_size, ref_c : ref_c + self.grid_size, :]
        else:
            ref_patch_a = scan_a["ref_patch"]
            ref_patch_b = scan_b["ref_patch"]

        sample = {
            "x_a": torch.from_numpy(patch_a).float(),
            "x_b": torch.from_numpy(patch_b).float(),
            "ref_a": torch.from_numpy(ref_patch_a).float(),
            "ref_b": torch.from_numpy(ref_patch_b).float(),
            "coord": torch.tensor([r, c], dtype=torch.long),
            "scan_id": torch.tensor(pair_idx, dtype=torch.long),
        }

        # Stage-0 oracle: sound mask over all 25 star probes (0..24), resolved
        # from the A-scan's specimen GT at the patch's probe coordinates.
        # Emitting [25] allows CrossConditionJEPA.forward to index via ctx_idx
        # for both legacy fixed masking (ctx_idx = 9..24) and contiguous_masking.
        # Missing GT -> all-True (blind inv for that sample, never a crash).
        if self.include_sound_mask:
            _gt = self.gt_by_specimen.get(scan_a["meta"]["specimen"])
            n_probes = len(self.offsets)
            if _gt is not None:
                _m = context_sound_mask(_gt, int(r), int(c), self.offsets, n_context=n_probes)
            else:
                _m = np.ones(n_probes, dtype=bool)
            sample["sound_mask"] = torch.from_numpy(_m)

        return sample
