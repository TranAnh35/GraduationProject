"""
Comprehensive Dataset Splitting Protocols for 5x5 Spatiotemporal PECT-JEPA.

Supports 5 rigorous scientific protocols:
1. 'leave_liftoff' (LOLO): Leave-One-Lift-off-Out (e.g. train on z1, z2 -> test on z3).
2. 'leave_sensor' (LOSO): Leave-One-Sensor-Out (e.g. train on Hall Air/Pot -> test on TMR).
3. 'leave_waveform' (LOWO): Leave-One-Waveform-Out (e.g. train on Square, Gauss -> test on Chirp).
4. 'leave_specimen' (LODO): Leave-One-Specimen-Out (e.g. train on Corrosion, Rivet_v1 -> test on Rivet_v2).
5. 'random' (In-Domain Baseline): Random file-level split (e.g. 80% train, 10% val, 10% test).

Guarantees 0 intra-file leakage and zero target domain contamination.
"""

import os
import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any


def extract_file_metadata(file_path: str) -> Dict[str, str]:
    """
    Extract sensor, specimen, waveform, and liftoff metadata from file path.
    """
    norm = os.path.normpath(file_path).replace("\\", "/")
    filename = os.path.basename(norm)

    # 1. Sensor
    sensor = "Unknown"
    for s in ["Hall_Air_Core", "Hall_Pot_Core", "TMR", "Differential_Pot_Core"]:
        if s.lower() in norm.lower():
            sensor = s
            break

    # 2. Specimen
    specimen = "Unknown"
    for sp in ["Rivet_v1", "Rivet_v2", "Corrosion"]:
        if sp.lower() in norm.lower():
            specimen = sp
            break

    # 3. Waveform
    waveform = "Unknown"
    for wf in ["Chirp", "Gaussian", "Gauss", "Square"]:
        if wf.lower() in norm.lower():
            waveform = "Gaussian" if wf.lower() in ("gaussian", "gauss") else wf
            break

    # 4. Lift-off (z1, z2, z3)
    liftoff = "Unknown"
    m = re.search(r"z([123])", norm.lower())
    if m:
        liftoff = f"z{m.group(1)}"

    return {
        "file_path": file_path,
        "filename": filename,
        "sensor": sensor,
        "specimen": specimen,
        "waveform": waveform,
        "liftoff": liftoff,
    }


def split_leave_one_liftoff(
    file_paths: List[str],
    test_liftoff: str = "z3",
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Leave-One-Lift-off-Out (LOLO):
    Holds out all files with `test_liftoff` for testing.
    Splits remaining files into training and validation.
    """
    train_pool = []
    test_files = []

    for fp in file_paths:
        meta = extract_file_metadata(fp)
        if meta["liftoff"].lower() == test_liftoff.lower():
            test_files.append(fp)
        else:
            train_pool.append(fp)

    rng = random.Random(seed)
    rng.shuffle(train_pool)

    n_val = max(1, int(len(train_pool) * val_ratio))
    train_files = train_pool[n_val:]
    val_files = train_pool[:n_val]

    return train_files, val_files, test_files


def split_leave_one_sensor(
    file_paths: List[str],
    test_sensor: str = "TMR",
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Leave-One-Sensor-Out (LOSO):
    Holds out all files with `test_sensor` for testing.
    Splits remaining files into training and validation.
    """
    train_pool = []
    test_files = []

    for fp in file_paths:
        meta = extract_file_metadata(fp)
        if meta["sensor"].lower() == test_sensor.lower():
            test_files.append(fp)
        else:
            train_pool.append(fp)

    rng = random.Random(seed)
    rng.shuffle(train_pool)

    n_val = max(1, int(len(train_pool) * val_ratio))
    train_files = train_pool[n_val:]
    val_files = train_pool[:n_val]

    return train_files, val_files, test_files


def split_leave_one_waveform(
    file_paths: List[str],
    test_waveform: str = "Chirp",
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Leave-One-Waveform-Out (LOWO):
    Holds out all files with `test_waveform` for testing.
    Splits remaining files into training and validation.
    """
    train_pool = []
    test_files = []

    for fp in file_paths:
        meta = extract_file_metadata(fp)
        if meta["waveform"].lower() == test_waveform.lower():
            test_files.append(fp)
        else:
            train_pool.append(fp)

    rng = random.Random(seed)
    rng.shuffle(train_pool)

    n_val = max(1, int(len(train_pool) * val_ratio))
    train_files = train_pool[n_val:]
    val_files = train_pool[:n_val]

    return train_files, val_files, test_files


def split_leave_one_specimen(
    file_paths: List[str],
    test_specimen: str = "Rivet_v2",
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Leave-One-Specimen-Out (LODO):
    Holds out all files with `test_specimen` for testing.
    Splits remaining files into training and validation.
    """
    train_pool = []
    test_files = []

    for fp in file_paths:
        meta = extract_file_metadata(fp)
        if meta["specimen"].lower() == test_specimen.lower():
            test_files.append(fp)
        else:
            train_pool.append(fp)

    rng = random.Random(seed)
    rng.shuffle(train_pool)

    n_val = max(1, int(len(train_pool) * val_ratio))
    train_files = train_pool[n_val:]
    val_files = train_pool[:n_val]

    return train_files, val_files, test_files


def split_by_files(
    file_paths: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Random split at full-file level to prevent intra-file data leakage.
    Maintained for backward compatibility.
    """
    rng = random.Random(seed)
    shuffled = list(file_paths)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total <= 1:
        return shuffled, shuffled, []

    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val)
    if n_train + n_val > n_total:
        n_val = max(1, n_total - n_train)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train: n_train + n_val]
    test_files = shuffled[n_train + n_val:]
    if len(val_files) == 0:
        val_files = [shuffled[-1]]

    return train_files, val_files, test_files


def split_compound_ood(
    file_paths: List[str],
    holdout_liftoff: str = "z3",
    holdout_sensor: str = "TMR",
    holdout_waveform: str = "Chirp",
    val_files_count: int = 4,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    """
    Compound Out-of-Distribution (OOD) Split Protocol (Option A):
    - Base Domain (24 files when all 81 are present):
      Excluded from holdout_liftoff ('z3'), holdout_sensor ('TMR'), and holdout_waveform ('Chirp').
      Includes all specimens ('Corrosion', 'Rivet_v1', 'Rivet_v2').
      Split into Train (e.g. 20 files) and In-Domain Validation (e.g. 4 files).
    - Held-out OOD Domain (57 files):
      All files containing at least one of the holdout factors.
      Sub-categorized into granular slices for detailed OOD generalization probing:
      - single_liftoff (12 files): Only liftoff == holdout_liftoff
      - single_sensor (12 files): Only sensor == holdout_sensor
      - single_waveform (12 files): Only waveform == holdout_waveform
      - compound_double (18 files): Exactly 2 shifted factors
      - compound_triple (3 files): All 3 shifted factors (Extreme Zero-shot OOD)
    """
    base_files = []
    test_files = []
    slices: Dict[str, List[str]] = {
        "single_liftoff": [],
        "single_sensor": [],
        "single_waveform": [],
        "compound_double": [],
        "compound_triple": [],
    }

    for fp in file_paths:
        m = extract_file_metadata(fp)
        is_lo = (m.get("liftoff") == holdout_liftoff)
        is_sensor = (m.get("sensor") == holdout_sensor)
        is_wave = (m.get("waveform") == holdout_waveform)

        shifted_count = int(is_lo) + int(is_sensor) + int(is_wave)

        if shifted_count == 0:
            base_files.append(fp)
        else:
            test_files.append(fp)
            if shifted_count == 1:
                if is_lo:
                    slices["single_liftoff"].append(fp)
                elif is_sensor:
                    slices["single_sensor"].append(fp)
                elif is_wave:
                    slices["single_waveform"].append(fp)
            elif shifted_count == 2:
                slices["compound_double"].append(fp)
            elif shifted_count == 3:
                slices["compound_triple"].append(fp)

    rng = random.Random(seed)
    shuffled_base = list(base_files)
    rng.shuffle(shuffled_base)

    n_val = min(val_files_count, max(1, len(shuffled_base) // 4)) if len(shuffled_base) > 1 else 0
    val_files = shuffled_base[:n_val]
    train_files = shuffled_base[n_val:]

    summary = {
        "protocol": "compound_ood",
        "holdout_liftoff": holdout_liftoff,
        "holdout_sensor": holdout_sensor,
        "holdout_waveform": holdout_waveform,
        "n_total": len(file_paths),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "n_test": len(test_files),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "test_slices": slices,
        "slice_counts": {k: len(v) for k, v in slices.items()},
    }

    return train_files, val_files, test_files, summary


def get_dataset_split(
    file_paths: List[str],
    protocol: str = "compound_ood",
    holdout_target: Optional[str] = None,
    holdout_liftoff: str = "z3",
    holdout_sensor: str = "TMR",
    holdout_waveform: str = "Chirp",
    val_ratio: float = 0.1,
    val_files_count: int = 4,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    """
    Unified Split Dispatcher for 5x5 PECT-JEPA.

    Args:
        file_paths: list of all available TDMS files
        protocol: 'compound_ood', 'leave_liftoff', 'leave_sensor', 'leave_waveform', 'leave_specimen', or 'random'
        holdout_target: target category to hold out for single-factor protocols
        holdout_liftoff: lift-off to hold out for compound protocol (default 'z3')
        holdout_sensor: sensor to hold out for compound protocol (default 'TMR')
        holdout_waveform: waveform to hold out for compound protocol (default 'Chirp')
        val_ratio: ratio of training pool used for validation (for single-factor protocols)
        val_files_count: number of validation files for compound protocol (default 4)
        seed: random seed for reproducibility

    Returns:
        (train_files, val_files, test_files, summary_info_dict)
    """
    proto = protocol.lower().strip()

    if proto in ("compound_ood", "tri_ood", "multi_ood", "option_a"):
        lo = holdout_liftoff or "z3"
        sensor = holdout_sensor or "TMR"
        wave = holdout_waveform or "Chirp"
        return split_compound_ood(
            file_paths,
            holdout_liftoff=lo,
            holdout_sensor=sensor,
            holdout_waveform=wave,
            val_files_count=val_files_count,
            seed=seed,
        )
    elif proto in ("leave_liftoff", "lolo"):
        target = holdout_target or "z3"
        train, val, test = split_leave_one_liftoff(file_paths, test_liftoff=target, val_ratio=val_ratio, seed=seed)
    elif proto in ("leave_sensor", "loso"):
        target = holdout_target or "TMR"
        train, val, test = split_leave_one_sensor(file_paths, test_sensor=target, val_ratio=val_ratio, seed=seed)
    elif proto in ("leave_waveform", "lowo"):
        target = holdout_target or "Chirp"
        train, val, test = split_leave_one_waveform(file_paths, test_waveform=target, val_ratio=val_ratio, seed=seed)
    elif proto in ("leave_specimen", "lodo"):
        target = holdout_target or "Rivet_v2"
        train, val, test = split_leave_one_specimen(file_paths, test_specimen=target, val_ratio=val_ratio, seed=seed)
    elif proto in ("random", "in_domain"):
        target = "random_80_20"
        train, val, test = split_by_files(file_paths, train_ratio=0.8, val_ratio=val_ratio, seed=seed)
        if not test:
            # If test empty, reserve 10% from val or train
            n_test = max(1, int(len(val) / 2))
            test = val[:n_test]
            val = val[n_test:]
    else:
        raise ValueError(
            f"Unknown split protocol: {protocol}. Supported: 'compound_ood', 'leave_liftoff', "
            f"'leave_sensor', 'leave_waveform', 'leave_specimen', 'random'."
        )

    summary = {
        "protocol": proto,
        "holdout_target": target,
        "n_total": len(file_paths),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_files": train,
        "val_files": val,
        "test_files": test,
    }

    return train, val, test, summary


# Backward compatibility aliases
split_cross_sensor = split_leave_one_sensor
split_cross_waveform = split_leave_one_waveform
split_cross_liftoff = split_leave_one_liftoff


__all__ = [
    "extract_file_metadata",
    "split_compound_ood",
    "split_leave_one_liftoff",
    "split_leave_one_sensor",
    "split_leave_one_waveform",
    "split_leave_one_specimen",
    "split_by_files",
    "get_dataset_split",
    "split_cross_sensor",
    "split_cross_waveform",
    "split_cross_liftoff",
]


