"""
Dataset splitting protocols and metadata parsing for PECT TDMS files.

Protocols:
1. Base Domain vs Held-out OOD Domain:
   - Base Domain: Sensor in [Hall_Air_Core, Hall_Pot_Core], Waveform in [Square, Gaussian], Liftoff in [z1, z3], Specimen in [Corrosion, Rivet_v1] (16 files).
   - Base Train Pool: 12-14 files.
   - Base Val Pool: 2-4 files.
   - Test (OOD) Domain: Remaining files (TMR sensor, Chirp waveform, z2 liftoff, Rivet_v2 specimen) -> 65-67 files.
2. Helper functions to classify any file into split: 'train', 'val', 'test'.
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple


@dataclass
class ConditionMetadata:
    file_path: str
    sensor: str
    specimen: str
    waveform: str
    liftoff: str


def extract_file_metadata(file_path: str) -> Dict[str, str]:
    """
    Extracts sensor, specimen, waveform, and lift-off metadata from file path.
    """
    p = file_path.replace("\\", "/")

    # Sensor
    sensor = "unknown"
    if "Hall_Air_Core" in p or "HallAirCore" in p:
        sensor = "Hall_Air_Core"
    elif "Hall_Pot_Core" in p or "HallPotCore" in p:
        sensor = "Hall_Pot_Core"
    elif "TMR" in p:
        sensor = "TMR"

    # Specimen
    specimen = "unknown"
    if "Corrosion" in p:
        specimen = "Corrosion"
    elif "Rivet_v1" in p or "/Rivet/" in p:
        specimen = "Rivet_v1"
    elif "Rivet_v2" in p or "Mixed" in p:
        specimen = "Rivet_v2"

    # Waveform
    waveform = "unknown"
    p_lower = p.lower()
    if "chirp" in p_lower:
        waveform = "Chirp"
    elif "gauss" in p_lower:
        waveform = "Gaussian"
    elif "square" in p_lower:
        waveform = "Square"

    # Liftoff
    liftoff = "unknown"
    if "_z1" in p_lower or "z1mm" in p_lower or "z1_" in p_lower:
        liftoff = "z1"
    elif "_z2" in p_lower or "z2mm" in p_lower or "z2_" in p_lower:
        liftoff = "z2"
    elif "_z3" in p_lower or "z3mm" in p_lower or "z3_" in p_lower:
        liftoff = "z3"

    return {
        "file_path": file_path,
        "sensor": sensor,
        "specimen": specimen,
        "waveform": waveform,
        "liftoff": liftoff,
    }


def get_dataset_condition_metadata(file_path: str) -> ConditionMetadata:
    meta = extract_file_metadata(file_path)
    return ConditionMetadata(**meta)


def is_base_domain_file(meta: Dict[str, str]) -> bool:
    """
    Checks if a file belongs to the 16-file Base Training Domain:
    - Sensor: Hall_Air_Core or Hall_Pot_Core
    - Waveform: Square or Gaussian
    - Liftoff: z1 or z3
    - Specimen: Corrosion or Rivet_v1
    """
    return (
        meta["sensor"] in ["Hall_Air_Core", "Hall_Pot_Core"] and
        meta["waveform"] in ["Square", "Gaussian"] and
        meta["liftoff"] in ["z1", "z3"] and
        meta["specimen"] in ["Corrosion", "Rivet_v1"]
    )


def filter_base_domain_files(file_list: List[str]) -> List[str]:
    """Filters list of file paths to only those in the Base Domain."""
    base_files = []
    for fp in file_list:
        meta = extract_file_metadata(fp)
        if is_base_domain_file(meta):
            base_files.append(fp)
    return base_files


def get_canonical_train_val_test_splits(
    file_list: List[str],
    val_files_count: int = 4,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Partitions full dataset into Train, Val, and Test splits:
    - Base Domain (16 files): split deterministically into Train (12 files) and Val (4 files).
    - Held-out Domain (65 files): all files with TMR, Chirp, z2, or Rivet_v2 -> Test.
    """
    base_files = []
    test_files = []

    for fp in file_list:
        meta = extract_file_metadata(fp)
        if is_base_domain_file(meta):
            base_files.append(fp)
        else:
            test_files.append(fp)

    rng = random.Random(seed)
    shuffled_base = sorted(base_files)
    rng.shuffle(shuffled_base)

    n_val = min(val_files_count, len(shuffled_base) // 4)
    val_files = sorted(shuffled_base[:n_val])
    train_files = sorted(shuffled_base[n_val:])
    test_files = sorted(test_files)

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def get_canonical_shift_files(
    file_list: List[str],
    shift_mode: str,
) -> Tuple[List[str], List[str]]:
    """
    Partitions files for a controlled single-variable domain adaptation scenario:
    - Source Domain (16 files): Entire Base Domain (train + val).
      [Sensors: Air/Pot, Waveforms: Square/Gauss, Liftoff: z1/z3, Specimens: Corrosion/Rivet_v1].
    - Target Domain (8 files): Expands ONLY 1 single condition, keeping all other 3 in Base Domain:
      * 'cross_sensor':   Sensor == 'TMR' (8 files)
      * 'cross_waveform': Waveform == 'Chirp' (8 files)
      * 'cross_specimen': Specimen == 'Rivet_v2' (8 files)
      * 'cross_liftoff':  Liftoff == 'z2' (8 files)
    """
    base_files = [fp for fp in file_list if is_base_domain_file(extract_file_metadata(fp))]
    source_files = sorted(base_files)

    target_files = []
    for fp in file_list:
        meta = extract_file_metadata(fp)
        s, w, z, sp = meta["sensor"], meta["waveform"], meta["liftoff"], meta["specimen"]
        if shift_mode == "cross_sensor":
            if s == "TMR" and w in ["Square", "Gaussian"] and z in ["z1", "z3"] and sp in ["Corrosion", "Rivet_v1"]:
                target_files.append(fp)
        elif shift_mode == "cross_waveform":
            if w == "Chirp" and s in ["Hall_Air_Core", "Hall_Pot_Core"] and z in ["z1", "z3"] and sp in ["Corrosion", "Rivet_v1"]:
                target_files.append(fp)
        elif shift_mode == "cross_specimen":
            if sp == "Rivet_v2" and s in ["Hall_Air_Core", "Hall_Pot_Core"] and w in ["Square", "Gaussian"] and z in ["z1", "z3"]:
                target_files.append(fp)
        elif shift_mode == "cross_liftoff":
            if z == "z2" and s in ["Hall_Air_Core", "Hall_Pot_Core"] and w in ["Square", "Gaussian"] and sp in ["Corrosion", "Rivet_v1"]:
                target_files.append(fp)

    return source_files, sorted(target_files)
