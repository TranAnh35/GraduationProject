"""
Dataset splitting protocols and metadata parsing for PECT TDMS files.

Two mutually exclusive source-domain protocols (selected by `split_mode`):

1. 'cross_specimen' (legacy, default for backward compatibility):
   Base Domain = Sensor in [Hall_Air_Core, Hall_Pot_Core], Waveform in
   [Square, Gaussian], Liftoff in [z1, z3], Specimen in [Corrosion, Rivet_v1]
   -> 16 files (12 Train / 4 Val); the remaining 65 files are Test
   (TMR sensor, Chirp waveform, z2 liftoff, Rivet_v2 specimen).

2. 'cross_condition' (adaptation to the measurement condition):
   Base Domain additionally contains Rivet_v2 base-condition scans, i.e.
   2 sensors x 3 specimens x 2 waveforms x 2 liftoffs = 24 files.
   Train/Val is stratified by specimen: 6 Train + 2 Val per specimen
   (18 Train / 6 Val); the remaining 57 files are Test (TMR, Chirp or z2 on
   all three specimens). Both protocols are deterministic given `seed`.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

SPLIT_MODES = ("cross_specimen", "cross_condition")

# Specimens that belong to the source (Base) Domain under each protocol.
_BASE_SPECIMENS = {
    "cross_specimen": ("Corrosion", "Rivet_v1"),
    "cross_condition": ("Corrosion", "Rivet_v1", "Rivet_v2"),
}
_BASE_SENSORS = ("Hall_Air_Core", "Hall_Pot_Core")
_BASE_WAVEFORMS = ("Square", "Gaussian")
_BASE_LIFTOFFS = ("z1", "z3")

# Val files per specimen under the stratified 'cross_condition' protocol.
_VAL_PER_SPECIMEN = 2


@dataclass
class ConditionMetadata:
    file_path: str
    sensor: str
    specimen: str
    waveform: str
    liftoff: str


def check_split_mode(split_mode: str) -> str:
    """Validates a split-mode name so a typo cannot silently change the protocol."""
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode '{split_mode}'; expected one of {SPLIT_MODES}.")
    return split_mode


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


def is_base_domain_file(meta: Dict[str, str], split_mode: str = "cross_specimen") -> bool:
    """
    Checks whether a file belongs to the Base (source) Domain.

    'cross_specimen' : Hall sensor, Square/Gaussian, z1/z3, Corrosion or Rivet_v1 (16 files).
    'cross_condition': same measurement conditions but all three specimens (24 files).
    """
    specimens = _BASE_SPECIMENS[check_split_mode(split_mode)]
    return (
        meta["sensor"] in _BASE_SENSORS and
        meta["waveform"] in _BASE_WAVEFORMS and
        meta["liftoff"] in _BASE_LIFTOFFS and
        meta["specimen"] in specimens
    )


def filter_base_domain_files(file_list: List[str], split_mode: str = "cross_specimen") -> List[str]:
    """Filters a list of file paths down to the Base Domain of the given protocol."""
    return [fp for fp in file_list if is_base_domain_file(extract_file_metadata(fp), split_mode)]


def _stratified_train_val(
    base_files: List[str],
    seed: int,
    val_per_specimen: int,
) -> Tuple[List[str], List[str]]:
    """Splits Base files into Train/Val keeping the specimen ratio identical.

    Each specimen contributes exactly `val_per_specimen` files to Val; the rest
    go to Train, so Train and Val have the same 1:1:1 specimen composition.
    Shuffling is seeded per specimen, hence deterministic and independent of the
    file ordering produced by glob().
    """
    by_specimen: Dict[str, List[str]] = {}
    for fp in base_files:
        by_specimen.setdefault(extract_file_metadata(fp)["specimen"], []).append(fp)

    train_files: List[str] = []
    val_files: List[str] = []
    for specimen in sorted(by_specimen):
        group = sorted(by_specimen[specimen])
        if len(group) <= val_per_specimen:
            raise ValueError(
                f"Specimen '{specimen}' has only {len(group)} Base-Domain file(s); "
                f"cannot hold out {val_per_specimen} for Val and keep at least one for Train."
            )
        random.Random(f"{seed}:{specimen}").shuffle(group)
        val_files.extend(group[:val_per_specimen])
        train_files.extend(group[val_per_specimen:])
    return sorted(train_files), sorted(val_files)


def get_canonical_train_val_test_splits(
    file_list: List[str],
    val_files_count: int = 4,
    seed: int = 42,
    split_mode: str = "cross_specimen",
    val_files_per_specimen: int = _VAL_PER_SPECIMEN,
) -> Dict[str, List[str]]:
    """
    Partitions the full TDMS dataset into deterministic Train/Val/Test splits.

    'cross_specimen': Base Domain (16 files) -> Train (12) / Val (`val_files_count`=4);
                      every non-Base file (TMR, Chirp, z2, Rivet_v2) -> Test (65).
    'cross_condition': Base Domain (24 files) -> Train (18) / Val (6), stratified by
                      specimen (`val_files_per_specimen`=2 each); every file with TMR,
                      Chirp or z2 on any specimen -> Test (57). `val_files_count` is
                      ignored in this mode because Val size is fixed per specimen.

    Returns:
        {"train": [...], "val": [...], "test": [...]} (all lists sorted).
    """
    check_split_mode(split_mode)
    base_files: List[str] = []
    test_files: List[str] = []
    for fp in file_list:
        if is_base_domain_file(extract_file_metadata(fp), split_mode):
            base_files.append(fp)
        else:
            test_files.append(fp)

    if split_mode == "cross_condition":
        train_files, val_files = _stratified_train_val(base_files, seed, val_files_per_specimen)
    else:
        rng = random.Random(seed)
        shuffled_base = sorted(base_files)
        rng.shuffle(shuffled_base)
        n_val = min(val_files_count, len(shuffled_base) // 4)
        val_files = sorted(shuffled_base[:n_val])
        train_files = sorted(shuffled_base[n_val:])

    return {
        "train": train_files,
        "val": val_files,
        "test": sorted(test_files),
    }


def get_canonical_shift_files(
    file_list: List[str],
    shift_mode: str,
    split_mode: str = "cross_specimen",
) -> Tuple[List[str], List[str]]:
    """
    Partitions files for a controlled single-variable domain adaptation scenario:
    - Source Domain: the entire Base Domain of `split_mode` (train + val).
    - Target Domain: expands ONLY one single condition, keeping the other three
      at their Base-Domain values:
      * 'cross_sensor':   Sensor == 'TMR'
      * 'cross_waveform': Waveform == 'Chirp'
      * 'cross_liftoff':  Liftoff == 'z2'
      * 'cross_specimen': Specimen == 'Rivet_v2' (only meaningful under
        split_mode='cross_specimen'; Rivet_v2 is part of the source domain
        under 'cross_condition').
    """
    check_split_mode(split_mode)
    if shift_mode == "cross_specimen" and split_mode == "cross_condition":
        raise ValueError(
            "shift_mode='cross_specimen' requires the legacy split_mode='cross_specimen'; "
            "under split_mode='cross_condition' Rivet_v2 is part of the source domain."
        )

    source_files = filter_base_domain_files(file_list, split_mode)
    source_set = set(source_files)

    target_files = []
    for fp in file_list:
        if fp in source_set:
            continue
        meta = extract_file_metadata(fp)
        s, w, z, sp = meta["sensor"], meta["waveform"], meta["liftoff"], meta["specimen"]
        if shift_mode == "cross_sensor":
            if s == "TMR" and w in _BASE_WAVEFORMS and z in _BASE_LIFTOFFS and sp in _BASE_SPECIMENS[split_mode]:
                target_files.append(fp)
        elif shift_mode == "cross_waveform":
            if w == "Chirp" and s in _BASE_SENSORS and z in _BASE_LIFTOFFS and sp in _BASE_SPECIMENS[split_mode]:
                target_files.append(fp)
        elif shift_mode == "cross_specimen":
            if sp == "Rivet_v2" and s in _BASE_SENSORS and w in _BASE_WAVEFORMS and z in _BASE_LIFTOFFS:
                target_files.append(fp)
        elif shift_mode == "cross_liftoff":
            if z == "z2" and s in _BASE_SENSORS and w in _BASE_WAVEFORMS and sp in _BASE_SPECIMENS[split_mode]:
                target_files.append(fp)

    return sorted(source_files), sorted(target_files)
