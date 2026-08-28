"""
Dataset Splitting Protocols for 1D Temporal PECT-JEPA.
Completely self-contained: provides file-level splitting (zero intra-file leakage)
and downstream protocol splits (Cross-Sensor, Cross-Wave, Cross-Lift-off).
"""

import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


def split_by_files(
    file_paths: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Random split at full-file level to prevent intra-file data leakage.
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


def split_cross_sensor(
    metadata_list: List[Dict[str, Any]],
    test_sensor: str = "TMR"
) -> Tuple[List[str], List[str]]:
    """
    Split files so test_sensor is held out for evaluation, remaining sensors used for training.
    """
    train_files = []
    test_files = []

    for meta in metadata_list:
        fp = meta["file_path"]
        if meta["sensor"].lower() == test_sensor.lower():
            test_files.append(fp)
        else:
            train_files.append(fp)

    return train_files, test_files


def split_cross_waveform(
    metadata_list: List[Dict[str, Any]],
    test_waveform: str = "Chirp"
) -> Tuple[List[str], List[str]]:
    """
    Hold out test_waveform for evaluation, train on other waveforms.
    """
    train_files = []
    test_files = []

    for meta in metadata_list:
        fp = meta["file_path"]
        if meta["waveform"].lower() == test_waveform.lower():
            test_files.append(fp)
        else:
            train_files.append(fp)

    return train_files, test_files


def split_cross_liftoff(
    metadata_list: List[Dict[str, Any]],
    test_liftoff: str = "z3"
) -> Tuple[List[str], List[str]]:
    """
    Hold out test_liftoff (e.g. 'z1', 'z2', 'z3') for evaluation, train on remaining lift-off levels.
    """
    train_files = []
    test_files = []

    for meta in metadata_list:
        fp = meta["file_path"]
        if meta["liftoff"].lower() == test_liftoff.lower() or test_liftoff.lower() in meta["liftoff"].lower():
            test_files.append(fp)
        else:
            train_files.append(fp)

    return train_files, test_files
