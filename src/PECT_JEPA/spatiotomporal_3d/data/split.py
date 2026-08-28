"""
Dataset splitting protocols for PECT-JEPA.
Implements file-level and specimen-level splitting to prevent data leakage (Section 2.3),
as well as Cross-Sensor, Cross-Wave, and Cross-Lift-off splits (Section 18).
"""

import os
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


def split_by_files(
    file_paths: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Random split at full-file level to prevent intra-file data leakage.
    """
    rng = random.Random(seed)
    shuffled = list(file_paths)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train : n_train + n_val]
    test_files = shuffled[n_train + n_val :]

    return train_files, val_files, test_files


def split_cross_sensor(
    metadata_list: List[Dict[str, Any]],
    test_sensor: str = "TMR"
) -> Tuple[List[str], List[str]]:
    """
    Split files for Cross-Sensor generalization evaluation (Section 18.2).
    """
    train_files = []
    test_files = []
    for meta in metadata_list:
        if meta["sensor"].lower() == test_sensor.lower():
            test_files.append(meta["file_path"])
        else:
            train_files.append(meta["file_path"])
    return train_files, test_files


def split_cross_waveform(
    metadata_list: List[Dict[str, Any]],
    test_waveform: str = "Chirp"
) -> Tuple[List[str], List[str]]:
    """
    Split files for Cross-Wave generalization evaluation (Section 18.3).
    """
    train_files = []
    test_files = []
    for meta in metadata_list:
        if meta["waveform"].lower() == test_waveform.lower():
            test_files.append(meta["file_path"])
        else:
            train_files.append(meta["file_path"])
    return train_files, test_files


def split_cross_liftoff(
    metadata_list: List[Dict[str, Any]],
    test_liftoff: str = "z3"
) -> Tuple[List[str], List[str]]:
    """
    Split files for Cross-Lift-off generalization evaluation (Section 18.4).
    """
    train_files = []
    test_files = []
    for meta in metadata_list:
        if meta["liftoff"].lower() == test_liftoff.lower():
            test_files.append(meta["file_path"])
        else:
            train_files.append(meta["file_path"])
    return train_files, test_files


def split_by_specimen(
    metadata_list: List[Dict[str, Any]],
    test_specimen: str = "Rivet_v2"
) -> Tuple[List[str], List[str]]:
    """
    Split files holding out a complete specimen type.
    """
    train_files = []
    test_files = []
    for meta in metadata_list:
        if meta["specimen"].lower() == test_specimen.lower():
            test_files.append(meta["file_path"])
        else:
            train_files.append(meta["file_path"])
    return train_files, test_files
