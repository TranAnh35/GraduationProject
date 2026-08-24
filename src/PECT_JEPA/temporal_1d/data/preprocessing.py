"""
Preprocessing and TDMS Reading for 1D Temporal PECT-JEPA.
Completely self-contained: reads TDMS files, extracts 1D transient waveforms,
resamples to target time points, and applies Min-Max per-file normalization.
"""

import os
import glob
import re
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from nptdms import TdmsFile


def parse_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Parse experiment metadata (sensor, waveform, defect type, liftoff) from file path and filename.
    """
    norm_path = os.path.normpath(file_path).replace("\\", "/")
    filename = os.path.basename(norm_path)

    # 1. Sensor type
    sensor = "Unknown"
    sensor_map = {
        "Differential_Pot_Core": "Differential_Pot_Core",
        "Hall_Air_Core": "Hall_Air_Core",
        "Hall_Pot_Core": "Hall_Pot_Core",
        "TMR": "TMR"
    }
    for key, val in sensor_map.items():
        if key.lower() in norm_path.lower():
            sensor = val
            break

    # 2. Excitation waveform
    waveform = "Unknown"
    waveform_map = {
        "Square": "Square",
        "Gaussian": "Gaussian",
        "Chirp": "Chirp"
    }
    for key, val in waveform_map.items():
        if key.lower() in norm_path.lower():
            waveform = val
            break

    # 3. Lift-off
    liftoff = "Unknown"
    liftoff_match = re.search(r'(\d+mm)', norm_path, re.IGNORECASE)
    if liftoff_match:
        liftoff = liftoff_match.group(1).lower()
    else:
        for lo in ["3mm", "2mm", "1mm", "z1", "z2", "z3"]:
            if lo in norm_path.lower():
                liftoff = lo
                break

    # 4. Defect type
    defect = "Unknown"
    if "defect" in norm_path.lower():
        defect = "Defect"
    elif "normal" in norm_path.lower() or "reference" in norm_path.lower():
        defect = "Normal"

    return {
        "file_path": file_path,
        "filename": filename,
        "sensor": sensor,
        "waveform": waveform,
        "liftoff": liftoff,
        "defect": defect
    }


def find_all_tdms_files(data_dir: str) -> List[str]:
    """
    Recursively search for all .tdms files in data_dir.
    Excludes .tdms_index files.
    """
    if not os.path.exists(data_dir):
        return []

    pattern = os.path.join(data_dir, "**", "*.tdms")
    files = glob.glob(pattern, recursive=True)
    valid_files = [f for f in files if not f.endswith(".tdms_index")]
    return sorted(valid_files)


def read_tdms_1d_waveforms(
    file_path: str,
    target_time_samples: int = 500,
    normalization: str = "min_max",
    raster_correction: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Reads a TDMS file and extracts all 1D transient waveforms as a 2D array [N_waveforms, target_time_samples].

    Args:
        file_path: Path to .tdms file
        target_time_samples: Resample length (default: 500)
        normalization: 'min_max' or 'none'
        raster_correction: Whether to rearrange bidirectional raster lines
        eps: Small epsilon for numerical stability

    Returns:
        waveforms: np.ndarray of shape [N_waveforms, target_time_samples] (float32)
    """
    tdms_file = TdmsFile.read(file_path)
    groups = tdms_file.groups()
    if len(groups) == 0:
        raise ValueError(f"No groups found in TDMS file: {file_path}")

    # Extract all channels
    channels = groups[0].channels()
    if len(channels) == 0:
        raise ValueError(f"No channels found in TDMS group: {groups[0].name}")

    # Read channel arrays
    raw_list = [ch.data[:] for ch in channels]
    raw_2d = np.array(raw_list, dtype=np.float32)  # [n_channels, raw_time_samples]

    # If shape is [sY * sX, T_raw], rearrange raster if needed
    N_ch, T_raw = raw_2d.shape

    # Determine spatial grid if square scan (e.g. 300x300 = 90000 channels)
    side = int(np.round(np.sqrt(N_ch)))
    if side * side == N_ch and raster_correction:
        cube = raw_2d.reshape(side, side, T_raw)
        # Reverse odd rows
        cube[1::2, :, :] = cube[1::2, ::-1, :]
        raw_2d = cube.reshape(N_ch, T_raw)

    # Resample temporal dimension if needed
    if T_raw != target_time_samples:
        indices = np.linspace(0, T_raw - 1, target_time_samples).astype(np.int64)
        raw_2d = raw_2d[:, indices]

    # Min-Max Normalization per-file
    if normalization == "min_max":
        f_min = float(np.min(raw_2d))
        f_max = float(np.max(raw_2d))
        raw_2d = (raw_2d - f_min) / (f_max - f_min + eps)

    return raw_2d.astype(np.float32)
