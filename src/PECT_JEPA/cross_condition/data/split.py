"""
Dataset splitting protocols and metadata parsing for PECT TDMS files.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Any


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
