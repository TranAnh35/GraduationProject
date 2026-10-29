"""Matched-factor protocols. Labels are never needed to build these roles."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from tqdm import tqdm

from ..data.split import extract_file_metadata
from .common import MAIN_BAR_FORMAT, digest, sha256

FACTORS = ("sensor", "waveform", "liftoff", "specimen")
SCENARIOS = {
    "air_tmr": ("sensor", "Hall_Air_Core", "TMR"),
    "pot_tmr": ("sensor", "Hall_Pot_Core", "TMR"),
    "square_chirp": ("waveform", "Square", "Chirp"),
    "gaussian_chirp": ("waveform", "Gaussian", "Chirp"),
    "z1_z3": ("liftoff", "z1", "z3"),
    "z3_z1": ("liftoff", "z3", "z1"),
}


def inventory(data_dir, transform_csv, verbose=True):
    data_dir = Path(data_dir).resolve()
    files = []
    candidates = sorted(data_dir.rglob("*.tdms"))
    iterator = tqdm(candidates, desc="Plan: hash TDMS", unit="files",
                    dynamic_ncols=True, mininterval=0.2,
                    bar_format=MAIN_BAR_FORMAT, disable=not verbose)
    for p in iterator:
        if p.name.endswith("_index.tdms") or "checkpoint" in str(p):
            continue
        m = extract_file_metadata(str(p))
        # Differential pot scans are a different acquisition family, not one
        # of the 81 factorial cells used by this protocol.
        if "Diferential_Pot_Core" in p.parts or "Differential_Pot_Core" in p.parts:
            continue
        if any(m[k] in (None, "Unknown", "unknown") for k in FACTORS):
            raise ValueError(f"Unrecognized metadata: {p}: {m}")
        files.append({"path": str(p), "sha256": sha256(p),
                      **{k: m[k] for k in FACTORS}})
    tuples = Counter(tuple(f[k] for k in FACTORS) for f in files)
    if len(files) != 81 or len(tuples) != 81 or any(n != 1 for n in tuples.values()):
        raise ValueError(f"Expected 81 unique factorial cells, got {len(files)}/{len(tuples)}")
    return {"files": files, "transform_csv": str(Path(transform_csv).resolve()),
            "transform_sha256": sha256(transform_csv)}


def matched_roles(files, scenario):
    if scenario == "compound":
        source = [f for f in files if f["sensor"] in ("Hall_Air_Core", "Hall_Pot_Core")
                  and f["waveform"] in ("Square", "Gaussian") and f["liftoff"] in ("z1", "z3")]
        val = []
        for specimen in sorted({f["specimen"] for f in source}):
            group = sorted(f["path"] for f in source if f["specimen"] == specimen)
            val.extend(group[i] for i in (1, 5))
        paths = [f["path"] for f in source]
        target = [f["path"] for f in files if f["path"] not in paths]
        return {"name": scenario, "axis": "compound", "source": paths,
                "fit": [p for p in paths if p not in val], "val": val, "target": target,
                "blocks": [], "claim": "mixed single/multiple shifts; not a sensor causal effect",
                "validation": "source-only whole-file"}
    axis, source, target = SCENARIOS[scenario]
    others = tuple(k for k in FACTORS if k != axis)
    src = {tuple(f[k] for k in others): f for f in files if f[axis] == source}
    tgt = {tuple(f[k] for k in others): f for f in files if f[axis] == target}
    if len(src) != 27 or src.keys() != tgt.keys():
        raise ValueError(f"Incomplete matched scenario {scenario}")
    blocks = [{"id": digest(key)[:12], "conditions": dict(zip(others, key)),
               "source": src[key]["path"], "target": tgt[key]["path"]}
              for key in sorted(src)]
    # Whole-file source validation: common deterministic nuisance blocks. Same
    # blocks for every method/seed. The corresponding target blocks remain test.
    val = []
    for specimen in sorted({f["specimen"] for f in files}):
        group = [b["source"] for b in blocks
                 if next(f for f in files if f["path"] == b["source"])["specimen"] == specimen]
        val.extend(group[i] for i in (1, 5))
    fit = [b["source"] for b in blocks if b["source"] not in val]
    return {"name": scenario, "axis": axis, "source_level": source,
            "target_level": target, "blocks": blocks, "fit": fit, "val": val,
            "source": [b["source"] for b in blocks],
            "target": [b["target"] for b in blocks],
            "claim": "new measurement level; existing specimens/geometries",
            "validation": "source-only whole-file; no target labels for selection"}


def spatial_masks(shape, max_radius=20, cell=90):
    """Geometry-only holdout for unseen physical defect locations.

    Centers in validation cells are separated by a 2*radius buffer from train
    centers. No CAD labels are consulted. Used as a secondary spatial test,
    not an assertion of independent specimens.
    """
    import numpy as np
    rr, cc = np.indices(shape)
    group = ((rr // cell) + 2 * (cc // cell)) % 5
    from scipy.ndimage import binary_dilation
    test = group == 0
    buffer = binary_dilation(test, iterations=2 * max_radius)
    return ~buffer, test
