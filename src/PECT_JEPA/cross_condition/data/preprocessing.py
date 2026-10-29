"""
Data Preprocessing and Canonical Geometric Transformation for Cross-Condition PECT.
Applies authoritative rotation, flipping, and border cropping from raw_tdms_rotate_crop.csv
so that all C-scans [270, 270, C] strictly align with CAD ground-truth masks.
"""

import os
import csv
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np

from src.PECT_JEPA.temporal_1d.data.preprocessing import (
    read_tdms_1d_waveforms,
    apply_lowpass_filter,
)
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import (
    linear_time_resample,
    normalize_waveforms_linear,
)


def compute_sha256_file(file_path: str, chunk_size: int = 65536) -> str:
    """Computes deterministic SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_transform_csv_path(
    csv_path: Optional[str] = None,
    project_root: Optional[str] = None,
) -> Path:
    """
    Resolves the authoritative path to raw_tdms_rotate_crop.csv independently of CWD.
    Searches project roots, env vars, and standard repository structures.
    Raises FileNotFoundError if the file cannot be located.
    """
    attempted = []

    # 1. Explicit path passed
    if csv_path:
        p = Path(csv_path)
        if p.is_absolute() and p.is_file():
            return p.resolve()
        attempted.append(str(p))

        if project_root:
            cand = Path(project_root) / csv_path
            if cand.is_file():
                return cand.resolve()
            attempted.append(str(cand))

    # 2. Environment variable
    env_path = os.environ.get("PECT_TRANSFORM_CSV")
    if env_path:
        ep = Path(env_path)
        if ep.is_file():
            return ep.resolve()
        attempted.append(str(ep))

    # 3. Known repository root relative to this file
    # This file: <repo>/src/PECT_JEPA/cross_condition/data/preprocessing.py
    # Repo root is parents[4]
    try:
        repo_root = Path(__file__).resolve().parents[4]
        cand = repo_root / "data" / "ground_truth" / "raw_tdms_rotate_crop.csv"
        if cand.is_file():
            return cand.resolve()
        attempted.append(str(cand))
    except Exception:
        pass

    # 4. Project root passed or current working directory candidates
    if project_root:
        cand = Path(project_root) / "data" / "ground_truth" / "raw_tdms_rotate_crop.csv"
        if cand.is_file():
            return cand.resolve()
        attempted.append(str(cand))

    cwd_cand = Path.cwd() / "data" / "ground_truth" / "raw_tdms_rotate_crop.csv"
    if cwd_cand.is_file():
        return cwd_cand.resolve()
    attempted.append(str(cwd_cand))

    if csv_path:
        cwd_rel = Path.cwd() / csv_path
        if cwd_rel.is_file():
            return cwd_rel.resolve()
        attempted.append(str(cwd_rel))

    raise FileNotFoundError(
        f"Authoritative canonical transform CSV not found! Attempted paths:\n"
        + "\n".join(f"  - {a}" for a in attempted)
        + "\nPlease set PECT_TRANSFORM_CSV or run within the GraduationProject repository."
    )


class CanonicalCoordinateTransformer:
    """
    Manages authoritative geometric transformations (rotation_deg, flip_lr, flip_ud, crop)
    from raw_tdms_rotate_crop.csv.
    Enforces fail-closed canonical registration: missing CSV or unmatched files raise errors
    unless explicitly configured under legacy unverified mode.
    """
    def __init__(
        self,
        csv_path: Optional[str] = None,
        project_root: Optional[str] = None,
        allow_identity_fallback: bool = False,
    ):
        self.allow_identity_fallback = allow_identity_fallback
        self.resolved_csv_path = resolve_transform_csv_path(csv_path, project_root)
        self.csv_sha256 = compute_sha256_file(str(self.resolved_csv_path))

        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.default_crop = {"crop_top": 15, "crop_bottom": 15, "crop_left": 15, "crop_right": 15}

        with open(self.resolved_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("source", "").strip()
                if not src:
                    continue
                if src == "__global__":
                    self.default_crop["crop_top"] = int(row["crop_top"]) if row.get("crop_top") else 15
                    self.default_crop["crop_bottom"] = int(row["crop_bottom"]) if row.get("crop_bottom") else 15
                    self.default_crop["crop_left"] = int(row["crop_left"]) if row.get("crop_left") else 15
                    self.default_crop["crop_right"] = int(row["crop_right"]) if row.get("crop_right") else 15
                    continue

                rot = int(row["rotation_deg"]) if row.get("rotation_deg") else 0
                flr = int(row["flip_lr"]) if row.get("flip_lr") else 0
                fud = int(row["flip_ud"]) if row.get("flip_ud") else 0
                ct = int(row["crop_top"]) if row.get("crop_top") else self.default_crop["crop_top"]
                cb = int(row["crop_bottom"]) if row.get("crop_bottom") else self.default_crop["crop_bottom"]
                cl = int(row["crop_left"]) if row.get("crop_left") else self.default_crop["crop_left"]
                cr = int(row["crop_right"]) if row.get("crop_right") else self.default_crop["crop_right"]

                self.transforms[src] = {
                    "rotation_deg": rot,
                    "flip_lr": flr,
                    "flip_ud": fud,
                    "crop_top": ct,
                    "crop_bottom": cb,
                    "crop_left": cl,
                    "crop_right": cr,
                }

    def _get_stem_prefix(self, filename: str) -> str:
        parts = filename.replace(".tdms", "").split("_")
        if len(parts) >= 3 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
            return "_".join(parts[:-2])
        return filename.replace(".tdms", "")

    def get_transform(
        self,
        file_path: str,
        allow_fallback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves the exact transform parameters for a TDMS file.
        Matches by:
          1. Relative path substring or exact relative match
          2. Exact basename match
          3. Timestamp match (parts[-2] + '_' + parts[-1])
          4. Stem prefix match (filename without timestamp)
        If unmatched and fallback not permitted, raises ValueError.
        """
        norm_path = file_path.replace("\\", "/")
        base_name = os.path.basename(file_path)

        # 1. Substring match
        for src_key, t in self.transforms.items():
            norm_src = src_key.replace("\\", "/")
            if norm_src in norm_path or norm_path.endswith("/" + norm_src):
                res = dict(t)
                res["match_strategy"] = "exact/substring"
                res["source_key"] = src_key
                return res

        # 2. Basename match
        for src_key, t in self.transforms.items():
            if os.path.basename(src_key) == base_name:
                res = dict(t)
                res["match_strategy"] = "basename"
                res["source_key"] = src_key
                return res

        # 3. Timestamp match
        parts = base_name.replace(".tdms", "").split("_")
        if len(parts) >= 2 and len(parts[-2]) == 8:
            ts = parts[-2] + "_" + parts[-1]
            ts_matches = [(k, v) for k, v in self.transforms.items() if ts in k]
            if len(ts_matches) == 1:
                res = dict(ts_matches[0][1])
                res["match_strategy"] = "timestamp"
                res["source_key"] = ts_matches[0][0]
                return res

        # 4. Stem prefix match (stripping timestamp)
        prefix = self._get_stem_prefix(base_name)
        prefix_matches = [
            (k, v) for k, v in self.transforms.items()
            if self._get_stem_prefix(os.path.basename(k)) == prefix
        ]
        if len(prefix_matches) == 1:
            res = dict(prefix_matches[0][1])
            res["match_strategy"] = "stem_prefix"
            res["source_key"] = prefix_matches[0][0]
            return res

        permit_fb = self.allow_identity_fallback if allow_fallback is None else allow_fallback
        if permit_fb:
            return {
                "rotation_deg": 0,
                "flip_lr": 0,
                "flip_ud": 0,
                **self.default_crop,
                "match_strategy": "identity_fallback_unverified",
                "source_key": "__fallback__",
                "is_fallback": True,
            }

        raise ValueError(
            f"Canonical registration error: No transform mapping found for '{file_path}' "
            f"in authoritative CSV '{self.resolved_csv_path}'. "
            f"Identity fallback is strictly disabled in canonical evaluation mode."
        )

    def transform_cscan(
        self,
        cscan: np.ndarray,
        file_path: str,
        allow_fallback: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Transforms raw [300, 300, C] C-scan into canonical [270, 270, C] coordinate space.
        """
        t = self.get_transform(file_path, allow_fallback=allow_fallback)
        img = cscan.copy()

        # Rotate (k * 90 deg counter-clockwise)
        rot_deg = t["rotation_deg"]
        k = (rot_deg // 90) % 4
        if k > 0:
            img = np.rot90(img, k=k, axes=(0, 1))

        # Horizontal flip
        if t["flip_lr"]:
            img = np.flip(img, axis=1)

        # Vertical flip
        if t["flip_ud"]:
            img = np.flip(img, axis=0)

        # Crop to canonical 270x270
        ct, cb = t["crop_top"], t["crop_bottom"]
        cl, cr = t["crop_left"], t["crop_right"]
        sY, sX = img.shape[:2]

        cropped = img[ct : sY - cb, cl : sX - cr]
        return np.ascontiguousarray(cropped, dtype=np.float32)

    def audit_coverage(self, file_paths: List[str]) -> Dict[str, Any]:
        """Audits registration coverage and effective transforms for a list of TDMS files."""
        records = {}
        all_matched = True
        for fp in file_paths:
            t = self.get_transform(fp, allow_fallback=False)
            records[fp] = {
                "rotation_deg": t["rotation_deg"],
                "flip_lr": t["flip_lr"],
                "flip_ud": t["flip_ud"],
                "crop": [t["crop_top"], t["crop_bottom"], t["crop_left"], t["crop_right"]],
                "match_strategy": t["match_strategy"],
                "source_key": t["source_key"],
            }
        return {
            "csv_path": str(self.resolved_csv_path),
            "csv_sha256": self.csv_sha256,
            "total_files": len(file_paths),
            "matched_files": len(records),
            "records": records,
        }


# Global Transformer instance
_GLOBAL_TRANSFORMER: Optional[CanonicalCoordinateTransformer] = None


def get_canonical_transformer(
    csv_path: Optional[str] = None,
    project_root: Optional[str] = None,
    reset: bool = False,
    allow_identity_fallback: bool = False,
) -> CanonicalCoordinateTransformer:
    global _GLOBAL_TRANSFORMER
    if _GLOBAL_TRANSFORMER is None or reset:
        _GLOBAL_TRANSFORMER = CanonicalCoordinateTransformer(
            csv_path=csv_path,
            project_root=project_root,
            allow_identity_fallback=allow_identity_fallback,
        )
    return _GLOBAL_TRANSFORMER


def load_canonical_cscan_from_tdms(
    file_path: str,
    time_samples: int = 500,
    temporal_samples: int = 128,
    lowpass_cutoff: float = 2500.0,
    normalization: str = "global_peak",
    raster_correction: bool = True,
    apply_transforms: bool = True,
    transformer: Optional[CanonicalCoordinateTransformer] = None,
    transform_csv: Optional[str] = None,
    allow_identity_fallback: bool = False,
) -> np.ndarray:
    """
    Loads TDMS, resamples to continuous temporal samples, applies lowpass,
    normalizes, and applies authoritative geometric rotation/flip/crop from raw_tdms_rotate_crop.csv.
    Returns:
        canonical_cscan: [270, 270, temporal_samples] float32 array strictly aligned with CAD ground truth.
    """
    raw = read_tdms_1d_waveforms(
        file_path=file_path,
        target_time_samples=time_samples,
        normalization="none",
        raster_correction=raster_correction,
    )
    resampled = linear_time_resample(raw, n_out=temporal_samples)
    lowpassed = apply_lowpass_filter(resampled, cutoff_hz=lowpass_cutoff, fs=25600.0, order=4)
    normed = normalize_waveforms_linear(lowpassed, normalization=normalization)

    # Reshape to nominal [300, 300, C]
    grid_300 = normed[:300 * 300].reshape(300, 300, temporal_samples)

    if apply_transforms:
        trans = transformer or get_canonical_transformer(
            csv_path=transform_csv,
            allow_identity_fallback=allow_identity_fallback,
        )
        canonical_grid = trans.transform_cscan(
            grid_300,
            file_path,
            allow_fallback=allow_identity_fallback,
        )
        return canonical_grid
    return grid_300.astype(np.float32)
