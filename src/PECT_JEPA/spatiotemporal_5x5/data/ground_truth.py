"""
Authoritative Ground-Truth Mask & Transformation Manager for PECT-JEPA.

Integrates:
1. `data/ground_truth/specimen_mask_features.json`:
   Precise physical CAD features (exact defect x,y coordinates in mm,
   defect diameters 3-14 mm, depths 0.1-1.0 mm, rivet diameters 3.95 mm, offsets, volumes).
2. `data/ground_truth/raw_tdms_rotate_crop.csv`:
   Per-file geometric transformation parameters (rotation degrees 0, 90, 180, 270,
   flip_lr, flip_ud, crop boundaries) across all TDMS files.

Provides:
- Standardized CAD ground-truth masks [270, 270] with core defect pixels (1),
  transition buffer zones (-1), and sound metal (0).
- File-specific C-scan geometric standardization (rotate, flip, crop).
- Per-file dynamic mask alignment (both forward-aligned and inverse-aligned modes).
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


SPECIMEN_ALIAS_MAP = {
    "corrosion": "corrosion",
    "corosion": "corrosion",
    "rivet": "rivet",
    "rivet_v1": "rivet",
    "rivet1": "rivet",
    "rivet_v2": "mixed",
    "rivet2": "mixed",
    "mixed": "mixed",
}


class GroundTruthManager:
    """
    Manages CAD defect metadata, file-specific spatial transformations,
    and ground-truth binary masks.
    """

    def __init__(
        self,
        features_json_path: Optional[str] = None,
        rotate_crop_csv_path: Optional[str] = None,
        data_dir: str = "data",
    ):
        self.data_dir = data_dir

        # Auto-discover paths if not explicitly provided
        if features_json_path is None:
            candidates = [
                os.path.join(data_dir, "ground_truth", "specimen_mask_features.json"),
                "data/ground_truth/specimen_mask_features.json",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    features_json_path = c
                    break

        if rotate_crop_csv_path is None:
            candidates = [
                os.path.join(data_dir, "ground_truth", "raw_tdms_rotate_crop.csv"),
                "data/ground_truth/raw_tdms_rotate_crop.csv",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    rotate_crop_csv_path = c
                    break

        self.features_json_path = features_json_path
        self.rotate_crop_csv_path = rotate_crop_csv_path

        self.cad_specs: Dict[str, Dict[str, Any]] = {}
        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.default_crop = {"crop_top": 15, "crop_bottom": 15, "crop_left": 15, "crop_right": 15}

        self._load_cad_features()
        self._load_transforms()

    def _load_cad_features(self):
        if not self.features_json_path or not os.path.isfile(self.features_json_path):
            return
        with open(self.features_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            key = item.get("key", "").lower()
            self.cad_specs[key] = item

    def _load_transforms(self):
        if not self.rotate_crop_csv_path or not os.path.isfile(self.rotate_crop_csv_path):
            return
        df = pd.read_csv(self.rotate_crop_csv_path)
        global_rows = df[df["source"] == "__global__"]
        if len(global_rows) > 0:
            grow = global_rows.iloc[0]
            self.default_crop = {
                "crop_top": int(grow["crop_top"]) if pd.notnull(grow["crop_top"]) else 15,
                "crop_bottom": int(grow["crop_bottom"]) if pd.notnull(grow["crop_bottom"]) else 15,
                "crop_left": int(grow["crop_left"]) if pd.notnull(grow["crop_left"]) else 15,
                "crop_right": int(grow["crop_right"]) if pd.notnull(grow["crop_right"]) else 15,
            }

        for _, row in df.iterrows():
            src = str(row["source"]).strip()
            if src == "__global__" or not src:
                continue
            rot = int(row["rotation_deg"]) if pd.notnull(row["rotation_deg"]) else 0
            flr = int(row["flip_lr"]) if pd.notnull(row["flip_lr"]) else 0
            fud = int(row["flip_ud"]) if pd.notnull(row["flip_ud"]) else 0
            ct = int(row["crop_top"]) if pd.notnull(row["crop_top"]) else self.default_crop["crop_top"]
            cb = int(row["crop_bottom"]) if pd.notnull(row["crop_bottom"]) else self.default_crop["crop_bottom"]
            cl = int(row["crop_left"]) if pd.notnull(row["crop_left"]) else self.default_crop["crop_left"]
            cr = int(row["crop_right"]) if pd.notnull(row["crop_right"]) else self.default_crop["crop_right"]

            self.transforms[src] = {
                "rotation_deg": rot,
                "flip_lr": flr,
                "flip_ud": fud,
                "crop_top": ct,
                "crop_bottom": cb,
                "crop_left": cl,
                "crop_right": cr,
            }

    def canonical_specimen_key(self, name_or_path: str) -> str:
        """
        Maps specimen name or file path to canonical key: 'corrosion', 'rivet', or 'mixed'.
        """
        s = os.path.basename(name_or_path).lower()
        if "corrosion" in s or "corosion" in s:
            return "corrosion"
        if "rivet_v1" in s or "rivet1" in s:
            return "rivet"
        if "rivet_v2" in s or "rivet2" in s or "mixed" in s:
            return "mixed"
        # Check directory parts
        norm = name_or_path.lower().replace("\\", "/")
        if "/corrosion/" in norm:
            return "corrosion"
        if "/rivet_v1/" in norm:
            return "rivet"
        if "/rivet_v2/" in norm or "/mixed/" in norm:
            return "mixed"
        if "/rivet/" in norm:
            return "rivet"
        return s

    def get_transform_for_file(self, file_path: str) -> Dict[str, Any]:
        """
        Finds the exact geometric transformation parameters for any given TDMS file.
        Uses exact relative path match, basename match, timestamp match, or falls back to defaults.
        """
        norm_path = file_path.replace("\\", "/")
        base_name = os.path.basename(file_path)

        # 1. Exact relative path match
        for src_key, trans in self.transforms.items():
            norm_src = src_key.replace("\\", "/")
            if norm_src in norm_path or norm_path.endswith(norm_src):
                return trans

        # 2. Exact basename match
        for src_key, trans in self.transforms.items():
            if os.path.basename(src_key) == base_name:
                return trans

        # 3. Timestamp match (e.g. 20260121_102900)
        parts = base_name.replace(".tdms", "").split("_")
        ts = parts[-2] + "_" + parts[-1] if len(parts) >= 2 and len(parts[-2]) == 8 else ""
        if ts:
            for src_key, trans in self.transforms.items():
                if ts in src_key:
                    return trans

        # 4. Fallback default
        return {
            "rotation_deg": 0,
            "flip_lr": 0,
            "flip_ud": 0,
            **self.default_crop,
        }

    def generate_nominal_cad_mask(
        self,
        specimen: str,
        grid_shape: Tuple[int, int] = (300, 300),
        buffer_px: int = 2,
    ) -> np.ndarray:
        """
        Generates the nominal 2D CAD ground-truth mask at physical resolution (default 300x300 mm).
        Values:
          1: Core defect pixel
         -1: Transition buffer pixel (boundary zone between r and r + buffer_px)
          0: Sound metal
        """
        spec_key = self.canonical_specimen_key(specimen)
        if spec_key not in self.cad_specs:
            raise KeyError(f"Specimen '{spec_key}' not found in CAD features. Available: {list(self.cad_specs.keys())}")

        sY, sX = grid_shape
        mask = np.zeros((sY, sX), dtype=np.int8)

        Y, X = np.ogrid[:sY, :sX]
        features = self.cad_specs[spec_key].get("features", [])

        # First pass: Mark transition buffer zones (-1)
        if buffer_px > 0:
            for feat in features:
                diam = feat.get("diameter")
                if diam is None or feat.get("kind") == "rivet only":
                    continue
                cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
                cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
                if cx is None or cy is None:
                    continue
                r_outer = (diam / 2.0) + buffer_px
                dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
                mask[dist_sq <= (r_outer ** 2)] = -1

        # Second pass: Mark core defect regions (1, overwrites buffer)
        for feat in features:
            diam = feat.get("diameter")
            if diam is None or feat.get("kind") == "rivet only":
                continue
            cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
            cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
            if cx is None or cy is None:
                continue
            r = diam / 2.0
            dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
            mask[dist_sq <= (r ** 2)] = 1

        return mask

    def get_standard_cropped_mask(
        self,
        specimen: str,
        buffer_px: int = 2,
    ) -> np.ndarray:
        """
        Returns the standardized cropped CAD ground-truth mask [270, 270].
        Standard crop removes default 15 px border on all sides.
        """
        nominal = self.generate_nominal_cad_mask(specimen, grid_shape=(300, 300), buffer_px=buffer_px)
        ct = self.default_crop["crop_top"]
        cb = self.default_crop["crop_bottom"]
        cl = self.default_crop["crop_left"]
        cr = self.default_crop["crop_right"]
        return nominal[ct : 300 - cb, cl : 300 - cr].copy()

    def transform_cscan_to_standard(
        self,
        raw_cscan: np.ndarray,  # [300, 300, ...] or [301, 300, ...]
        file_path: str,
    ) -> np.ndarray:
        """
        Transforms a raw C-scan into standard CAD coordinate alignment:
        1. Slices raw scan to [300, 300] if [301, 300].
        2. Applies rotation (rotation_deg: 0, 90, 180, 270 counter-clockwise).
        3. Applies horizontal/vertical flips if indicated in raw_tdms_rotate_crop.csv.
        4. Crops boundary edges [crop_top: 300-crop_bottom, crop_left: 300-crop_right].
        Result is an exact [270, 270, ...] standardized C-scan map.
        """
        trans = self.get_transform_for_file(file_path)
        img = raw_cscan[:300, :300].copy()

        rot_deg = trans["rotation_deg"]
        k = (rot_deg // 90) % 4
        if k > 0:
            img = np.rot90(img, k=k, axes=(0, 1))

        if trans["flip_lr"]:
            # Horizontal flip (along column axis 1)
            img = np.flip(img, axis=1)

        if trans["flip_ud"]:
            # Vertical flip (along row axis 0)
            img = np.flip(img, axis=0)

        ct = trans["crop_top"]
        cb = trans["crop_bottom"]
        cl = trans["crop_left"]
        cr = trans["crop_right"]

        cropped = img[ct : 300 - cb, cl : 300 - cr]
        return np.ascontiguousarray(cropped)

    def get_ground_truth_mask_for_file(
        self,
        file_path: str,
        aligned_scan: bool = True,
        buffer_px: int = 2,
    ) -> np.ndarray:
        """
        Retrieves the ground-truth mask corresponding to a specific TDMS file.
        
        Args:
            file_path: Path to the TDMS file.
            aligned_scan: 
                If True (recommended default), the C-scan has been standardized using
                transform_cscan_to_standard(), so the mask returned is the standard CAD mask [270, 270].
                If False, the C-scan is unaligned/raw, so this method applies the inverse
                transformation to the CAD mask so it aligns directly with the raw C-scan.
            buffer_px: Thickness of boundary transition ignore zone (-1).
        """
        spec_key = self.canonical_specimen_key(file_path)
        if aligned_scan:
            return self.get_standard_cropped_mask(spec_key, buffer_px=buffer_px)

        # Unaligned scan: invert the transformation on the 300x300 nominal mask, then crop
        trans = self.get_transform_for_file(file_path)
        nominal = self.generate_nominal_cad_mask(spec_key, grid_shape=(300, 300), buffer_px=buffer_px)

        # Inverse of: rotate(k), then flip_lr, then flip_ud
        # Inverse order: flip_ud (self-inverse), flip_lr (self-inverse), rotate(-k)
        inv_mask = nominal.copy()
        if trans["flip_ud"]:
            inv_mask = np.flip(inv_mask, axis=0)
        if trans["flip_lr"]:
            inv_mask = np.flip(inv_mask, axis=1)

        rot_deg = trans["rotation_deg"]
        k = (rot_deg // 90) % 4
        if k > 0:
            inv_k = (4 - k) % 4
            inv_mask = np.rot90(inv_mask, k=inv_k, axes=(0, 1))

        ct = trans["crop_top"]
        cb = trans["crop_bottom"]
        cl = trans["crop_left"]
        cr = trans["crop_right"]

        cropped = inv_mask[ct : 300 - cb, cl : 300 - cr]
        return np.ascontiguousarray(cropped)

    def generate_multiclass_mask(
        self,
        specimen: str,
        buffer_px: int = 2,
    ) -> np.ndarray:
        """
        Generates 4-Class Semantic Ground-Truth Mask [270, 270]:
          0: Sound metal
          1: Free corrosion (Corrosion calibration plate)
          2: Sound rivet fastener (pure fastener heads without defect)
          3: Rivet with corrosion (Rivet_v1 concentric and Rivet_v2 offset defects)
         -1: Transition buffer zone
        """
        spec_key = self.canonical_specimen_key(specimen)
        sY, sX = 300, 300
        nominal = np.zeros((sY, sX), dtype=np.int8)
        Y, X = np.ogrid[:sY, :sX]

        features = self.cad_specs.get(spec_key, {}).get("features", [])

        # 1. Mark transition buffers (-1)
        if buffer_px > 0:
            for feat in features:
                diam = feat.get("diameter")
                if diam is not None and feat.get("kind") != "rivet only":
                    cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
                    cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
                    if cx is not None and cy is not None:
                        r_outer = (diam / 2.0) + buffer_px
                        nominal[(X - cx) ** 2 + (Y - cy) ** 2 <= r_outer ** 2] = -1

        # 2. Mark sound rivets (Class 2)
        for feat in features:
            rx = feat.get("x")
            ry = feat.get("y")
            rd = feat.get("rivetDiameter")
            if rx is not None and ry is not None and rd is not None:
                r_rivet = rd / 2.0
                nominal[(X - rx) ** 2 + (Y - ry) ** 2 <= r_rivet ** 2] = 2

        # 3. Mark corrosion defects (Class 1 for free corrosion, Class 3 for rivet+corrosion)
        for feat in features:
            diam = feat.get("diameter")
            if diam is not None and feat.get("kind") != "rivet only":
                cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
                cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
                if cx is not None and cy is not None:
                    r = diam / 2.0
                    target_cls = 1 if spec_key == "corrosion" else 3
                    nominal[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = target_cls

        ct = self.default_crop["crop_top"]
        cb = self.default_crop["crop_bottom"]
        cl = self.default_crop["crop_left"]
        cr = self.default_crop["crop_right"]
        return nominal[ct : 300 - cb, cl : 300 - cr].copy()

    def generate_depth_map(
        self,
        specimen: str,
    ) -> np.ndarray:
        """
        Generates continuous physical defect depth map [270, 270] in millimeters (float32).
        Sound metal has depth = 0.0 mm. Defect pixels have their true CAD depth in [0.1, 1.0] mm.
        """
        spec_key = self.canonical_specimen_key(specimen)
        sY, sX = 300, 300
        depth_nominal = np.zeros((sY, sX), dtype=np.float32)
        Y, X = np.ogrid[:sY, :sX]

        features = self.cad_specs.get(spec_key, {}).get("features", [])
        for feat in features:
            diam = feat.get("diameter")
            dp = feat.get("depth")
            if diam is not None and dp is not None and feat.get("kind") != "rivet only":
                cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
                cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
                if cx is not None and cy is not None:
                    r = diam / 2.0
                    depth_nominal[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = float(dp)

        ct = self.default_crop["crop_top"]
        cb = self.default_crop["crop_bottom"]
        cl = self.default_crop["crop_left"]
        cr = self.default_crop["crop_right"]
        return depth_nominal[ct : 300 - cb, cl : 300 - cr].copy()

    def generate_severity_mask(
        self,
        specimen: str,
        buffer_px: int = 2,
    ) -> np.ndarray:
        """
        Generates defect severity / depth binning mask [270, 270]:
          0: Sound metal (depth = 0.0)
          1: Shallow defect (0 < depth <= 0.2 mm)
          2: Medium defect (0.2 < depth <= 0.6 mm)
          3: Severe defect (depth > 0.6 mm)
         -1: Transition buffer zone
        """
        spec_key = self.canonical_specimen_key(specimen)
        sY, sX = 300, 300
        sev_nominal = np.zeros((sY, sX), dtype=np.int8)
        Y, X = np.ogrid[:sY, :sX]

        features = self.cad_specs.get(spec_key, {}).get("features", [])
        if buffer_px > 0:
            for feat in features:
                diam = feat.get("diameter")
                if diam is not None and feat.get("kind") != "rivet only":
                    cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
                    cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
                    if cx is not None and cy is not None:
                        r_outer = (diam / 2.0) + buffer_px
                        sev_nominal[(X - cx) ** 2 + (Y - cy) ** 2 <= r_outer ** 2] = -1

        for feat in features:
            diam = feat.get("diameter")
            dp = feat.get("depth")
            if diam is not None and dp is not None and feat.get("kind") != "rivet only":
                cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else feat.get("x")
                cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else feat.get("y")
                if cx is not None and cy is not None:
                    r = diam / 2.0
                    val = float(dp)
                    if val <= 0.25:
                        s_cls = 1
                    elif val <= 0.65:
                        s_cls = 2
                    else:
                        s_cls = 3
                    sev_nominal[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = s_cls

        ct = self.default_crop["crop_top"]
        cb = self.default_crop["crop_bottom"]
        cl = self.default_crop["crop_left"]
        cr = self.default_crop["crop_right"]
        return sev_nominal[ct : 300 - cb, cl : 300 - cr].copy()



    def get_flaw_features(
        self,
        specimen: str,
        coordinate_system: str = "cropped",
    ) -> List[Dict[str, Any]]:
        """
        Returns standardized defect flaw features for a specimen.
        
        Args:
            specimen: 'corrosion', 'rivet_v1', 'rivet_v2' (or file path/stem)
            coordinate_system:
                - 'cropped': (default) Coordinates `x, y`, `corrosionX, corrosionY` are shifted by
                  `-crop_left, -crop_top` to match the standardized [270, 270] C-scan arrays.
                  Original CAD coordinates are preserved in `cad_x, cad_y`, `cad_corrosionX, cad_corrosionY`.
                - 'cad': Coordinates `x, y` match the original nominal [300, 300] plate blueprint.
        """
        spec_key = self.canonical_specimen_key(specimen)
        raw_features = self.cad_specs.get(spec_key, {}).get("features", [])
        cl = self.default_crop["crop_left"]
        ct = self.default_crop["crop_top"]
        
        out_features = []
        for feat in raw_features:
            f = dict(feat)
            cad_x = f.get("x")
            cad_y = f.get("y")
            f["cad_x"] = cad_x
            f["cad_y"] = cad_y
            
            cad_cx = f.get("corrosionX")
            cad_cy = f.get("corrosionY")
            if cad_cx is not None:
                f["cad_corrosionX"] = cad_cx
            if cad_cy is not None:
                f["cad_corrosionY"] = cad_cy
                
            if coordinate_system.lower() == "cropped":
                if cad_x is not None:
                    f["x"] = round(float(cad_x - cl), 3)
                if cad_y is not None:
                    f["y"] = round(float(cad_y - ct), 3)
                if cad_cx is not None:
                    f["corrosionX"] = round(float(cad_cx - cl), 3)
                if cad_cy is not None:
                    f["corrosionY"] = round(float(cad_cy - ct), 3)
                    
            out_features.append(f)
        return out_features


# Global singleton instance for easy import across modules
_GT_MANAGER: Optional[GroundTruthManager] = None


def get_ground_truth_manager(data_dir: str = "data") -> GroundTruthManager:
    global _GT_MANAGER
    if _GT_MANAGER is None:
        _GT_MANAGER = GroundTruthManager(data_dir=data_dir)
    return _GT_MANAGER
