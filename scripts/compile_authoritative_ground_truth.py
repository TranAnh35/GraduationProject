"""
Authoritative Ground-Truth Compiler for PECT-JEPA.

Synthesizes:
1. `specimen_mask_features.json` (exact physical CAD features, diameters, depths, volumes)
2. `raw_tdms_rotate_crop.csv` (authoritative per-file rotation, flip, and crop metadata)

Generates:
- Standardized binary ground-truth masks:
    data/ground_truth/corrosion/corrosion_gt_mask.npy [270, 270]
    data/ground_truth/rivet_v1/rivet_v1_gt_mask.npy   [270, 270]
    data/ground_truth/rivet_v2/rivet_v2_gt_mask.npy   [270, 270]
- Complete CAD flaw configuration JSONs:
    data/ground_truth/corrosion/corrosion_flaws_config.json
    data/ground_truth/rivet_v1/rivet_v1_flaws_config.json
    data/ground_truth/rivet_v2/rivet_v2_flaws_config.json
- Publication-ready 3-panel verification overlays (.png):
    data/ground_truth/corrosion/corrosion_verification_overlay.png
    data/ground_truth/rivet_v1/rivet_v1_verification_overlay.png
    data/ground_truth/rivet_v2/rivet_v2_verification_overlay.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.PECT_JEPA.spatiotemporal_5x5.data.ground_truth import get_ground_truth_manager
from src.PECT_JEPA.spatiotemporal_5x5.data.preprocessing import read_tdms_1d_waveforms


SPECIMEN_MAPPINGS = [
    {
        "dir_name": "corrosion",
        "cad_key": "corrosion",
        "display_name": "Corrosion Calibration Grid (5x5)",
        "ref_file": "data/Hall_Pot_Core/Corrosion/Square/corosion_frontside_square_300x300_200Hz_amp2.1_z1_20260105_125851.tdms",
    },
    {
        "dir_name": "rivet_v1",
        "cad_key": "rivet",
        "display_name": "Rivet v1 (Concentric Corrosion & Fasteners)",
        "ref_file": "data/Hall_Pot_Core/Rivet_v1/Square/rivet_frontside_square_300x300_z1_20260119_191806.tdms",
    },
    {
        "dir_name": "rivet_v2",
        "cad_key": "mixed",
        "display_name": "Rivet v2 / Mixed (Offset Corrosion & Fasteners)",
        "ref_file": "data/Hall_Pot_Core/Rivet_v2/Square/mixed_frontside_square_300x300_z1_20260112_125721.tdms",
    },
]


def compile_and_verify(buffer_px: int = 2):
    gt_mgr = get_ground_truth_manager()
    print("=== PECT-JEPA Authoritative Ground-Truth Compiler ===")
    print(f"Loaded CAD specs: {list(gt_mgr.cad_specs.keys())}")
    print(f"Loaded Transformations: {len(gt_mgr.transforms)} files\n")

    for item in SPECIMEN_MAPPINGS:
        dir_name = item["dir_name"]
        cad_key = item["cad_key"]
        disp_name = item["display_name"]
        ref_file = item["ref_file"]

        out_dir = os.path.join(ROOT_DIR, "data", "ground_truth", dir_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"--- Processing Specimen: {dir_name.upper()} ({disp_name}) ---")
        cad_spec = gt_mgr.cad_specs[cad_key]
        features = cad_spec.get("features", [])

        # 1. Generate standardized cropped mask [270, 270]
        mask = gt_mgr.get_standard_cropped_mask(cad_key, buffer_px=buffer_px)
        n_def = int(np.sum(mask == 1))
        n_buf = int(np.sum(mask == -1))
        n_snd = int(np.sum(mask == 0))
        print(f"  Mask Shape: {mask.shape} | Defect Px: {n_def} | Buffer Px: {n_buf} | Sound Px: {n_snd}")

        mask_path = os.path.join(out_dir, f"{dir_name}_gt_mask.npy")
        np.save(mask_path, mask)
        print(f"  [Saved Array] {mask_path}")

        # 2. Export updated flaws configuration JSON
        cfg_path = os.path.join(out_dir, f"{dir_name}_flaws_config.json")
        cropped_features = gt_mgr.get_flaw_features(cad_key, coordinate_system="cropped")
        flaws_config = {
            "specimen": dir_name,
            "title": disp_name,
            "cad_key": cad_key,
            "source_json": "data/ground_truth/specimen_mask_features.json",
            "source_transform_csv": "data/ground_truth/raw_tdms_rotate_crop.csv",
            "volume_formula": cad_spec.get("volumeFormula"),
            "grid_shape": list(mask.shape),
            "crop_border": gt_mgr.default_crop["crop_top"],
            "coordinate_system": "cropped_standardized (x=col, y=row on cropped grid; cad_x, cad_y on nominal 300x300 CAD)",
            "buffer_px": buffer_px,
            "num_total_features": len(cropped_features),
            "num_defects": len([f for f in cropped_features if f.get("diameter") is not None and f.get("kind") != "rivet only"]),
            "reference_file": ref_file,
            "features": cropped_features,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(flaws_config, f, indent=2)
        print(f"  [Saved Config] {cfg_path}")

        # 3. Load reference TDMS and standardize coordinates
        if not os.path.isfile(ref_file):
            print(f"  [Warning] Reference file not found: {ref_file}, skipping plot.")
            continue

        raw_waveforms = read_tdms_1d_waveforms(ref_file, target_time_samples=50, raster_correction=True)
        # Reshape to [300, 300] Peak-to-Peak
        raw_p2p = np.ptp(raw_waveforms, axis=1)[:90000].reshape(300, 300)
        # Transform using authoritative metadata from CSV
        aligned_scan = gt_mgr.transform_cscan_to_standard(raw_p2p, ref_file)
        print(f"  Aligned Scan Shape: {aligned_scan.shape}")

        # 4. Generate 3-Panel Verification Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # Panel 1: Physical C-scan amplitude
        im0 = axes[0].imshow(aligned_scan, cmap="viridis", origin="lower")
        axes[0].set_title(f"1. Standardized Physical C-Scan\n({os.path.basename(ref_file)[:32]}...)", fontsize=11, fontweight="bold")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: Physical C-Scan + Exact CAD ROIs
        im1 = axes[1].imshow(aligned_scan, cmap="viridis", origin="lower")
        axes[1].set_title("2. Physical C-Scan + CAD Overlay\n(Red: Defect ROIs, Cyan: Fasteners)", fontsize=11, fontweight="bold")

        for feat in cropped_features:
            rx = feat.get("x")
            ry = feat.get("y")
            rd = feat.get("rivetDiameter")
            if rx is not None and ry is not None and rd is not None:
                # Fastener rivet ring (already in cropped coordinates)
                axes[1].add_patch(Circle((rx, ry), rd / 2.0, color="cyan", fill=False, linewidth=1.2, linestyle="--"))

            diam = feat.get("diameter")
            if diam is not None and feat.get("kind") != "rivet only":
                cx = feat.get("corrosionX") if feat.get("corrosionX") is not None else rx
                cy = feat.get("corrosionY") if feat.get("corrosionY") is not None else ry
                axes[1].add_patch(Circle((cx, cy), diam / 2.0, color="red", fill=False, linewidth=1.5))
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Binary Ground-Truth Mask
        mask_float = mask.astype(float)
        im2 = axes[2].imshow(mask_float, cmap="coolwarm", origin="lower", vmin=-1.0, vmax=1.0)
        axes[2].set_title(f"3. Exact CAD Ground-Truth Mask\n({n_def} Defect Px, {n_buf} Buffer Px)", fontsize=11, fontweight="bold")
        cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
        cbar2.set_ticklabels(["Buffer (-1)", "Sound (0)", "Defect (1)"])

        plt.suptitle(f"PECT Ground-Truth Verification: {disp_name}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        overlay_path = os.path.join(out_dir, f"{dir_name}_verification_overlay.png")
        plt.savefig(overlay_path, dpi=160)
        plt.close(fig)
        print(f"  [Saved Overlay] {overlay_path}\n")

    print("=== All ground-truth masks, configs, and verification overlays successfully compiled! ===")


if __name__ == "__main__":
    compile_and_verify()
