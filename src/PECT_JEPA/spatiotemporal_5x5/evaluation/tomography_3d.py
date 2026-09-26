"""
3D Defect Tomography & Volumetric Slicing Suite for PECT-JEPA.

Transforms 4-scale spatio-spectral skin-depth representations into physical 3D voxel
grids V(y, x, z) in mm, providing multi-slice B-scan cross-sections, orthogonal depth
projections, and interactive 3D volumetric visualizations.
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Depth layer definitions calibrated from analytical skin-depth dispersion
DEPTH_LAYERS = [
    {"name": "Layer 0 (Near-Surface / Fastener Head)", "z_nominal_mm": 0.25, "z_range_mm": (0.0, 0.5), "scale_idx": 3},
    {"name": "Layer 1 (Subsurface / 1st Layer Notch)", "z_nominal_mm": 0.85, "z_range_mm": (0.5, 1.2), "scale_idx": 2},
    {"name": "Layer 2 (Inter-layer Boundary)",         "z_nominal_mm": 1.60, "z_range_mm": (1.2, 2.0), "scale_idx": 1},
    {"name": "Layer 3 (Deepest / Back-Wall)",          "z_nominal_mm": 2.50, "z_range_mm": (2.0, 3.0), "scale_idx": 0},
]


def plot_3d_ortho_slices(
    volume_3d: np.ndarray,            # [sY, sX, 4]
    mask_2d: Optional[np.ndarray] = None, # [sY, sX] CAD ground truth mask
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    specimen: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plots a multi-panel publication-grade 3D Orthogonal Tomography figure:
      - Top Row: 4 horizontal XY C-scan slices across the 4 physical depth layers.
      - Bottom Row:
          Panel 1: Ground Truth CAD defect mask (if available) or 2D Depth map.
          Panel 2: Horizontal B-scan cross-section (XZ plane along peak flaw Y).
          Panel 3: Vertical B-scan cross-section (YZ plane along peak flaw X).
          Panel 4: Flaw vs Sound Metal Depth Attenuation Profile.
    """
    sY, sX, n_depth = volume_3d.shape
    assert n_depth == 4, f"Expected 4 depth layers, got {n_depth}"

    # Find peak flaw coordinate to slice B-scans through the defect center
    composite_energy = volume_3d[:, :, 1:3].mean(axis=-1)  # Focus on subsurface layers
    if mask_2d is not None and np.sum(mask_2d > 0) > 0:
        defect_idx = np.where(mask_2d > 0)
        peak_y = int(np.median(defect_idx[0]))
        peak_x = int(np.median(defect_idx[1]))
    else:
        peak_pos = np.unravel_index(np.argmax(composite_energy), (sY, sX))
        peak_y, peak_x = int(peak_pos[0]), int(peak_pos[1])

    peak_y = np.clip(peak_y, 0, sY - 1)
    peak_x = np.clip(peak_x, 0, sX - 1)

    fig = plt.figure(figsize=(20, 10), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # 1. Top Row: 4 Depth Layer XY C-Scans
    layer_names = [
        r"$\mathbf{Layer\ 0\ (Surface:\ 0.0-0.5\,mm)}$",
        r"$\mathbf{Layer\ 1\ (Subsurface:\ 0.5-1.2\,mm)}$",
        r"$\mathbf{Layer\ 2\ (Mid-Deep:\ 1.2-2.0\,mm)}$",
        r"$\mathbf{Layer\ 3\ (Back-Wall:\ 2.0-3.0\,mm)}$",
    ]

    for k in range(4):
        ax = fig.add_subplot(2, 4, k + 1)
        slice_k = volume_3d[:, :, k]
        im = ax.imshow(slice_k, cmap="plasma", origin="upper", aspect="equal")
        ax.set_title(layer_names[k], fontsize=11, fontweight="bold")
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        
        # Mark cross-section lines
        ax.axhline(peak_y, color="cyan", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(peak_x, color="lime", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anomaly Energy")

    # 2. Bottom Row, Panel 1: CAD Ground Truth Mask or Maximum Intensity Projection (MIP)
    ax_gt = fig.add_subplot(2, 4, 5)
    if mask_2d is not None and np.any(mask_2d > 0):
        im_gt = ax_gt.imshow(mask_2d, cmap="hot", origin="upper", aspect="equal")
        ax_gt.set_title(r"$\mathbf{CAD\ Ground\ Truth\ Mask}$", fontsize=11, fontweight="bold")
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, label="Defect Label")
    else:
        mip = volume_3d.max(axis=-1)
        im_gt = ax_gt.imshow(mip, cmap="magma", origin="upper", aspect="equal")
        ax_gt.set_title(r"$\mathbf{Max\ Intensity\ Projection\ (MIP)}$", fontsize=11, fontweight="bold")
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, label="Max Energy")
    ax_gt.set_xlabel("X (mm)", fontsize=9)
    ax_gt.set_ylabel("Y (mm)", fontsize=9)

    # 3. Bottom Row, Panel 2: Horizontal B-scan XZ along peak_y
    ax_xz = fig.add_subplot(2, 4, 6)
    bscan_xz = volume_3d[peak_y, :, :].T  # [4, sX]
    z_coords = [0.25, 0.85, 1.60, 2.50]
    im_xz = ax_xz.imshow(
        bscan_xz,
        cmap="plasma",
        origin="upper",
        aspect="auto",
        extent=[0, sX, z_coords[-1], z_coords[0]],
    )
    ax_xz.set_title(rf"$\mathbf{{B-Scan\ XZ\ (at\ Y={peak_y}\,mm)}}$", fontsize=11, fontweight="bold")
    ax_xz.set_xlabel("X (mm)", fontsize=9)
    ax_xz.set_ylabel("Depth Z (mm)", fontsize=9)
    plt.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04, label="Energy")

    # 4. Bottom Row, Panel 3: Vertical B-scan YZ along peak_x
    ax_yz = fig.add_subplot(2, 4, 7)
    bscan_yz = volume_3d[:, peak_x, :].T  # [4, sY]
    im_yz = ax_yz.imshow(
        bscan_yz,
        cmap="plasma",
        origin="upper",
        aspect="auto",
        extent=[0, sY, z_coords[-1], z_coords[0]],
    )
    ax_yz.set_title(rf"$\mathbf{{B-Scan\ YZ\ (at\ X={peak_x}\,mm)}}$", fontsize=11, fontweight="bold")
    ax_yz.set_xlabel("Y (mm)", fontsize=9)
    ax_yz.set_ylabel("Depth Z (mm)", fontsize=9)
    plt.colorbar(im_yz, ax=ax_yz, fraction=0.046, pad=0.04, label="Energy")

    # 5. Bottom Row, Panel 4: Flaw vs Sound Metal Depth Attenuation Curve
    ax_prof = fig.add_subplot(2, 4, 8)
    flaw_prof = volume_3d[peak_y, peak_x, :]
    sound_y = (peak_y + 30) % sY
    sound_x = (peak_x + 30) % sX
    sound_prof = volume_3d[sound_y, sound_x, :]

    ax_prof.plot(z_coords, flaw_prof, "ro-", linewidth=2.0, label="Flaw Core", markersize=6)
    ax_prof.plot(z_coords, sound_prof, "bs--", linewidth=1.5, label="Sound Metal", markersize=5)
    ax_prof.set_title(r"$\mathbf{Depth\ Attenuation\ Profile}$", fontsize=11, fontweight="bold")
    ax_prof.set_xlabel("Physical Depth Z (mm)", fontsize=9)
    ax_prof.set_ylabel("Anomaly Discrepancy", fontsize=9)
    ax_prof.set_ylim(bottom=0.0)
    ax_prof.grid(True, linestyle=":", alpha=0.6)
    ax_prof.legend(loc="best", fontsize=9)

    main_title = title or "3D PECT-JEPA Volumetric Defect Tomography"
    fig.suptitle(f"{main_title}\n[Peak Flaw sliced at X={peak_x} mm, Y={peak_y} mm]", fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return {
        "peak_x": peak_x,
        "peak_y": peak_y,
        "flaw_depth_profile": flaw_prof.tolist(),
        "sound_depth_profile": sound_prof.tolist(),
        "save_path": save_path,
    }


def export_3d_interactive_html(
    volume_3d: np.ndarray,            # [sY, sX, 4]
    save_path: str,
    title: Optional[str] = None,
    threshold_percentile: float = 95.0,
    max_voxels: int = 15000,
) -> Optional[str]:
    """
    Exports an interactive 3D volumetric scatter visualization using Plotly.
    Renders defect voxels in physical (X, Y, Z) space with depth color-coding.
    Allows 360-degree rotation, slicing, and interactive coordinate inspection.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [Notice] Plotly not installed, skipping interactive 3D HTML export.")
        return None

    sY, sX, n_depth = volume_3d.shape
    z_coords = np.array([0.25, 0.85, 1.60, 2.50])

    # Threshold active anomaly voxels
    thresh = np.percentile(volume_3d, threshold_percentile)
    active_y, active_x, active_z_idx = np.where(volume_3d >= thresh)

    if len(active_y) == 0:
        thresh = np.percentile(volume_3d, 90.0)
        active_y, active_x, active_z_idx = np.where(volume_3d >= thresh)

    # Subsample if too many points for browser rendering performance
    if len(active_y) > max_voxels:
        sub_idx = np.random.choice(len(active_y), size=max_voxels, replace=False)
        active_y = active_y[sub_idx]
        active_x = active_x[sub_idx]
        active_z_idx = active_z_idx[sub_idx]

    vals = volume_3d[active_y, active_x, active_z_idx]
    active_z = z_coords[active_z_idx]

    fig = go.Figure(data=[go.Scatter3d(
        x=active_x,
        y=active_y,
        z=active_z,
        mode="markers",
        marker=dict(
            size=3.5,
            color=vals,
            colorscale="Plasma",
            opacity=0.8,
            colorbar=dict(title="Anomaly Energy", thickness=15, len=0.7),
        ),
        text=[f"X: {x}mm<br>Y: {y}mm<br>Depth Z: {z:.2f}mm<br>Energy: {v:.4f}"
              for x, y, z, v in zip(active_x, active_y, active_z, vals)],
        hoverinfo="text",
    )])

    fig.update_layout(
        title=dict(text=title or "3D PECT-JEPA Volumetric Defect Reconstruction", font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="X Scan Width (mm)", range=[0, sX]),
            yaxis=dict(title="Y Scan Length (mm)", range=[0, sY]),
            zaxis=dict(title="Penetration Depth Z (mm)", range=[3.0, 0.0]),  # Invert so 0 is top surface
            aspectratio=dict(x=1.0, y=1.0, z=0.35),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs="cdn")
    return save_path
