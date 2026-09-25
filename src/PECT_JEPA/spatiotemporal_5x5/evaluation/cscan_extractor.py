"""
Full C-Scan Feature Extractor for 5x5 PECT-JEPA.
Slides a 5x5 spatial window over any C-scan (e.g. 300x300),
extracting the latent representation at the center point (2, 2)
to produce an exact 1-to-1 spatial resolution feature map [300, 300, D].
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from ..models.jepa_5x5 import PECT_JEPA_5x5
from ..data.preprocessing import (
    read_tdms_1d_waveforms,
    build_two_channel_input,
    linear_time_resample,
    normalize_waveforms_linear,
    apply_lowpass_filter,
)


from ..data.topologies import get_spatial_topology_offsets


@torch.no_grad()
def extract_full_cscan_map(
    model: PECT_JEPA_5x5,
    full_cscan_3d: np.ndarray,  # [sY, sX, C]
    batch_size: int = 2048,
    device: str = "cuda",
    show_pbar: bool = False
) -> np.ndarray:
    """
    Extract exact [sY, sX, D] feature map from a 3D C-scan grid [sY, sX, C].
    Uses high-speed vectorized spatial topology indexing (20x faster than single-slice loops).
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    sY, sX, C = full_cscan_3d.shape
    topology = getattr(model.config, "spatial_topology", "concentric_star")
    star_radii = getattr(model.config, "star_radii", (1, 3, 7))
    grid_size = getattr(model.config, "grid_size", 5)

    offsets = get_spatial_topology_offsets(topology=topology, star_radii=star_radii, grid_size=grid_size)
    max_off = int(np.max(np.abs(offsets)))
    pad = max(grid_size // 2, max_off)

    padded = np.pad(full_cscan_3d, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out_map = np.zeros((sY, sX, model.config.embed_dim), dtype=np.float32)

    all_r, all_c = np.meshgrid(np.arange(sY), np.arange(sX), indexing="ij")
    all_r = all_r.reshape(-1)
    all_c = all_c.reshape(-1)
    total_pts = sY * sX

    iterator = range(0, total_pts, batch_size)
    if show_pbar:
        iterator = tqdm(iterator, desc="[Vectorized 1-to-1 C-Scan Extraction]", dynamic_ncols=True)

    with torch.inference_mode():
        for k in iterator:
            k_end = min(k + batch_size, total_pts)
            rows_b = all_r[k:k_end] + pad
            cols_b = all_c[k:k_end] + pad

            sample_r = rows_b[:, None] + offsets[None, :, 0]
            sample_c = cols_b[:, None] + offsets[None, :, 1]

            patch_b = padded[sample_r, sample_c, :].reshape(-1, grid_size, grid_size, C)
            x_b = torch.from_numpy(patch_b).float().to(dev)

            z_center = model.extract_center_feature(x_b).cpu().numpy()
            out_map[all_r[k:k_end], all_c[k:k_end]] = z_center

    return out_map


def load_cscan_from_tdms(
    file_path: str,
    time_samples: int = 500,
    temporal_samples: int = 128,
    resample_mode: str = "linear",
    normalization: str = "global_peak",
    raster_correction: bool = True,
    sX: int = 300,
    sY: int = 300,
    crop_border: int = 10,
    apply_lowpass: bool = True,
    lowpass_cutoff: float = 2500.0,
    lowpass_order: int = 4,
    standardize_coords: bool = True,
) -> np.ndarray:
    """
    Reads a TDMS file and converts it into a [sY, sX, C] grid (C=128 for linear, C=256 for dual_channel).
    If standardize_coords is True (default), applies authoritative geometric transformations
    (rotation, flip, crop) from raw_tdms_rotate_crop.csv to align directly with CAD ground-truth masks.
    Otherwise, optionally crops outer boundary pixels (crop_border on each side).
    """
    raw = read_tdms_1d_waveforms(
        file_path=file_path,
        target_time_samples=time_samples,
        normalization="none",
        raster_correction=raster_correction
    )
    if resample_mode == "linear":
        x_resampled = linear_time_resample(raw, n_out=temporal_samples)
        if apply_lowpass:
            x_resampled = apply_lowpass_filter(
                x_resampled, cutoff_hz=lowpass_cutoff, fs=25600.0, order=lowpass_order
            )
        flat_c = normalize_waveforms_linear(x_resampled, normalization=normalization)
    else:
        two_ch = build_two_channel_input(
            raw,
            log_time_samples=temporal_samples,
            normalization=normalization
        )  # [N, 2, 128]
        N, channels, T_prime = two_ch.shape
        flat_c = two_ch.reshape(N, channels * T_prime)  # [N, 256]

    C = flat_c.shape[-1]
    grid = flat_c[:sY * sX].reshape(sY, sX, C)

    if standardize_coords:
        try:
            from ..data.ground_truth import get_ground_truth_manager
            gt_mgr = get_ground_truth_manager()
            grid = gt_mgr.transform_cscan_to_standard(grid, file_path)
        except Exception as e:
            if crop_border > 0:
                cb = crop_border
                grid = grid[cb: sY - cb, cb: sX - cb, :]
    elif crop_border > 0:
        cb = crop_border
        grid = grid[cb: sY - cb, cb: sX - cb, :]

    return grid.astype(np.float32)

