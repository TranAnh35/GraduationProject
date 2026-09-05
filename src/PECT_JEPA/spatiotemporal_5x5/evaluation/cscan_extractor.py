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
)


@torch.no_grad()
def extract_full_cscan_map(
    model: PECT_JEPA_5x5,
    full_cscan_3d: np.ndarray,  # [sY, sX, C]
    batch_size: int = 512,
    device: str = "cuda",
    show_pbar: bool = False
) -> np.ndarray:
    """
    Extract exact [sY, sX, D] feature map from a 3D C-scan grid [sY, sX, C].
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()

    sY, sX, C = full_cscan_3d.shape
    pad = model.config.grid_size // 2  # 2 for 5x5
    padded = np.pad(full_cscan_3d, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    out_map = np.zeros((sY, sX, model.config.embed_dim), dtype=np.float32)

    patches = []
    coords = []
    total_pts = sY * sX

    iterator = range(sY)
    if show_pbar:
        iterator = tqdm(iterator, desc="[Extracting 1-to-1 C-Scan Features]", dynamic_ncols=True)

    for i in iterator:
        for j in range(sX):
            sub = padded[i:i + model.config.grid_size, j:j + model.config.grid_size, :]
            patches.append(sub)
            coords.append((i, j))

            if len(patches) >= batch_size:
                x_b = torch.from_numpy(np.stack(patches, axis=0)).float().to(dev)
                z_center = model.extract_center_feature(x_b).cpu().numpy()
                for (ci, cj), z in zip(coords, z_center):
                    out_map[ci, cj] = z
                patches, coords = [], []

    if patches:
        x_b = torch.from_numpy(np.stack(patches, axis=0)).float().to(dev)
        z_center = model.extract_center_feature(x_b).cpu().numpy()
        for (ci, cj), z in zip(coords, z_center):
            out_map[ci, cj] = z

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
) -> np.ndarray:
    """
    Reads a TDMS file and converts it into a [sY, sX, C] grid (C=128 for linear, C=256 for dual_channel).
    Optionally crops outer boundary pixels (crop_border on each side) to remove air/edge effect.
    """
    raw = read_tdms_1d_waveforms(
        file_path=file_path,
        target_time_samples=time_samples,
        normalization="none",
        raster_correction=raster_correction
    )
    if resample_mode == "linear":
        x_resampled = linear_time_resample(raw, n_out=temporal_samples)
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
    if crop_border > 0:
        cb = crop_border
        grid = grid[cb: sY - cb, cb: sX - cb, :]
    return grid.astype(np.float32)
