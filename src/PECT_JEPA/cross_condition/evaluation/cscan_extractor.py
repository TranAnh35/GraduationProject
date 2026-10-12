"""
Fast Vectorized C-Scan Feature Extractor for Cross-Condition PECT-JEPA.
Converts any [270, 270, 128] C-scan into a dense latent representation tensor [270, 270, 2*D]
in sub-second GPU time using coordinate broadcasting and batched forward passes.
"""

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from ..data.topologies import ConcentricStarTopology
from ..models.cross_condition_jepa import CrossConditionJEPA


class CScanFeatureExtractor:
    """
    Sub-second GPU feature extractor for full 270x270 C-scans.
    """
    def __init__(
        self,
        model: CrossConditionJEPA,
        device: torch.device,
        batch_size: int = 4096,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.model.eval()

        self.topology = ConcentricStarTopology(star_radii=(1, 3, 7))
        self.offsets = self.topology.offsets  # [25, 2] in grid units (-7..+7)

    @torch.no_grad()
    def extract_full_cscan(self, cscan: np.ndarray, self_calibrate: bool = False) -> np.ndarray:
        """
        Extracts unified downstream latent representation for entire C-scan [270, 270, C].
        Args:
            cscan: [270, 270, 128] float32 numpy array.
            self_calibrate: If True, performs unsupervised spatial self-calibration (zero-centering
                            against C-scan median sound-metal background) to align inter-scan baselines.
        Returns:
            latents: [270, 270, 2*D] float32 numpy array.
        """
        H, W, C = cscan.shape
        pad = 7
        padded = np.pad(cscan, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

        # Meshgrid of all pixel centers
        rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        rr_flat = rr.flatten() + pad
        cc_flat = cc.flatten() + pad
        N_pts = len(rr_flat)

        embed_dim = self.model.config.embed_dim * 2
        out_features = np.zeros((N_pts, embed_dim), dtype=np.float32)

        offsets_r = self.offsets[:, 0]  # [25]
        offsets_c = self.offsets[:, 1]  # [25]

        # Process in batches
        for start_idx in range(0, N_pts, self.batch_size):
            end_idx = min(start_idx + self.batch_size, N_pts)
            b_r = rr_flat[start_idx:end_idx]
            b_c = cc_flat[start_idx:end_idx]

            # Vectorized coordinate broadcasting: [B, 25]
            r_coords = b_r[:, None] + offsets_r[None, :]
            c_coords = b_c[:, None] + offsets_c[None, :]

            # Fetch patches: [B, 25, C] -> reshape to 5x5 concentric grid
            # Since ConcentricStarTopology maps 25 points, we place them into [B, 5, 5, C]
            # using standard indexing
            grid_5x5 = self.topology.grid_indices  # [5, 5] maps to 0..24
            patch_25 = padded[r_coords, c_coords]  # [B, 25, C]
            patch_5x5 = patch_25[:, grid_5x5]      # [B, 5, 5, C]

            x_tensor = torch.from_numpy(patch_5x5).to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp and self.device.type == "cuda"):
                feats = self.model.extract_features(x_tensor)  # [B, 2*D]

            out_features[start_idx:end_idx] = feats.cpu().float().numpy()

        if self_calibrate:
            # Unsupervised spatial self-calibration: zero-center against plate median sound metal (>98% pixels)
            bg_median = np.median(out_features, axis=0, keepdims=True)
            out_features = out_features - bg_median

        return out_features.reshape(H, W, embed_dim)
