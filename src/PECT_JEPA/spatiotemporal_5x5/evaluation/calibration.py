"""
Unsupervised and Few-Shot Calibration Module for PECT-JEPA (Phase 1 Pre-registration).

Implements Adaptation Modes:
- M0: Zero-shot uncalibrated transfer (identity/frozen readouts).
- M1: Unsupervised Scan-Level Self-Calibration (NDT sound metal plate baseline).
      Automatically selects sound metal region (>95% of C-scan) without labels,
      extracts reference waveform x_sound(t), and zero-centers representations.
- M2: Few-Shot Labeled Calibration with K in {1, 3, 5} known flaw objects.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def extract_scan_sound_reference(
    cscan_3d: np.ndarray,
    energy_percentile_low: float = 25.0,
    energy_percentile_high: float = 75.0,
    crop_border_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unsupervised extraction of the sound-metal reference waveform x_sound(t) from a C-scan.
    
    Physical Rationale:
      In PECT inspection, sound metal plate constitutes >95% of scanned area,
      while defects occupy <2.5%. The interquartile energy range [Q_25, Q_75]
      excludes air/edge boundaries and flaw anomalies without requiring CAD labels.

    Args:
        cscan_3d: [H, W, C] 3D C-scan array.
        energy_percentile_low: Lower percentile of spatial energy (default 25.0).
        energy_percentile_high: Upper percentile of spatial energy (default 75.0).
        crop_border_ratio: Fraction of outer border to exclude (avoids fixture edges).

    Returns:
        x_sound: [C] 1D sound-metal reference waveform.
        sound_mask: [H, W] boolean mask where True indicates verified sound metal.
        stats: Dictionary containing diagnostic statistics.
    """
    H, W, C = cscan_3d.shape

    # 1. Compute pixel-wise transient RMS energy
    energy_map = np.sqrt(np.mean(cscan_3d ** 2, axis=-1))  # [H, W]

    # 2. Exclude outermost scanner boundary
    border_y = int(H * crop_border_ratio)
    border_x = int(W * crop_border_ratio)
    interior_mask = np.zeros((H, W), dtype=bool)
    interior_mask[border_y:H - border_y, border_x:W - border_x] = True

    interior_energy = energy_map[interior_mask]
    q_low = np.percentile(interior_energy, energy_percentile_low)
    q_high = np.percentile(interior_energy, energy_percentile_high)

    # 3. Sound-metal mask: interior pixels with homogeneous median energy
    sound_mask = interior_mask & (energy_map >= q_low) & (energy_map <= q_high)

    assert np.sum(sound_mask) > 100, "Insufficient sound metal pixels detected for calibration!"

    # 4. Extract robust spatial median waveform across sound metal
    sound_waveforms = cscan_3d[sound_mask]  # [N_sound, C]
    x_sound = np.median(sound_waveforms, axis=0)  # [C]
    sigma_sound = np.std(sound_waveforms, axis=0)  # [C]

    stats = {
        "n_sound_pixels": int(np.sum(sound_mask)),
        "sound_pixel_ratio": float(np.mean(sound_mask)),
        "median_energy": float(np.median(interior_energy)),
        "q25_energy": float(q_low),
        "q75_energy": float(q_high),
        "mean_sound_waveform_std": float(np.mean(sigma_sound)),
    }

    return x_sound.astype(np.float32), sound_mask, stats


def apply_m1_unsupervised_calibration(
    fmap: np.ndarray,
    sound_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Mode M1: Latent-space Unsupervised Self-Calibration.
    Zero-centers the feature map [H, W, D] with respect to verified sound metal.
    Eliminates DC sensor transfer function drift across disparate inspection scans.
    """
    H, W, D = fmap.shape
    if sound_mask is None:
        # Fallback to global spatial median across whole scan (>95% sound metal)
        z_sound = np.median(fmap.reshape(-1, D), axis=0, keepdims=True)
    else:
        assert sound_mask.shape == (H, W), f"Shape mismatch: {sound_mask.shape} vs {(H, W)}"
        z_sound = np.median(fmap[sound_mask], axis=0, keepdims=True)

    return fmap - z_sound.reshape(1, 1, D)


def compute_m1_mahalanobis_score(
    fmap: np.ndarray,
    sound_mask: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Mode M1 Unsupervised Mahalanobis Anomaly Detection.
    Computes coordinate-wise standardized Mahalanobis distance strictly calibrated
    on the sound-metal reference distribution.

    Returns:
        score_map: [H, W] float32 anomaly map.
    """
    H, W, D = fmap.shape
    flat = fmap.reshape(-1, D)

    if sound_mask is None:
        med = np.median(flat, axis=0, keepdims=True)
        mad = np.median(np.abs(flat - med), axis=0, keepdims=True)
    else:
        sound_pts = fmap[sound_mask]
        med = np.median(sound_pts, axis=0, keepdims=True)
        mad = np.median(np.abs(sound_pts - med), axis=0, keepdims=True)

    mad = np.maximum(mad, eps)
    std_z = (flat - med) / mad
    score = np.linalg.norm(std_z, axis=-1)
    return score.reshape(H, W).astype(np.float32)


class FewShotM2Calibrator:
    """
    Mode M2: Few-Shot Labeled Calibration with K in {1, 3, 5} known flaw objects.
    Trains a lightweight linear readout using K flaw objects and background sound metal,
    evaluated across all remaining unseen flaw objects on the target condition.
    """
    def __init__(self, k_flaws: int = 1, seed: int = 42):
        self.k_flaws = int(k_flaws)
        self.seed = seed
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, random_state=seed)

    def fit(self, X_flaws_k: np.ndarray, X_sound: np.ndarray):
        """
        Fit on K flaw object representations and sampled sound metal.
        """
        n_flaw = len(X_flaws_k)
        n_sound = min(len(X_sound), n_flaw * 10)  # Balanced background sample
        rng = np.random.RandomState(self.seed)
        sound_sample_idx = rng.choice(len(X_sound), size=n_sound, replace=False)
        X_train = np.vstack([X_flaws_k, X_sound[sound_sample_idx]])
        y_train = np.hstack([np.ones(n_flaw, dtype=int), np.zeros(n_sound, dtype=int)])

        X_train_s = self.scaler.fit_transform(X_train)
        self.clf.fit(X_train_s, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X_test_s = self.scaler.transform(X_test)
        return self.clf.predict_proba(X_test_s)[:, 1]
