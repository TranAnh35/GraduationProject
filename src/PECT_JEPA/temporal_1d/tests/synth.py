"""Synthetic PECT-like waveform generator for unit tests (no TDMS dependency)."""

import numpy as np


def make_waveforms(n: int = 64, T: int = 500, seed: int = 0) -> np.ndarray:
    """
    PECT-like decaying transients: x(t) = A * exp(-t/tau) * sin(2*pi*f*t + phi) + noise.
    Returns [n, T] float32.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, T)
    A = rng.uniform(0.5, 2.0, size=(n, 1))
    tau = rng.uniform(0.1, 0.5, size=(n, 1))
    f = rng.uniform(2.0, 20.0, size=(n, 1))
    phi = rng.uniform(0.0, 2 * np.pi, size=(n, 1))
    x = A * np.exp(-t[None, :] / tau) * np.sin(2 * np.pi * f * t[None, :] + phi)
    x += 0.01 * rng.standard_normal(x.shape)
    return x.astype(np.float32)
