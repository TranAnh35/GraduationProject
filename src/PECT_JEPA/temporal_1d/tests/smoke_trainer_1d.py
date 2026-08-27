"""
End-to-end smoke test for 1D Temporal PECT-JEPA: synthetic multi-file dataset ->
JEPATrainer1D (2 epochs, CPU) -> probe matrix + effective rank.

Run (project root):
    python -m src.PECT_JEPA.temporal_1d.tests.smoke_trainer_1d
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..configs.config import Temporal1DConfig
from ..data.dataset import PECT1DDataset, FileBalancedBatchSampler, collate_1d_batch
from ..models.jepa_1d import PECT_JEPA_1D
from ..training.trainer import JEPATrainer1D
from ..probing.probe_matrix import extract_pooled_features, run_probe_matrix, effective_rank
from .synth import make_waveforms


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 4 synthetic "files" with distinct decay constants; sensor labels random
    # (NOT encoded in the waveform -> probe should be near chance).
    n_files, n_pts, T, n_te = 4, 40, 500, 10
    raw_tr, raw_te, sensor_tr, sensor_te, file_tr, file_te = [], [], [], [], [], []
    for f in range(n_files):
        x = make_waveforms(n=n_pts, T=T, seed=f)
        x *= rng.uniform(0.8, 1.2)  # per-file gain nuisance
        raw_tr.append(x[:-n_te]); raw_te.append(x[-n_te:])
        sensor_tr += [f % 3] * (n_pts - n_te); sensor_te += [f % 3] * n_te
        file_tr += [f] * (n_pts - n_te); file_te += [f] * n_te
    raw_tr = np.concatenate(raw_tr); raw_te = np.concatenate(raw_te)

    cfg = Temporal1DConfig(
        log_time_samples=128, num_patches=16, embed_dim=64,
        encoder_depth=2, predictor_depth=1, batch_size=32, k_per_file=8,
        epochs=2, warmup_epochs=0, device="cpu", mixed_precision=False,
        save_dir="checkpoints/smoke_1d",
    )

    train_set = PECT1DDataset(arrays=raw_tr, use_memmap=False)
    val_set = PECT1DDataset(arrays=raw_te, use_memmap=False)

    sampler = FileBalancedBatchSampler(
        train_set.file_point_counts, batch_size=cfg.batch_size,
        k_per_file=cfg.k_per_file, seed=0
    )
    train_loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate_1d_batch)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_1d_batch)

    model = PECT_JEPA_1D(cfg)
    trainer = JEPATrainer1D(model, train_loader, val_loader, cfg)
    history = trainer.train()

    assert history["train_loss"][1] < history["train_loss"][0], "loss must decrease"
    val_metrics = trainer.evaluate()
    print(f"Val metrics: {val_metrics}")

    feats = extract_pooled_features(model, val_loader, torch.device("cpu"))
    n_val = len(val_set)
    labels = {"sensor": np.array(sensor_te), "file_identity": np.array(file_te)}
    half = n_val // 2
    tr, te = np.zeros(n_val, bool), np.zeros(n_val, bool)
    tr[:half], te[half:] = True, True
    probes = run_probe_matrix(feats, labels, tr, te)
    print(f"Probe matrix: {probes}")
    print(f"Effective rank (val features): {effective_rank(feats):.2f}")

    # Smoke-level contract: probes run and return valid accuracies in [0, 1].
    for v in probes.values():
        assert np.isfinite(v) and 0.0 <= v <= 1.0
    print("[PASS] smoke_trainer_1d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
