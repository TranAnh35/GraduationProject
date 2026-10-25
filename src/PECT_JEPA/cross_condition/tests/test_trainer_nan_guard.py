"""
Trainer guard: a non-finite loss must skip the step, not poison the weights.

Why this test exists: on Colab, one bad batch (AMP overflow) produced NaN
latents; the crash surfaced much later inside `compute_svd_diagnostics`
(`linalg.eigh ... error code: 62`) while the weights had already been destroyed.
The guard skips the optimizer step for that batch, keeps the epoch averages
finite, and logs the skip count into run.log.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig  # noqa: E402
from src.PECT_JEPA.cross_condition.training.trainer import CrossConditionTrainer  # noqa: E402


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class _FakeModel(nn.Module):
    """Returns a NaN loss when the batch marker is 1.0, else a finite loss."""

    def __init__(self, embed_dim=8, n_ctx=4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.embed_dim = embed_dim
        self.n_ctx = n_ctx
        self.calls = 0

    def update_target_encoder(self, momentum: float = 0.99):
        return None

    def forward(self, x_a, x_b, ref_b, ref_a=None, scan_id=None,
                scan_id_b=None, inv_scale: float = 1.0):
        self.calls += 1
        B = x_a.shape[0]
        nan_batch = bool(float(x_a[0, 0, 0, 0]) > 0.5)
        value = float("nan") if nan_batch else 0.5
        loss = torch.as_tensor(value, dtype=torch.float32) + 0.0 * self.weight.sum()
        reps = torch.randn(B, self.n_ctx, self.embed_dim)
        if nan_batch:
            reps = reps * float("nan")
        scalar = lambda v: torch.as_tensor(v, dtype=torch.float32)  # noqa: E731
        return {
            "total_loss": loss,
            "pred_loss": scalar(0.4),
            "aux_loss": scalar(0.1),
            "var_loss": scalar(0.1),
            "cov_loss": scalar(0.0),
            "inv_loss": scalar(0.0),
            "inv_keep_frac": scalar(1.0),
            "H_ctx_a": reps,
        }


def _batch(marker: float):
    return {
        "x_a": torch.full((2, 5, 5, 8), marker),
        "x_b": torch.zeros(2, 5, 5, 8),
        "ref_b": torch.zeros(2, 5, 5, 8),
        "scan_id": torch.tensor([0, 1]),
        "scan_id_b": torch.tensor([2, 3]),
    }


class TestNonFiniteLossGuard(unittest.TestCase):
    def _make_trainer(self):
        cfg = CrossConditionConfig(mixed_precision=False, epochs=2, lr=1e-3)
        logger = _FakeLogger()
        trainer = CrossConditionTrainer(cfg, torch.device("cpu"), logger=logger)
        trainer.model = _FakeModel()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        return trainer, logger

    def test_nan_batch_is_skipped_and_weights_stay_finite(self):
        trainer, logger = self._make_trainer()
        loader = [_batch(0.0), _batch(1.0), _batch(0.0)]  # 1 NaN batch of 3

        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=10)

        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])),
                        "epoch average picked up the NaN batch")
        self.assertAlmostEqual(metrics["total_loss"], 0.5, places=5)
        self.assertTrue(torch.isfinite(trainer.model.weight).all(),
                        "NaN loss reached backward() and poisoned the weights")
        self.assertTrue(any("skipped 1/3 batches with non-finite loss" in m
                            for m in logger.messages), logger.messages)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["effective_rank"])))

    def test_all_finite_batches_take_the_normal_path(self):
        trainer, logger = self._make_trainer()
        metrics = trainer.train_epoch([_batch(0.0)] * 3, epoch=1, total_epochs=10)

        self.assertAlmostEqual(metrics["total_loss"], 0.5, places=5)
        self.assertFalse(any("non-finite" in m for m in logger.messages), logger.messages)
        self.assertGreater(metrics["effective_rank"], 1.0)

    def test_nan_latents_do_not_break_the_spectrum(self):
        trainer, logger = self._make_trainer()
        # First 2 batches are sampled for R_eff, and batch #2 is the NaN one.
        metrics = trainer.train_epoch([_batch(0.0), _batch(1.0), _batch(0.0)],
                                      epoch=1, total_epochs=10)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["effective_rank"])))
        self.assertTrue(any("non-finite and excluded from the R_eff" in m
                            for m in logger.messages), logger.messages)


if __name__ == "__main__":
    unittest.main()
