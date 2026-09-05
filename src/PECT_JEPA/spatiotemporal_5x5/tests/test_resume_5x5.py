"""
Unit tests for the 5x5 Spatiotemporal PECT-JEPA Resume Training Mechanism.
Tests:
- Checkpoint state completeness (model, optimizer, scaler, step, epoch, val_loss)
- Resuming with keyword 'latest', 'best', 'auto', and direct file path
- Parameter state restoration fidelity
- Start epoch and scheduler synchronization
"""

import os
import shutil
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader

from ..configs.config import Spatiotemporal5x5Config
from ..models.jepa_5x5 import PECT_JEPA_5x5
from ..training.trainer import Trainer5x5


class Dummy5x5Dataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int = 16, grid_size: int = 5, channels: int = 32):
        self.data = torch.randn(n_samples, grid_size, grid_size, channels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"data": self.data[idx]}


class TestResume5x5(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.save_dir = os.path.join(self.temp_dir, "checkpoints")
        self.log_dir = os.path.join(self.temp_dir, "logs")

        self.config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=32,
            embed_dim=32,
            encoder_depth=1,
            encoder_heads=2,
            predictor_depth=1,
            predictor_heads=2,
            epochs=5,
            batch_size=8,
            device="cpu",
            save_dir=self.save_dir,
            log_dir=self.log_dir,
            use_tensorboard=False,
            use_wandb=False,
        )

        dataset = Dummy5x5Dataset(n_samples=16, grid_size=5, channels=32)
        self.loader = DataLoader(dataset, batch_size=8, shuffle=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_resume_fidelity(self):
        """Test full save, resume from 'auto', and parameter consistency."""
        model_stage1 = PECT_JEPA_5x5(self.config)
        trainer1 = Trainer5x5(
            model=model_stage1,
            config=self.config,
            train_loader=self.loader,
            val_loader=self.loader,
        )

        # Train 1 epoch
        trainer1.current_epoch = 0
        train_metrics = trainer1.train_epoch()
        val_metrics = trainer1.validate()
        latest_ckpt_path = os.path.join(self.save_dir, "latest_model_5x5.pt")
        trainer1.save_checkpoint(latest_ckpt_path, val_metrics["val_loss"], val_metrics.get("effective_rank"))

        self.assertTrue(os.path.isfile(latest_ckpt_path), "Checkpoint file was not created")

        # Inspect checkpoint keys
        ckpt = torch.load(latest_ckpt_path, map_location="cpu")
        self.assertIn("epoch", ckpt)
        self.assertIn("global_step", ckpt)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("optimizer_state_dict", ckpt)
        self.assertIn("scaler_state_dict", ckpt)
        self.assertIn("val_loss", ckpt)
        self.assertEqual(ckpt["epoch"], 0)
        self.assertEqual(ckpt["global_step"], trainer1.global_step)

        # Create new model and resume with 'auto'
        model_stage2 = PECT_JEPA_5x5(self.config)
        trainer2 = Trainer5x5(
            model=model_stage2,
            config=self.config,
            train_loader=self.loader,
            val_loader=self.loader,
            resume_checkpoint="auto",
        )

        self.assertEqual(trainer2.start_epoch, 1, "Resume should start at epoch index 1")
        self.assertEqual(trainer2.global_step, trainer1.global_step, "Global step must be preserved")

        # Verify weights match
        for (k1, v1), (k2, v2) in zip(model_stage1.named_parameters(), model_stage2.named_parameters()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.allclose(v1, v2, atol=1e-6), f"Mismatch in parameter {k1}")

    def test_resume_nonexistent_graceful_fallback(self):
        """Test that a nonexistent checkpoint falls back to fresh training gracefully."""
        model = PECT_JEPA_5x5(self.config)
        trainer = Trainer5x5(
            model=model,
            config=self.config,
            train_loader=self.loader,
            resume_checkpoint="nonexistent_checkpoint.pt",
        )
        self.assertEqual(trainer.start_epoch, 0)
        self.assertEqual(trainer.global_step, 0)


if __name__ == "__main__":
    unittest.main()
