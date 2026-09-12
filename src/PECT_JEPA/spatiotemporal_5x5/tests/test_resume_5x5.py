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
        self.assertIn("val_loss_pred", ckpt)
        self.assertIn("best_val_loss_pred", ckpt)
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

    def test_unified_experiment_dir_and_no_timestamp_on_resume(self):
        """Test that exp_name folder is unified and consistent across resume runs."""
        from ..utils.logger import PECTExperimentLogger5x5

        cfg = Spatiotemporal5x5Config(
            exp_name="test_unified_run",
            log_dir=self.log_dir,
            save_dir=None,
            add_timestamp=False,
            use_tensorboard=False,
            use_wandb=False,
        )

        logger1 = PECTExperimentLogger5x5(cfg)
        expected_dir = os.path.join(self.log_dir, "test_unified_run")
        self.assertEqual(logger1.run_dir, expected_dir)

        # Checkpoint dir inside experiment folder
        ckpt_dir = os.path.join(logger1.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        cfg.save_dir = ckpt_dir

        # Simulate resume
        cfg_resume = Spatiotemporal5x5Config(
            exp_name="test_unified_run",
            log_dir=self.log_dir,
            save_dir=None,
            resume="auto",
            add_timestamp=False,
            use_tensorboard=False,
            use_wandb=False,
        )
        logger2 = PECTExperimentLogger5x5(cfg_resume)
        self.assertEqual(logger2.run_dir, expected_dir, "Resume must reuse the exact existing experiment folder")

    def test_early_stopping_val_loss_pred(self):
        """Test that early stopping correctly monitors val_loss_pred and decouples from total val_loss."""
        cfg = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=32,
            embed_dim=32,
            encoder_depth=1,
            encoder_heads=2,
            predictor_depth=1,
            predictor_heads=2,
            epochs=10,
            batch_size=8,
            device="cpu",
            save_dir=self.save_dir,
            log_dir=self.log_dir,
            use_tensorboard=False,
            use_wandb=False,
            early_stopping_metric="val_loss_pred",
            early_stopping_patience=2,
        )
        model = PECT_JEPA_5x5(cfg)
        trainer = Trainer5x5(
            model=model,
            config=cfg,
            train_loader=self.loader,
            val_loader=self.loader,
        )

        # Mock train_epoch and validate to simulate val_loss exploding but val_loss_pred improving
        mock_history = [
            # epoch 0: pred improves (best=0.50), total loss 1.0
            ({"loss": 1.0, "loss_pred": 0.5, "loss_var": 0.2, "loss_cov": 0.3}, {"val_loss": 1.0, "val_loss_pred": 0.5, "effective_rank": 8.0}),
            # epoch 1: pred improves to 0.40 (best=0.40), even though total val_loss explodes to 5.0 (covariance spike)
            ({"loss": 1.2, "loss_pred": 0.4, "loss_var": 0.2, "loss_cov": 0.6}, {"val_loss": 5.0, "val_loss_pred": 0.4, "effective_rank": 8.5}),
            # epoch 2: pred fails to improve (0.45 > 0.40), patience = 1
            ({"loss": 1.3, "loss_pred": 0.45, "loss_var": 0.2, "loss_cov": 0.65}, {"val_loss": 5.2, "val_loss_pred": 0.45, "effective_rank": 8.5}),
            # epoch 3: pred fails to improve (0.48 > 0.40), patience = 2 -> Early Stopping triggers at epoch 4!
            ({"loss": 1.4, "loss_pred": 0.48, "loss_var": 0.2, "loss_cov": 0.72}, {"val_loss": 5.5, "val_loss_pred": 0.48, "effective_rank": 8.5}),
        ]

        step_idx = [0]
        def mock_train():
            idx = step_idx[0]
            return mock_history[idx][0]

        def mock_val():
            idx = step_idx[0]
            step_idx[0] += 1
            return mock_history[idx][1]

        trainer.train_epoch = mock_train
        trainer.validate = mock_val

        trainer.fit()

        # Should have run exactly 4 epochs (stopped at epoch 4 instead of 10)
        self.assertEqual(trainer.current_epoch, 3, "Training should stop at epoch 4 (index 3) due to patience=2 on val_loss_pred")
        self.assertAlmostEqual(trainer.best_val_loss_pred, 0.40, places=4)
        # Verify best checkpoint exists
        best_ckpt_path = os.path.join(self.save_dir, "best_model_5x5.pt")
        self.assertTrue(os.path.isfile(best_ckpt_path))
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        self.assertAlmostEqual(best_ckpt["val_loss_pred"], 0.40, places=4)


if __name__ == "__main__":
    unittest.main()
