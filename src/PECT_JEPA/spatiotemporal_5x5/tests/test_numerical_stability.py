"""
Tests for numerical stability in JEPALoss5x5 and Trainer5x5 validate().
Verifies that FP16 inputs do not overflow in VICReg covariance/variance,
and that validation does not poison total_loss if an individual batch produces NaN.
"""

import os
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.losses.jepa_loss import JEPALoss5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.attention import MultiheadSelfAttention, MultiheadCrossAttention
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.training.trainer import Trainer5x5


class TestNumericalStability(unittest.TestCase):

    def test_vicreg_fp16_stability(self):
        """Verify that VICReg loss handles FP16 inputs without overflow/NaN."""
        loss_fn = JEPALoss5x5()
        B, N, D = 512, 12, 128
        # Values with magnitude 3.5 in FP16: inner product sum over 6144 exceeds 65504
        H_pred_fp16 = (torch.randn(B, N, D, dtype=torch.float16) * 3.5)
        H_tgt_fp16 = (torch.randn(B, N, D, dtype=torch.float16) * 3.5)

        loss_dict = loss_fn(H_pred_fp16, H_tgt_fp16)
        total_loss = loss_dict["loss"]

        self.assertFalse(torch.isnan(total_loss), "Loss should not be NaN for FP16 inputs")
        self.assertFalse(torch.isinf(total_loss), "Loss should not be Inf for FP16 inputs")
        self.assertEqual(total_loss.dtype, torch.float32, "Output loss must be in FP32")

    def test_variance_hinge_zero_variance(self):
        """Verify variance hinge handles zero-variance inputs without NaN."""
        loss_fn = JEPALoss5x5()
        # Constant tensor: variance is exactly 0.0
        H_pred = torch.ones(64, 12, 128, dtype=torch.float32)
        l_var = loss_fn.variance_hinge(H_pred)
        self.assertFalse(torch.isnan(l_var), "Variance hinge should not produce NaN on zero variance")
        self.assertGreater(float(l_var.item()), 0.0)

    def test_validate_nan_resilience(self):
        """Verify validate() skips NaN batches and averages over valid batches only."""
        config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=32,
            encoder_depth=1,
            predictor_depth=1,
            epochs=1,
            batch_size=4,
            mixed_precision=False,
            device="cpu",
            use_tensorboard=False,
            use_wandb=False,
        )
        model = PECT_JEPA_5x5(config)
        
        # Create a mock dataset with 3 batches of shape [B, 5, 5, 128]
        x_data = torch.randn(12, 5, 5, 128)
        dummy_loader = DataLoader(
            TensorDataset(x_data),
            batch_size=4,
            collate_fn=lambda b: {"data": torch.stack([item[0] for item in b])}
        )

        trainer = Trainer5x5(
            model=model,
            config=config,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
        )

        # Monkey-patch model forward to return NaN on the second batch
        orig_forward = model.forward
        call_count = [0]

        def flaky_forward(x, **kwargs):
            call_count[0] += 1
            res = orig_forward(x, **kwargs)
            if call_count[0] == 2:
                # Force NaN loss on batch 2
                res["loss"] = torch.tensor(float("nan"))
                res["loss_pred"] = torch.tensor(float("nan"))
            return res

        model.forward = flaky_forward

        val_metrics = trainer.validate()

        # Val loss must be valid (computed from batch 1 and 3)
        self.assertIn("val_loss", val_metrics)
        self.assertFalse(np.isnan(val_metrics["val_loss"]), "Val loss must not be poisoned by NaN batch")
        self.assertGreater(val_metrics["val_loss"], 0.0)
        self.assertIn("effective_rank", val_metrics)
        trainer.logger.close()

    def test_downstream_probe_execution(self):
        """Verify that downstream probe extracts C-scan, computes CNR, and saves heatmap."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=32,
            encoder_depth=1,
            predictor_depth=1,
            epochs=1,
            batch_size=4,
            device="cpu",
            log_dir=temp_dir,
            save_dir=os.path.join(temp_dir, "checkpoints"),
            use_tensorboard=False,
            use_wandb=False,
        )
        model = PECT_JEPA_5x5(config)
        trainer = Trainer5x5(
            model=model,
            config=config,
            train_loader=[],
            val_loader=None,
        )

        # Mock a small C-scan grid [15, 15, 128]
        trainer.probe_grid = np.random.randn(15, 15, 128).astype(np.float32)
        # Add a synthetic defect in the center
        trainer.probe_grid[6:9, 6:9, :] += 3.0
        trainer.probe_fname = "mock_defect_scan"

        probe_res = trainer.run_downstream_probe(epoch=1)

        self.assertIsNotNone(probe_res)
        self.assertIn("contrast_ratio_cnr", probe_res)
        self.assertGreater(probe_res["contrast_ratio_cnr"], 0.0)

        # Check that heatmap was created
        probe_dir = os.path.join(trainer.logger.run_dir, "probe_heatmaps")
        self.assertTrue(os.path.isdir(probe_dir))
        heatmaps = os.listdir(probe_dir)
        self.assertEqual(len(heatmaps), 1)
        self.assertTrue(heatmaps[0].endswith(".png"))

        trainer.logger.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_self_attention_fp16_extreme_stability(self):
        """Verify MultiheadSelfAttention handles extreme FP16 dot products without NaN/Inf."""
        attn = MultiheadSelfAttention(embed_dim=128, num_heads=4)
        attn.eval()
        # Create input with large variance (std=50.0) that would overflow naive FP16 dot products
        x = torch.randn(8, 25, 128, dtype=torch.float32) * 50.0
        out = attn(x)
        self.assertFalse(torch.isnan(out).any(), "Self-Attention produced NaN on extreme inputs")
        self.assertFalse(torch.isinf(out).any(), "Self-Attention produced Inf on extreme inputs")

    def test_cross_attention_fp16_extreme_stability(self):
        """Verify MultiheadCrossAttention handles extreme query/key inputs without NaN/Inf."""
        cross = MultiheadCrossAttention(embed_dim=128, num_heads=4)
        cross.eval()
        q = torch.randn(8, 15, 128, dtype=torch.float32) * 50.0
        kv = torch.randn(8, 10, 128, dtype=torch.float32) * 50.0
        out = cross(query=q, key_value=kv)
        self.assertFalse(torch.isnan(out).any(), "Cross-Attention produced NaN on extreme inputs")
        self.assertFalse(torch.isinf(out).any(), "Cross-Attention produced Inf on extreme inputs")

    def test_vicreg_context_gradient_flow(self):
        """Verify C-JEPA: VICReg applied to H_ctx directly flows gradients to context representation."""
        loss_fn = JEPALoss5x5(vicreg_target="context", var_weight=1.0, cov_weight=0.5)
        H_ctx = torch.randn(4, 10, 32, requires_grad=True)
        H_pred = torch.randn(4, 15, 32, requires_grad=True)
        H_tgt = torch.randn(4, 15, 32)

        loss_dict = loss_fn(H_pred, H_tgt, H_ctx=H_ctx)
        loss_dict["loss"].backward()

        self.assertIsNotNone(H_ctx.grad, "H_ctx must receive direct gradients when vicreg_target='context'")
        self.assertGreater(torch.norm(H_ctx.grad).item(), 0.0)

    def test_momentum_scheduler_clamping(self):
        """Verify momentum scheduler never exceeds 0.999 (preventing target freeze at 1.0)."""
        from src.PECT_JEPA.spatiotemporal_5x5.training.optimizer import MomentumScheduler5x5
        scheduler = MomentumScheduler5x5(base_momentum=0.996, final_momentum=0.999, total_steps=100)
        # Check initial, intermediate, and final step momentum values
        m_start = scheduler.get_momentum(0)
        m_mid = scheduler.get_momentum(50)
        m_end = scheduler.get_momentum(100)
        m_over = scheduler.get_momentum(150)

        self.assertAlmostEqual(m_start, 0.996, places=5)
        self.assertLess(m_start, m_mid)
        self.assertLessEqual(m_end, 0.999)
        self.assertLessEqual(m_over, 0.999)
        self.assertLess(m_end, 1.0, "Momentum must be strictly less than 1.0 to keep targets dynamic")


if __name__ == "__main__":
    unittest.main()
