"""
Unit and integration test for EXP-11:
Pure JEPA with DualDomainAttentionTokenizer5x5, ResidualDiffusionPredictor5x5,
ContiguousClusterMasker5x5, Single Encoder + Stop-Gradient Target, and zero contractive losses.
"""

import unittest
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.tokenizer_5x5 import DualDomainAttentionTokenizer5x5
from src.PECT_JEPA.spatiotemporal_5x5.models.predictor import ResidualDiffusionPredictor5x5
from src.PECT_JEPA.spatiotemporal_5x5.masking.cluster_mask import ContiguousClusterMasker5x5, build_masker_5x5


class TestEXP11Pipeline(unittest.TestCase):
    def setUp(self):
        self.config = Spatiotemporal5x5Config(
            grid_size=5,
            in_channels=128,
            embed_dim=64,
            tokenizer_type="dual_domain_attention",
            predictor_type="residual_diffusion",
            masker_type="contiguous_cluster",
            min_masked=10,
            max_masked=15,
            use_target_ema=False,
            liftoff_invar_weight=0.0,
            phase_align_weight=0.0,
            fluct_weight=2.0,
            var_weight=1.0,
            cov_weight=1.0,
            learning_rate=3e-4,
        )

    def test_model_construction(self):
        model = PECT_JEPA_5x5(self.config)
        self.assertIsInstance(model.tokenizer, DualDomainAttentionTokenizer5x5)
        self.assertIsInstance(model.predictor, ResidualDiffusionPredictor5x5)
        masker = build_masker_5x5(self.config)
        self.assertIsInstance(masker, ContiguousClusterMasker5x5)

    def test_forward_and_loss(self):
        model = PECT_JEPA_5x5(self.config)
        model.train()
        B = 4
        x = torch.randn(B, 5, 5, 128)
        loss_dict = model(x)

        self.assertIn("loss", loss_dict)
        self.assertIn("loss_pred", loss_dict)
        self.assertIn("loss_var", loss_dict)
        self.assertIn("loss_cov", loss_dict)

        loss = loss_dict["loss"]
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0.0)

        # Backward pass verification
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and "target_encoder" not in name and "depth_head" not in name:
                self.assertIsNotNone(param.grad, f"Missing grad for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad in {name}")

    def test_extract_center_feature(self):
        model = PECT_JEPA_5x5(self.config)
        model.eval()
        B = 2
        x = torch.randn(B, 5, 5, 128)
        feat = model.extract_center_feature(x)
        self.assertEqual(feat.shape, (B, 64))
        self.assertFalse(torch.isnan(feat).any())

    def test_multi_waveform_shapes(self):
        """Test with different batch sizes representing multi-waveform batches."""
        model = PECT_JEPA_5x5(self.config)
        model.train()
        for B in [1, 2, 8]:
            x = torch.randn(B, 5, 5, 128)
            loss_dict = model(x)
            self.assertFalse(torch.isnan(loss_dict["loss"]))


if __name__ == "__main__":
    unittest.main()
