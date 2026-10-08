"""
Unit test suite for Radial Inward Diffusion Masking (EXP-34).
Tests:
1. RadialDiffusionMasker5x5 shape and partition properties.
2. Radial Diffusion Masker factory integration in build_masker_5x5.
3. PECT_JEPA_5x5 forward pass, backward gradient flow, and loss dictionary.
4. Downstream representation extraction consistency (unified, foundation, depth, anomaly score).
"""

import unittest
import torch

from src.PECT_JEPA.spatiotemporal_5x5.masking.radial_mask import RadialDiffusionMasker5x5
from src.PECT_JEPA.spatiotemporal_5x5.masking.cluster_mask import build_masker_5x5
from src.PECT_JEPA.spatiotemporal_5x5.configs.config import Spatiotemporal5x5Config
from src.PECT_JEPA.spatiotemporal_5x5.models.jepa_5x5 import PECT_JEPA_5x5


class TestRadialDiffusionMasker5x5(unittest.TestCase):

    def test_inward_core_partition(self):
        """Test that inward_core partitions 25 probes into 16 outer context and 9 inner core targets."""
        masker = RadialDiffusionMasker5x5(mode="inward_core", use_mask_bank=False)
        ctx, tgt, mask_bool = masker.sample_mask(batch_size=8)

        self.assertEqual(ctx.shape, (8, 16))
        self.assertEqual(tgt.shape, (8, 9))
        self.assertEqual(mask_bool.shape, (8, 25))

        # Context must be probes 9..24 (Ring 2 at r=3mm and Ring 3 at r=7mm)
        expected_ctx = list(range(9, 25))
        self.assertEqual(ctx[0].tolist(), expected_ctx)

        # Target must be probes 0..8 (Center at r=0 and Ring 1 at r=1mm)
        expected_tgt = list(range(0, 9))
        self.assertEqual(tgt[0].tolist(), expected_tgt)

        # mask_bool must be True exactly at target indices
        for b in range(8):
            self.assertTrue(torch.all(mask_bool[b, :9]))
            self.assertFalse(torch.any(mask_bool[b, 9:]))

    def test_inward_center_partition(self):
        """Test that inward_center partitions into 16 outer context and 1 center target."""
        masker = RadialDiffusionMasker5x5(mode="inward_center", use_mask_bank=False)
        ctx, tgt, mask_bool = masker.sample_mask(batch_size=4)

        self.assertEqual(ctx.shape, (4, 16))
        self.assertEqual(tgt.shape, (4, 1))
        self.assertEqual(tgt[0].item(), 0)
        self.assertEqual(ctx[0].tolist(), list(range(9, 25)))

    def test_mask_bank_sampling(self):
        """Test that GPU/CPU mask bank returns correct shapes and device."""
        masker = RadialDiffusionMasker5x5(mode="inward_core", use_mask_bank=True, bank_size=128)
        device = torch.device("cpu")
        ctx, tgt, mask_bool = masker.sample_mask(batch_size=16, device=device, seed=42)

        self.assertEqual(ctx.shape, (16, 16))
        self.assertEqual(tgt.shape, (16, 9))
        self.assertEqual(mask_bool.shape, (16, 25))
        self.assertEqual(ctx.device, device)

    def test_factory_builder(self):
        """Test build_masker_5x5 factory correctly builds radial_diffusion masker."""
        masker = build_masker_5x5("radial_diffusion", radial_mask_mode="inward_core")
        self.assertIsInstance(masker, RadialDiffusionMasker5x5)
        self.assertEqual(masker.mode, "inward_core")

    def test_jepa_5x5_integration(self):
        """Test end-to-end PECT_JEPA_5x5 forward, backward, and representation extraction with radial mask."""
        config = Spatiotemporal5x5Config(
            in_channels=128,
            embed_dim=64,
            encoder_depth=2,
            predictor_depth=2,
            spatial_topology="concentric_star",
            star_radii=(1, 3, 7),
            masker_type="radial_diffusion",
            radial_mask_mode="inward_core",
            relative_perturbation_target=True,
            feature_extraction_mode="unified",
        )
        model = PECT_JEPA_5x5(config)
        masker = build_masker_5x5("radial_diffusion", radial_mask_mode="inward_core")

        B = 4
        x = torch.randn(B, 5, 5, 128)
        ctx_idx, tgt_idx, mask_bool = masker.sample_mask(batch_size=B)

        # 1. Forward pass with internal masker
        loss_dict_auto = model(x=x)
        total_loss_auto = loss_dict_auto["loss"]
        self.assertFalse(torch.isnan(total_loss_auto))
        self.assertFalse(torch.isinf(total_loss_auto))
        self.assertGreater(total_loss_auto.item(), 0.0)

        # Forward pass with explicit custom indices
        loss_dict = model(
            x=x,
            custom_context_indices=ctx_idx,
            custom_target_indices=tgt_idx,
        )

        total_loss = loss_dict["loss"]
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        # 2. Backward pass
        total_loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

        # 3. Downstream Feature Extraction
        model.eval()
        with torch.no_grad():
            # Unified features: [B, 2 * D]
            z_unified = model.extract_unified_features(x)
            self.assertEqual(z_unified.shape, (B, 128))
            self.assertFalse(torch.isnan(z_unified).any())

            # Foundation features: [B, 2 * D]
            z_found = model.extract_foundation_representation(x)
            self.assertEqual(z_found.shape, (B, 128))
            self.assertFalse(torch.isnan(z_found).any())

            # Unified & Depth features: [B, 2 * D] and [B, 4]
            z_u, v_depth = model.extract_unified_and_depth_features(x)
            self.assertEqual(z_u.shape, (B, 128))
            self.assertEqual(v_depth.shape, (B, 4))
            self.assertFalse(torch.isnan(v_depth).any())

            # Center anomaly score: [B]
            anomaly_score = model.compute_center_anomaly_score(x)
            self.assertEqual(anomaly_score.shape, (B,))
            self.assertFalse(torch.isnan(anomaly_score).any())
            self.assertTrue(torch.all(anomaly_score >= 0.0))


if __name__ == "__main__":
    unittest.main()
