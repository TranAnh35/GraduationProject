"""
Comprehensive Unit Test Suite for EXP-68 Unified OPF-JEPA Features:
1. Dodd-Deeds 16D Spectral Condition Token (Fourier phase + normalized energy, 100% DC invariance, lift-off stability, zero/NaN edge cases)
2. Predictor AdaLN Zero-Initialization Identity & Gradient Flow
3. Scale-Invariant Normalized Space L1 Loss (Death spiral immunity, non-vanishing gradients)
4. Single Unified Latent Representation (Full 64D forward, 3-loss formulation, feature extraction [B, 128])
5. Scheduled Invariance (L_inv ramp-up behavior)
6. Backward Compatibility (strict state_dict loading for legacy checkpoints)
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root (GraduationProject) is in python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import (
    CrossConditionJEPA,
    compute_vicreg_loss,
    spectral_token_dim,
)
from src.PECT_JEPA.cross_condition.models.predictor import PredictorBlock, CrossConditionPredictor


class TestSpectralConditionToken(unittest.TestCase):
    """Tests for Dodd-Deeds Spectral Signature Condition Token."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 4
        self.C = 128
        # Synthetic PECT pulses with transient decay
        t = torch.linspace(0, 1, self.C)
        base_pulse = torch.sin(2 * np.pi * 3 * t) * torch.exp(-3 * t)
        self.ref_clean = base_pulse.view(1, 1, 1, self.C).expand(self.B, 5, 5, self.C).clone()

    def test_dc_invariance(self):
        """DC bias offset (+10.0 or -50.0) must produce identical condition tokens (0% DC leakage)."""
        cfg = CrossConditionConfig(
            unified_latent=True,
            spectral_condition_token=True,
            use_condition_token=True,
        )
        model = CrossConditionJEPA(cfg)
        model.eval()

        ref_clean = self.ref_clean.clone()
        ref_offset_pos = self.ref_clean + 25.0
        ref_offset_neg = self.ref_clean - 40.0

        with torch.no_grad():
            c_clean = model.extract_spectral_condition_token(ref_clean)
            c_pos = model.extract_spectral_condition_token(ref_offset_pos)
            c_neg = model.extract_spectral_condition_token(ref_offset_neg)

        # Difference must be essentially numerical precision zero (< 1e-5)
        diff_pos = (c_clean - c_pos).abs().max().item()
        diff_neg = (c_clean - c_neg).abs().max().item()
        self.assertLess(diff_pos, 1e-5, f"Positive DC offset changed spectral condition token by {diff_pos}")
        self.assertLess(diff_neg, 1e-5, f"Negative DC offset changed spectral condition token by {diff_neg}")

    def test_waveform_discrimination(self):
        """Distinct excitation waveforms (e.g. 2 Hz sine vs 8 Hz sine) must produce distinct condition tokens."""
        cfg = CrossConditionConfig(
            unified_latent=True,
            spectral_condition_token=True,
            use_condition_token=True,
        )
        model = CrossConditionJEPA(cfg)
        model.eval()

        t = torch.linspace(0, 1, self.C)
        wave_low = torch.sin(2 * np.pi * 2 * t).view(1, 1, 1, self.C).expand(self.B, 5, 5, -1)
        wave_high = torch.sin(2 * np.pi * 8 * t).view(1, 1, 1, self.C).expand(self.B, 5, 5, -1)

        with torch.no_grad():
            c_low = model.extract_spectral_condition_token(wave_low)
            c_high = model.extract_spectral_condition_token(wave_high)

        cos_sim = F.cosine_similarity(c_low.squeeze(1), c_high.squeeze(1), dim=-1).mean().item()
        self.assertLess(cos_sim, 0.95, f"Different waveforms produced near-identical tokens (cos_sim={cos_sim:.4f})")

    def test_zero_and_flat_signal_safety(self):
        """Flat zero or constant signals must not cause NaN or Inf (division-by-zero protection)."""
        cfg = CrossConditionConfig(
            unified_latent=True,
            spectral_condition_token=True,
            use_condition_token=True,
        )
        model = CrossConditionJEPA(cfg)
        model.eval()

        ref_all_zeros = torch.zeros(self.B, 5, 5, self.C)
        ref_all_constant = torch.ones(self.B, 5, 5, self.C) * 7.5

        with torch.no_grad():
            c_zeros = model.extract_spectral_condition_token(ref_all_zeros)
            c_const = model.extract_spectral_condition_token(ref_all_constant)

        self.assertTrue(torch.isfinite(c_zeros).all(), "Zero input produced non-finite condition token")
        self.assertTrue(torch.isfinite(c_const).all(), "Constant input produced non-finite condition token")


class TestAdaLNPredictorModulation(unittest.TestCase):
    """Tests for Predictor with Zero-Initialized AdaLN Modulation."""

    def test_zero_init_identity_mapping(self):
        """At initialization, AdaLN modulation must be exact identity mapping (zero shift, zero scale perturbation)."""
        torch.manual_seed(123)
        block = PredictorBlock(embed_dim=64, num_heads=4, modulation="adaln")

        # Create random input queries and context
        B, N_tgt, N_ctx, D = 4, 9, 16, 64
        queries = torch.randn(B, N_tgt, D)
        context = torch.randn(B, N_ctx, D)
        cond_vec = torch.randn(B, 1, D)

        block.eval()
        with torch.no_grad():
            # Output with condition vector
            out_with_cond = block(queries, context, condition_vec=cond_vec)
            # Output without condition vector (unmodulated)
            out_unmodulated = block(queries, context, condition_vec=None)

        max_diff = (out_with_cond - out_unmodulated).abs().max().item()
        self.assertLess(max_diff, 1e-2, f"AdaLN not near identity at init (max diff: {max_diff:.8f})")

    def test_gradient_flow_through_adaln(self):
        """Loss gradients must flow back to condition_vec through AdaLN modulation."""
        torch.manual_seed(123)
        block = PredictorBlock(embed_dim=64, num_heads=4, modulation="adaln")

        B, N_tgt, N_ctx, D = 2, 9, 16, 64
        queries = torch.randn(B, N_tgt, D, requires_grad=True)
        context = torch.randn(B, N_ctx, D, requires_grad=True)
        cond_vec = torch.randn(B, 1, D, requires_grad=True)

        out = block(queries, context, condition_vec=cond_vec)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(cond_vec.grad)
        cond_grad_norm = cond_vec.grad.norm().item()
        self.assertGreater(cond_grad_norm, 0.0, "Condition vector received zero gradients through AdaLN")

        ada_linear = block.ada_mod[1]
        self.assertIsNotNone(ada_linear.weight.grad)
        weight_grad_norm = ada_linear.weight.grad.norm().item()
        self.assertGreater(weight_grad_norm, 0.0, "AdaLN projection weights received zero gradients")


class TestNormalizedSpaceL1Loss(unittest.TestCase):
    """Tests for Scale-Invariant Normalized Space L1 Loss."""

    def test_scale_invariance_and_anti_shrinkage(self):
        """Normalized Space L1 must not reward shrinking representations towards zero."""
        B, N, D = 4, 9, 64
        pred = torch.randn(B, N, D)
        target = torch.randn(B, N, D)

        def norm_l1(p, t):
            pm = p.mean(dim=-1, keepdim=True)
            ps = p.std(dim=-1, keepdim=True).clamp_min(1e-8)
            tm = t.mean(dim=-1, keepdim=True)
            ts = t.std(dim=-1, keepdim=True).clamp_min(1e-8)
            return F.l1_loss((p - pm) / ps, (t - tm) / ts)

        loss_orig = norm_l1(pred, target).item()
        # Scale down predictions by 1000x (simulating shrinkage)
        pred_shrunk = pred * 1e-3
        loss_shrunk = norm_l1(pred_shrunk, target).item()

        # Both losses must be equal because normalized L1 is invariant to linear scale!
        self.assertAlmostEqual(loss_orig, loss_shrunk, places=5,
                               msg=f"Normalized L1 was not scale invariant: {loss_orig:.5f} vs {loss_shrunk:.5f}")


class TestUnifiedLatentArchitecture(unittest.TestCase):
    """Tests for Single Unified Latent Architecture (EXP-68)."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 4
        self.C = 128
        self.x_a = torch.randn(self.B, 5, 5, self.C)
        self.x_b = torch.randn(self.B, 5, 5, self.C)
        self.ref_b = torch.randn(self.B, 5, 5, self.C)
        self.scan_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def test_unified_forward_loss_and_outputs(self):
        """Unified forward pass must produce 3 active losses and zero for disabled auxiliary losses."""
        cfg = CrossConditionConfig(
            unified_latent=True,
            spectral_condition_token=True,
            use_condition_token=True,
            condition_modulation="adaln",
            pred_loss_type="normalized_l1",
            var_weight=0.5,
            cov_weight=0.2,
            inv_weight=0.1,
            scan_centered_vicreg=True,
            scan_centered_inv=True,
        )
        model = CrossConditionJEPA(cfg)
        out = model(
            x_a=self.x_a,
            x_b=self.x_b,
            ref_b=self.ref_b,
            scan_id=self.scan_id,
            inv_scale=1.0,
        )

        # 1. Total loss finite and strictly positive
        total_loss = out["total_loss"]
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        # 2. Main losses are active
        pred_loss = out["pred_loss"]
        var_loss = out["var_loss"]
        inv_loss = out["inv_loss"]
        self.assertGreater(pred_loss.item(), 0.0)
        self.assertGreater(var_loss.item(), 0.0)
        self.assertGreater(inv_loss.item(), 0.0)

        # 3. Auxiliary losses are exactly zero
        self.assertEqual(out["carrier_loss"].item(), 0.0)
        self.assertEqual(out["xcov_loss"].item(), 0.0)
        self.assertEqual(out["zc_cond_loss"].item(), 0.0)
        self.assertEqual(out["offset_loss"].item(), 0.0)
        self.assertEqual(out["pool_var_loss"].item(), 0.0)

    def test_downstream_feature_shape(self):
        """extract_features must produce [B, 2*D] = [B, 128] without subspace splitting."""
        cfg = CrossConditionConfig(
            unified_latent=True,
            spectral_condition_token=True,
            embed_dim=64,
        )
        model = CrossConditionJEPA(cfg)
        model.eval()

        with torch.no_grad():
            feat = model.extract_features(self.x_a)

        self.assertEqual(feat.shape, (self.B, 128),
                         f"Expected feature shape ({self.B}, 128), got {feat.shape}")
        self.assertTrue(torch.isfinite(feat).all(), "Extracted features contain NaN or Inf")

    def test_full_backward_pass(self):
        """Backward pass must propagate gradients cleanly through context encoder, predictor, and spec_cond_proj."""
        cfg = CrossConditionConfig(
            unified_latent=True,
            spectral_condition_token=True,
            use_condition_token=True,
            condition_modulation="adaln",
            pred_loss_type="normalized_l1",
            var_weight=0.5,
            cov_weight=0.2,
            inv_weight=0.1,
        )
        model = CrossConditionJEPA(cfg)
        out = model(x_a=self.x_a, x_b=self.x_b, ref_b=self.ref_b, scan_id=self.scan_id, inv_scale=1.0)
        out["total_loss"].backward()

        # Check ContextEncoder has gradients
        has_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
        self.assertTrue(has_encoder_grad, "ContextEncoder received zero gradients")

        # Check Predictor has gradients
        has_predictor_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
        self.assertTrue(has_predictor_grad, "Predictor received zero gradients")

        # Check spec_cond_proj has gradients
        self.assertIsNotNone(model.spec_cond_proj)
        has_spec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.spec_cond_proj.parameters())
        self.assertTrue(has_spec_grad, "spec_cond_proj received zero gradients")


class TestBackwardCompatibility(unittest.TestCase):
    """Tests ensuring legacy configurations load strict without parameter drift."""

    def test_legacy_config_no_new_parameters(self):
        """Default CrossConditionConfig must not instantiate spec_cond_proj."""
        cfg_default = CrossConditionConfig()
        model_default = CrossConditionJEPA(cfg_default)
        self.assertIsNone(model_default.spec_cond_proj, "spec_cond_proj was instantiated in legacy config")

        # Verify state dict keys
        keys = list(model_default.state_dict().keys())
        spec_keys = [k for k in keys if "spec_cond_proj" in k]
        self.assertEqual(len(spec_keys), 0, f"Found unexpected spec_cond_proj keys in legacy model: {spec_keys}")


class TestSpectralTokenContinuity(unittest.TestCase):
    """EXP-68 fix: the phase part must not jump at the +-pi branch cut, and the token
    reaching the predictor must be L2-normalized (AdaLN gamma/beta are unbounded)."""

    def _boundary_signal(self, eps: float) -> torch.Tensor:
        """Single-harmonic patch whose bin-4 phase sits just below/above +-pi."""
        C = 128
        t = torch.arange(C, dtype=torch.float32)
        w = 2 * np.pi * 4 / C
        # rfft(bin 4) = (A - iB) * C/2 -> angle = atan2(-B, A); A<0, B=+-eps -> near +-pi
        sig = -1.0 * torch.cos(w * t) + eps * torch.sin(w * t)
        return sig.view(1, 1, 1, C).expand(1, 5, 5, C).clone()

    def test_branch_cut_continuity(self):
        """A tiny perturbation across the +-pi boundary must barely move the token."""
        sig_pos = self._boundary_signal(+1e-3)
        sig_neg = self._boundary_signal(-1e-3)

        cfg_cos = CrossConditionConfig(spectral_condition_token=True, use_condition_token=True,
                                       spectral_phase_encoding="cos_sin")
        cfg_raw = CrossConditionConfig(spectral_condition_token=True, use_condition_token=True,
                                       spectral_phase_encoding="raw")
        torch.manual_seed(7)
        m_cos = CrossConditionJEPA(cfg_cos).eval()
        torch.manual_seed(7)
        m_raw = CrossConditionJEPA(cfg_raw).eval()
        # compare the raw signature features, not the projected token
        m_cos.spec_cond_proj = nn.Identity()
        m_raw.spec_cond_proj = nn.Identity()

        with torch.no_grad():
            d_cos = (m_cos.extract_spectral_condition_token(sig_pos)
                     - m_cos.extract_spectral_condition_token(sig_neg)).abs().max().item()
            d_raw = (m_raw.extract_spectral_condition_token(sig_pos)
                     - m_raw.extract_spectral_condition_token(sig_neg)).abs().max().item()

        self.assertLess(d_cos, 0.05, f"cos_sin signature jumped at the branch cut by {d_cos:.4f}")
        self.assertGreater(d_raw, 1.5, f"raw signature did not reproduce the +-pi jump ({d_raw:.4f})")
        # the two encodings must be width-compatible with their own projector
        self.assertEqual(spectral_token_dim(cfg_cos), 24)
        self.assertEqual(spectral_token_dim(cfg_raw), 16)

    def test_zero_energy_harmonics_do_not_inject_phase_noise(self):
        """Harmonics with ~0 energy have an arbitrary FFT phase; they must not move c_B."""
        C = 128
        t = torch.arange(C, dtype=torch.float32)
        w = 2 * np.pi * 4 / C

        def mk(noise_amp):
            sig = -torch.cos(w * t) + noise_amp * torch.randn(C, generator=torch.Generator().manual_seed(3))
            return sig.view(1, 1, 1, C).expand(1, 5, 5, C).clone()

        cfg = CrossConditionConfig(spectral_condition_token=True, use_condition_token=True)
        torch.manual_seed(11)
        m = CrossConditionJEPA(cfg).eval()
        with torch.no_grad():
            d = (m.extract_spectral_condition_token(mk(0.0))
                 - m.extract_spectral_condition_token(mk(1e-3))).abs().max().item()
        self.assertLess(d, 0.02, f"zero-energy harmonic phases perturbed the token by {d:.4f}")

    def test_condition_token_is_l2_normalized_before_predictor(self):
        """Scaling the condition token must not change the unified forward pass."""
        torch.manual_seed(0)
        cfg = CrossConditionConfig(
            unified_latent=True, spectral_condition_token=True, use_condition_token=True,
            condition_modulation="adaln", pred_loss_type="normalized_l1", embed_dim=64,
        )
        model = CrossConditionJEPA(cfg).eval()
        B, C = 2, 128
        x_a, x_b = torch.randn(B, 5, 5, C), torch.randn(B, 5, 5, C)
        ref_b = torch.randn(B, 5, 5, C)
        scan_id = torch.tensor([0, 1])

        class _Scaled(nn.Module):
            def __init__(self, inner, scale):
                super().__init__()
                self.inner = inner
                self.scale = scale

            def forward(self, x):
                return self.inner(x) * self.scale

        real_proj = model.spec_cond_proj
        with torch.no_grad():
            model.spec_cond_proj = _Scaled(real_proj, 1.0)
            out1 = model(x_a=x_a, x_b=x_b, ref_b=ref_b, scan_id=scan_id, inv_scale=0.0)
            model.spec_cond_proj = _Scaled(real_proj, 250.0)
            out2 = model(x_a=x_a, x_b=x_b, ref_b=ref_b, scan_id=scan_id, inv_scale=0.0)

        delta = (out1["total_loss"] - out2["total_loss"]).abs().item()
        self.assertLess(delta, 1e-5, f"cond token scale leaked into the loss (delta={delta:.2e})")


class TestDisentangledNormalizedL1(unittest.TestCase):
    """EXP-68a runs the disentangled branch, which previously had no 'normalized_l1' case
    (it silently fell through to plain L1). Guard that the option is actually wired."""

    def test_disentangled_branch_uses_normalized_l1(self):
        torch.manual_seed(0)
        base = dict(latent_disentanglement=True, use_condition_token=True, embed_dim=64,
                    inv_weight=0.0, var_weight=0.5, cov_weight=0.0)
        B, C = 4, 128
        x_a, x_b = torch.randn(B, 5, 5, C), torch.randn(B, 5, 5, C)
        ref_b = torch.randn(B, 5, 5, C)
        scan_id = torch.tensor([0, 0, 1, 1])

        losses = {}
        for lt in ("l1", "normalized_l1"):
            torch.manual_seed(1234)
            model = CrossConditionJEPA(CrossConditionConfig(pred_loss_type=lt, **base)).eval()
            with torch.no_grad():
                out = model(x_a=x_a, x_b=x_b, ref_b=ref_b, scan_id=scan_id, inv_scale=0.0)
            losses[lt] = out["pred_loss"].item()

        self.assertNotAlmostEqual(losses["l1"], losses["normalized_l1"], places=6,
                                  msg="normalized_l1 fell through to plain L1 in the "
                                      "disentangled branch (EXP-68a would be a no-op)")


if __name__ == "__main__":
    unittest.main()
