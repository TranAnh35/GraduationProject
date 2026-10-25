"""
Model / loss contract tests for the canonical PECT-JEPA pipeline (exp70 lineage).

This file replaces the legacy `test_exp67_features.py` (two-stream / subspace
branches) and the exp68-only parts of `test_exp68_features.py`, both of which
tested code paths that no longer exist. It covers the surface the current
work order actually depends on:

1. Spectral condition token c_B: DC invariance, waveform discrimination,
   +-pi branch-cut continuity, zero-energy harmonic safety, L2 normalization.
2. Predictor AdaLN zero-init identity + gradient flow.
3. Normalized-space L1 (scale invariance / anti-shrinkage).
4. Unified forward contract: `aux_loss` / `inv_keep_frac` keys, anti-hijack cap.
5. `inv_predictor` (BYOL-style head) and the unsupervised background gate
   (`inv_trim_q`).
6. Canonical tokenizer: C+5 time input (C+6 with `global_bg_energy_anchor`),
   patch-polarity invariance, determinism.
7. exp70 checkpoint strict load (skipped when the local run is absent).

The CLI surface (train.py argparse -> CrossConditionConfig -> dataset) is
covered separately by `tests/test_train_cli_contract.py`.
"""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import (
    CrossConditionJEPA,
    corner_background_energy,
    spectral_token_dim,
)
from src.PECT_JEPA.cross_condition.models.predictor import PredictorBlock

EXP70_CKPT = os.path.join(
    REPO_ROOT, "experiments", "cross_condition", "runs",
    "exp70_full_cross_pairing_20261023_175853", "checkpoints",
    "best_cross_condition_jepa.pt",
)


class TestSpectralConditionToken(unittest.TestCase):
    """Dodd-Deeds spectral signature c_B (DC-free, branch-cut continuous)."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 4
        self.C = 128
        t = torch.linspace(0, 1, self.C)
        base_pulse = torch.sin(2 * np.pi * 3 * t) * torch.exp(-3 * t)
        self.ref_clean = base_pulse.view(1, 1, 1, self.C).expand(self.B, 5, 5, self.C).clone()

    def test_token_dim_is_canonical(self):
        self.assertEqual(spectral_token_dim(), 24)
        cfg = CrossConditionConfig()
        model = CrossConditionJEPA(cfg)
        self.assertEqual(model.spec_cond_proj[0].in_features, 24)

    def test_dc_invariance(self):
        """DC bias offset (+25 / -40) must not change the condition token."""
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        with torch.no_grad():
            c_clean = model.extract_spectral_condition_token(self.ref_clean)
            c_pos = model.extract_spectral_condition_token(self.ref_clean + 25.0)
            c_neg = model.extract_spectral_condition_token(self.ref_clean - 40.0)

        self.assertLess((c_clean - c_pos).abs().max().item(), 1e-5, "positive DC offset leaked into c_B")
        self.assertLess((c_clean - c_neg).abs().max().item(), 1e-5, "negative DC offset leaked into c_B")

    def test_waveform_discrimination(self):
        """Different excitation waveforms must map to different tokens."""
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        t = torch.linspace(0, 1, self.C)
        wave_low = torch.sin(2 * np.pi * 2 * t).view(1, 1, 1, self.C).expand(self.B, 5, 5, -1)
        wave_high = torch.sin(2 * np.pi * 8 * t).view(1, 1, 1, self.C).expand(self.B, 5, 5, -1)

        with torch.no_grad():
            c_low = model.extract_spectral_condition_token(wave_low)
            c_high = model.extract_spectral_condition_token(wave_high)

        cos_sim = F.cosine_similarity(c_low.squeeze(1), c_high.squeeze(1), dim=-1).mean().item()
        self.assertLess(cos_sim, 0.95, f"waveforms collapsed to one token (cos_sim={cos_sim:.4f})")

    def test_zero_and_flat_signal_safety(self):
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        zeros = torch.zeros(self.B, 5, 5, self.C)
        constant = torch.ones(self.B, 5, 5, self.C) * 7.5
        with torch.no_grad():
            c_zeros = model.extract_spectral_condition_token(zeros)
            c_const = model.extract_spectral_condition_token(constant)
        self.assertTrue(torch.isfinite(c_zeros).all())
        self.assertTrue(torch.isfinite(c_const).all())

    def _boundary_signal(self, eps: float) -> torch.Tensor:
        """Single-harmonic patch whose bin-4 phase sits just below/above +-pi."""
        C = 128
        t = torch.arange(C, dtype=torch.float32)
        w = 2 * np.pi * 4 / C
        sig = -1.0 * torch.cos(w * t) + eps * torch.sin(w * t)
        return sig.view(1, 1, 1, C).expand(1, 5, 5, C).clone()

    def test_branch_cut_continuity(self):
        """cos/sin encoding: a tiny perturbation across +-pi must barely move c_B."""
        torch.manual_seed(7)
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        model.spec_cond_proj = nn.Identity()
        with torch.no_grad():
            d = (model.extract_spectral_condition_token(self._boundary_signal(+1e-3))
                 - model.extract_spectral_condition_token(self._boundary_signal(-1e-3))).abs().max().item()
        self.assertLess(d, 0.05, f"c_B signature jumped at the +-pi branch cut by {d:.4f}")

    def test_zero_energy_harmonics_do_not_inject_phase_noise(self):
        C = 128
        t = torch.arange(C, dtype=torch.float32)
        w = 2 * np.pi * 4 / C

        def mk(noise_amp):
            sig = -torch.cos(w * t) + noise_amp * torch.randn(C, generator=torch.Generator().manual_seed(3))
            return sig.view(1, 1, 1, C).expand(1, 5, 5, C).clone()

        torch.manual_seed(11)
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        with torch.no_grad():
            d = (model.extract_spectral_condition_token(mk(0.0))
                 - model.extract_spectral_condition_token(mk(1e-3))).abs().max().item()
        self.assertLess(d, 0.02, f"zero-energy harmonic phases perturbed c_B by {d:.4f}")

    def test_condition_token_is_l2_normalized_before_predictor(self):
        """Scaling the raw token must not change the forward loss."""
        torch.manual_seed(0)
        cfg = CrossConditionConfig(condition_modulation="adaln", pred_loss_type="normalized_l1")
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
        self.assertLess(delta, 1e-5, f"condition-token scale leaked into the loss (delta={delta:.2e})")


class TestAdaLNPredictorModulation(unittest.TestCase):
    """Predictor with zero-initialized AdaLN modulation."""

    def test_zero_init_identity_mapping(self):
        torch.manual_seed(123)
        block = PredictorBlock(embed_dim=64, num_heads=4, modulation="adaln")
        B, N_tgt, N_ctx, D = 4, 9, 16, 64
        queries = torch.randn(B, N_tgt, D)
        context = torch.randn(B, N_ctx, D)
        cond_vec = torch.randn(B, 1, D)

        block.eval()
        with torch.no_grad():
            out_with = block(queries, context, condition_vec=cond_vec)
            out_without = block(queries, context, condition_vec=None)
        self.assertLess((out_with - out_without).abs().max().item(), 1e-2,
                        "AdaLN is not near-identity at initialization")

    def test_gradient_flow_through_adaln(self):
        torch.manual_seed(123)
        block = PredictorBlock(embed_dim=64, num_heads=4, modulation="adaln")
        B, N_tgt, N_ctx, D = 2, 9, 16, 64
        queries = torch.randn(B, N_tgt, D, requires_grad=True)
        context = torch.randn(B, N_ctx, D, requires_grad=True)
        cond_vec = torch.randn(B, 1, D, requires_grad=True)

        out = block(queries, context, condition_vec=cond_vec)
        out.sum().backward()

        self.assertIsNotNone(cond_vec.grad)
        self.assertGreater(cond_vec.grad.norm().item(), 0.0, "condition vector received zero gradient")
        self.assertGreater(block.ada_mod[1].weight.grad.norm().item(), 0.0,
                           "AdaLN projection weights received zero gradient")


class TestNormalizedSpaceL1Loss(unittest.TestCase):
    def test_scale_invariance_and_anti_shrinkage(self):
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
        loss_shrunk = norm_l1(pred * 1e-3, target).item()
        self.assertAlmostEqual(loss_orig, loss_shrunk, places=5,
                               msg="normalized L1 is not invariant to prediction scale")


class TestUnifiedForwardContract(unittest.TestCase):
    """Forward/loss contract: keys, gating, anti-hijack, gradients."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B, self.C = 4, 128
        self.x_a = torch.randn(self.B, 5, 5, self.C)
        self.x_b = torch.randn(self.B, 5, 5, self.C)
        self.ref_b = torch.randn(self.B, 5, 5, self.C)
        self.scan_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def _forward(self, cfg, inv_scale=1.0):
        model = CrossConditionJEPA(cfg)
        return model, model(x_a=self.x_a, x_b=self.x_b, ref_b=self.ref_b,
                            scan_id=self.scan_id, inv_scale=inv_scale)

    def test_loss_keys_and_values(self):
        cfg = CrossConditionConfig(condition_modulation="adaln", pred_loss_type="normalized_l1",
                                   var_weight=0.5, cov_weight=0.2, inv_weight=0.1)
        _, out = self._forward(cfg)

        for key in ("total_loss", "pred_loss", "var_loss", "cov_loss", "aux_loss",
                    "inv_loss", "inv_keep_frac"):
            self.assertIn(key, out)
            self.assertTrue(torch.isfinite(out[key]).all(), f"{key} is not finite")

        self.assertGreater(out["pred_loss"].item(), 0.0)
        self.assertGreater(out["var_loss"].item(), 0.0)
        self.assertGreater(out["inv_loss"].item(), 0.0)
        # No legacy carrier / xcov / zc / offset / pool keys anymore
        for legacy in ("carrier_loss", "xcov_loss", "zc_cond_loss", "offset_loss", "pool_var_loss"):
            self.assertNotIn(legacy, out)

    def test_inv_disabled_reports_full_keep(self):
        cfg = CrossConditionConfig(inv_weight=0.0)
        _, out = self._forward(cfg)
        self.assertEqual(out["inv_loss"].item(), 0.0)
        self.assertEqual(out["inv_keep_frac"].item(), 1.0)

    def test_background_gate_keeps_trim_q_fraction(self):
        """inv_trim_q selects the lowest-|Delta E| tokens (sound metal) for L_inv."""
        cfg = CrossConditionConfig(inv_weight=1.0, inv_trim_q=0.5, inv_predictor=False)
        _, out = self._forward(cfg)
        frac = out["inv_keep_frac"].item()
        self.assertLess(frac, 1.0, "background gate kept every token (no gating applied)")
        self.assertGreaterEqual(frac, 0.25)
        self.assertLessEqual(frac, 0.75)

    def test_anti_hijack_never_caps_vicreg(self):
        """FM22: VICReg keeps full strength; only the optional L_inv term may be capped.

        exp71 ran with anti_hijack_ratio=0.25 under the old cap-everything code:
        aux was pinned at exactly 0.25*pred every epoch, VICReg was starved and the
        representation collapsed (R_eff 1.57D-2.18D, val_auc 0.53).
        """
        cfg = CrossConditionConfig(inv_weight=0.0, var_weight=0.5, cov_weight=0.2,
                                   pred_loss_type="l1", anti_hijack_ratio=0.0)
        model = CrossConditionJEPA(cfg).eval()
        off = model(x_a=self.x_a, x_b=self.x_b, ref_b=self.ref_b,
                    scan_id=self.scan_id, inv_scale=1.0)
        model.config.anti_hijack_ratio = 0.25
        on = model(x_a=self.x_a, x_b=self.x_b, ref_b=self.ref_b,
                   scan_id=self.scan_id, inv_scale=1.0)

        ratio = on["aux_loss"].item() / off["aux_loss"].item()
        self.assertAlmostEqual(ratio, 1.0, places=6,
                               msg="anti_hijack_ratio scaled VICReg (FM22 death spiral)")
        self.assertAlmostEqual(
            on["aux_loss"].item(),
            0.5 * on["var_loss"].item() + 0.2 * on["cov_loss"].item(), places=6,
            msg="aux_loss is not the full-strength VICReg when L_inv is disabled")

    def test_anti_hijack_caps_only_inv_term(self):
        cfg = CrossConditionConfig(inv_weight=10.0, inv_trim_q=1.0, anti_hijack_ratio=0.25,
                                   var_weight=0.5, cov_weight=0.2, pred_loss_type="l1")
        _, out = self._forward(cfg)
        pred = out["pred_loss"].detach().item()
        vic = 0.5 * out["var_loss"].item() + 0.2 * out["cov_loss"].item()
        inv_term = out["aux_loss"].item() - vic
        self.assertGreater(inv_term, 0.0)
        self.assertLessEqual(inv_term, 0.25 * pred + 1e-5,
                             f"L_inv term {inv_term:.4f} exceeded 0.25 * pred_loss {pred:.4f}")
        self.assertAlmostEqual(out["total_loss"].item(), pred + out["aux_loss"].item(), places=5)

    def test_inv_predictor_head_receives_gradient(self):
        cfg = CrossConditionConfig(inv_weight=1.0, inv_predictor=True, inv_trim_q=1.0)
        model, out = self._forward(cfg)
        self.assertIsNotNone(model.inv_head)
        out["total_loss"].backward()

        self.assertGreater(model.inv_head[0].weight.grad.abs().sum().item(), 0.0,
                           "inv_head received no gradient")
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.context_encoder.parameters()),
                        "context encoder received no gradient")

    def test_default_config_has_no_inv_head(self):
        model = CrossConditionJEPA(CrossConditionConfig())
        self.assertIsNone(model.inv_head)
        self.assertEqual([k for k in model.state_dict() if k.startswith("inv_head")], [])

    def test_feature_shape_and_global_bg_anchor(self):
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        with torch.no_grad():
            feat = model.extract_features(self.x_a)
        self.assertEqual(feat.shape, (self.B, 128))
        self.assertTrue(torch.isfinite(feat).all())

        cfg_anchor = CrossConditionConfig(global_bg_energy_anchor=True)
        self.assertEqual(cfg_anchor.global_bg_energy_anchor, True)
        model_anchor = CrossConditionJEPA(cfg_anchor).eval()
        self.assertEqual(model_anchor.tokenizer.time_proj.in_features, self.C + 6)
        bg = torch.full((self.B,), 0.25)
        with torch.no_grad():
            feat_anchor = model_anchor.extract_features(self.x_a, bg_ref_energy=bg)
        self.assertEqual(feat_anchor.shape, (self.B, 128))
        self.assertTrue(torch.isfinite(feat_anchor).all())

    def test_corner_background_energy_matches_dataset_definition(self):
        cscan = np.random.RandomState(0).randn(270, 270, 128).astype(np.float32)
        corners = np.stack([cscan[0:5, 0:5, :], cscan[0:5, -5:, :],
                            cscan[-5:, 0:5, :], cscan[-5:, -5:, :]], axis=0)
        ref = np.median(corners, axis=0).reshape(-1, 128)
        expected = float(np.median(ref.std(axis=-1) ** 2))
        self.assertAlmostEqual(corner_background_energy(cscan), expected, places=6)

    def test_full_backward_pass(self):
        cfg = CrossConditionConfig(condition_modulation="adaln", pred_loss_type="normalized_l1",
                                   var_weight=0.5, cov_weight=0.2, inv_weight=0.1)
        model, out = self._forward(cfg)
        out["total_loss"].backward()

        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.context_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.predictor.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.spec_cond_proj.parameters()))

    def test_forward_is_deterministic(self):
        cfg = CrossConditionConfig(pred_loss_type="normalized_l1", inv_weight=0.2)
        model = CrossConditionJEPA(cfg)
        model.train()

        torch.manual_seed(0)
        out1 = model(x_a=self.x_a, x_b=self.x_b, ref_b=self.ref_b,
                     scan_id=self.scan_id, inv_scale=1.0)
        torch.manual_seed(0)
        out2 = model(x_a=self.x_a, x_b=self.x_b, ref_b=self.ref_b,
                     scan_id=self.scan_id, inv_scale=1.0)

        for key in ("total_loss", "pred_loss", "aux_loss", "inv_keep_frac"):
            self.assertAlmostEqual(out1[key].item(), out2[key].item(), places=6,
                                   msg=f"{key} is not deterministic under a fixed seed")


class TestTokenizerCanonical(unittest.TestCase):
    """Canonical energy-adaptive tokenizer: C+5 input, polarity-aligned, deterministic."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.C = 3, 128
        self.x = torch.randn(self.B, 5, 5, self.C)

    def test_time_branch_input_width(self):
        cfg = CrossConditionConfig()
        model = CrossConditionJEPA(cfg)
        self.assertEqual(model.tokenizer.time_proj.in_features, self.C + 5)
        self.assertFalse(model.tokenizer.global_bg_energy_anchor)

    def test_patch_polarity_invariance(self):
        """Flipping the whole patch polarity must not change the tokens (TMR/Hall fix)."""
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        with torch.no_grad():
            tok_pos, pos_pos = model.tokenizer(self.x)
            tok_neg, pos_neg = model.tokenizer(-self.x)
        self.assertTrue(torch.allclose(tok_pos, tok_neg, atol=1e-5),
                        "polarity flip changed the token (patch_polarity_align broken)")
        self.assertTrue(torch.allclose(pos_pos, pos_neg, atol=1e-6))

    def test_tokenizer_is_deterministic(self):
        model = CrossConditionJEPA(CrossConditionConfig()).eval()
        with torch.no_grad():
            t1, p1 = model.tokenizer(self.x)
            t2, p2 = model.tokenizer(self.x)
        self.assertTrue(torch.equal(t1, t2))
        self.assertTrue(torch.equal(p1, p2))


class TestExp70StrictLoad(unittest.TestCase):
    """The canonical cleanup must keep exp70 loadable with strict=True."""

    def test_exp70_checkpoint_loads_strict(self):
        if not os.path.exists(EXP70_CKPT):
            self.skipTest("exp70 checkpoint not present locally (experiments/ is gitignored)")

        ckpt = torch.load(EXP70_CKPT, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = CrossConditionJEPA(cfg)
        model.load_state_dict(ckpt["model_state_dict"])  # strict=True

        self.assertIsNone(model.inv_head)
        self.assertFalse(model.tokenizer.global_bg_energy_anchor)
        self.assertEqual(model.tokenizer.time_proj.in_features, 128 + 5)
        self.assertEqual(model.spec_cond_proj[0].in_features, 24)
        self.assertEqual(cfg.condition_modulation, "adaln")
        self.assertEqual(cfg.pred_loss_type, "normalized_l1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
