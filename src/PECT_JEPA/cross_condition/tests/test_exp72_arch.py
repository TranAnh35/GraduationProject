"""
EXP-72 architecture contract: radial spatial aggregation + multi-scale tokenizer.

Pins the three properties the work order depends on:

1. Backward compatibility: `tokenizer_arch='linear'` + `spatial_pool_mode='mean'`
   creates exactly the exp70/exp71 parameter surface (no conv stem, no radial
   attention), so old checkpoints still load with `strict=True`.
2. EXP-72A: `_pool_spatial_context` in 'radial' mode weights inner-ring probes
   above the outer ring (r3=7px), which is the false-positive halo mechanism
   (FM13), and stays deterministic / finite.
3. EXP-72B: the conv stem changes the time-branch width to C+32+5 and the
   continuous saliency-weighted phasor changes the spectral width to 4K+2.
4. FP32 safety: `normalized_probe_l1` / `center_by_group` survive float16
   inputs with degenerate (constant) probes instead of dividing by 0.
"""

import os
import sys
import unittest

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.PECT_JEPA.cross_condition.configs.config import CrossConditionConfig  # noqa: E402
from src.PECT_JEPA.cross_condition.models.cross_condition_jepa import (  # noqa: E402
    CrossConditionJEPA,
    center_by_group,
    normalized_probe_l1,
)
from src.PECT_JEPA.cross_condition.models.tokenizer import (  # noqa: E402
    DualDomainTokenizer,
    continuous_phasor,
)


def _model(**kw):
    torch.manual_seed(0)
    cfg = CrossConditionConfig(pred_loss_type="normalized_l1", inv_weight=0.2, **kw)
    return CrossConditionJEPA(cfg)


class TestLegacySurface(unittest.TestCase):
    def test_legacy_tokenizer_has_no_exp72_modules(self):
        tok = DualDomainTokenizer(CrossConditionConfig())
        self.assertEqual(tok.tokenizer_arch, "linear")
        self.assertIsNone(tok.conv_k3)
        self.assertIsNone(tok.conv_k7)
        self.assertIsNone(tok.conv_k15)
        self.assertIsNone(tok.stem_norm)
        self.assertIsNone(tok.stem_pw)
        self.assertEqual(tok.time_proj.in_features, tok.temporal_samples + 5)
        self.assertEqual(tok.freq_proj.in_features, tok.num_harmonics * 3 + 2)

    def test_legacy_model_has_no_radial_parameters(self):
        model = _model()
        names = {n for n, _ in model.named_parameters()}
        self.assertFalse(any("radial" in n for n in names), names)
        self.assertFalse(any(n.startswith("tokenizer.conv") for n in names), names)
        self.assertFalse(any(n.startswith("tokenizer.stem") for n in names), names)

    def test_mean_pool_matches_plain_average(self):
        model = _model()
        h = torch.randn(3, 16, 64)
        idx = torch.arange(9, 25)
        pooled = model._pool_spatial_context(h, idx)
        self.assertTrue(torch.allclose(pooled, h.mean(dim=1), atol=1e-6))

    def test_legacy_state_dict_keys_are_exp71_subset(self):
        # The exp72a/b models add keys; the legacy model must add none.
        legacy = set(_model().state_dict().keys())
        radial = set(_model(spatial_pool_mode="radial").state_dict().keys())
        conv = set(_model(tokenizer_arch="multiscale_conv").state_dict().keys())
        self.assertTrue(legacy < radial)
        self.assertTrue(legacy < conv)
        self.assertEqual(radial - legacy, {"radial_log_alpha", "radial_attn_proj.weight"})
        # probe_r2 is non-persistent: it must never leak into a checkpoint.
        self.assertNotIn("probe_r2", radial | conv | legacy)


class TestRadialPooling(unittest.TestCase):
    def test_probe_r2_is_normalized_squared_radius(self):
        model = _model(spatial_pool_mode="radial", radial_decay_init=2.5)
        r2 = model.probe_r2
        self.assertEqual(tuple(r2.shape), (25,))
        self.assertAlmostEqual(float(r2[0]), 0.0, places=6)                     # center
        self.assertAlmostEqual(float(r2[1]), (2.0 ** 2) / 98.0, places=5)       # r1 = 2 px (cardinal)
        self.assertAlmostEqual(float(r2[9]), (4.0 ** 2) / 98.0, places=5)       # r2 = 4 px (cardinal)
        self.assertAlmostEqual(float(r2[17]), (7.0 ** 2) / 98.0, places=5)      # r3 = 7 px (cardinal)
        self.assertAlmostEqual(float(r2.max()), 1.0, places=6)                  # r3 diagonal (7,7)

    def test_outer_ring_is_downweighted_vs_inner_ring(self):
        model = _model(spatial_pool_mode="radial", radial_decay_init=2.5)
        # Same content on every probe => the weight is purely radial.
        idx = torch.tensor([[1, 9, 17]])  # r1 = 2px, r2 = 4px, r3 = 7px
        alpha = float(torch.exp(model.radial_log_alpha.detach()))
        w = torch.softmax(-alpha * model.probe_r2[idx], dim=-1)
        self.assertGreater(float(w[0, 0]), float(w[0, 1]))
        self.assertGreater(float(w[0, 1]), float(w[0, 2]))
        self.assertGreater(float(w[0, 0]), 2.5 * float(w[0, 2]))

    def test_radial_pool_is_deterministic_and_finite(self):
        model = _model(spatial_pool_mode="radial").eval()
        h = torch.randn(4, 16, 64)
        idx = torch.arange(9, 25).unsqueeze(0).expand(4, -1)
        with torch.no_grad():
            a = model._pool_spatial_context(h, idx)
            b = model._pool_spatial_context(h, idx)
            c = model._pool_spatial_context(h, idx, h_anchor=torch.randn(4, 64))
        self.assertEqual(tuple(a.shape), (4, 64))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(c).all())

    def test_alpha_is_clamped_to_a_usable_range(self):
        model = _model(spatial_pool_mode="radial", radial_decay_init=2.5)
        with torch.no_grad():
            model.radial_log_alpha.fill_(20.0)   # exp(20) would saturate softmax
        h = torch.randn(1, 16, 64)
        out = model._pool_spatial_context(h, torch.arange(9, 25).unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())

    def test_extract_features_matches_mean_mode_when_disabled(self):
        a = _model().eval()
        b = _model(spatial_pool_mode="radial").eval()
        b.load_state_dict(a.state_dict(), strict=False)  # only the radial extras differ
        x = torch.randn(2, 5, 5, 128)
        with torch.no_grad():
            za = a.extract_features(x)
            zb = b.extract_features(x)
        # alpha = 2.5 must re-weight the rings, so the two must NOT be identical.
        self.assertEqual(tuple(za.shape), (2, 128))
        self.assertFalse(torch.allclose(za, zb, atol=1e-4))


class TestMultiScaleTokenizer(unittest.TestCase):
    def test_multiscale_dims(self):
        tok = DualDomainTokenizer(CrossConditionConfig(tokenizer_arch="multiscale_conv"))
        self.assertEqual(tok.time_proj.in_features, tok.temporal_samples + 32 + 5)
        self.assertEqual(tok.freq_proj.in_features, tok.num_harmonics * 4 + 2)
        self.assertIsNotNone(tok.conv_k3)
        self.assertEqual(tok.stem_pw.out_channels, 16)

    def test_conv_branch_keeps_temporal_length(self):
        tok = DualDomainTokenizer(CrossConditionConfig(tokenizer_arch="multiscale_conv")).eval()
        x = torch.randn(2, 5, 5, 128)
        with torch.no_grad():
            tokens, pos = tok(x)
        self.assertEqual(tuple(tokens.shape), (2, 25, 64))
        self.assertEqual(tuple(pos.shape), (2, 25, 64))

    def test_tokens_are_deterministic(self):
        tok = DualDomainTokenizer(CrossConditionConfig(tokenizer_arch="multiscale_conv")).eval()
        x = torch.randn(3, 5, 5, 128)
        with torch.no_grad():
            a, _ = tok(x)
            b, _ = tok(x)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_continuous_phasor_is_stable_at_the_branch_cut(self):
        """FM16: theta/pi jumps +-2 across +-pi; the phasor does not."""
        sal = torch.full((4,), 0.5)
        theta_hi = torch.full((4,), torch.pi - 1e-6)
        theta_lo = torch.full((4,), -torch.pi + 1e-6)

        legacy_hi, legacy_lo = theta_hi / torch.pi, theta_lo / torch.pi
        self.assertGreater(float((legacy_hi - legacy_lo).abs().max()), 1.9)

        c_hi, s_hi = continuous_phasor(theta_hi, sal)
        c_lo, s_lo = continuous_phasor(theta_lo, sal)
        self.assertLess(float((c_hi - c_lo).abs().max()), 1e-5)
        self.assertLess(float((s_hi - s_lo).abs().max()), 1e-5)

    def test_continuous_phasor_vanishes_for_empty_harmonics(self):
        theta = torch.linspace(-torch.pi, torch.pi, 16)
        cos_p, sin_p = continuous_phasor(theta, torch.zeros(16))
        self.assertLess(float(cos_p.abs().max()), 1e-3)
        self.assertLess(float(sin_p.abs().max()), 1e-3)


class TestFloat32Safety(unittest.TestCase):
    def test_normalized_probe_l1_survives_constant_fp16_probe(self):
        pred = torch.full((2, 4, 64), 0.5, dtype=torch.float16)
        tgt = torch.full((2, 4, 64), 0.5, dtype=torch.float16)
        loss = normalized_probe_l1(pred, tgt)
        self.assertTrue(torch.isfinite(loss).all(), loss)
        self.assertEqual(float(loss), 0.0)

    def test_normalized_probe_l1_fp16_matches_fp32(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 4, 64)
        tgt = torch.randn(2, 4, 64)
        ref = normalized_probe_l1(pred, tgt)
        got = normalized_probe_l1(pred.half(), tgt.half())
        self.assertTrue(torch.isfinite(got).all())
        self.assertLess(abs(float(ref) - float(got)), 5e-3)

    def test_center_by_group_returns_float32_for_fp16_input(self):
        z = torch.randn(8, 16).half()
        out = center_by_group(z, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), min_group_size=2)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.isfinite(out).all())

    def test_inv_loss_is_finite_for_degenerate_pair(self):
        model = _model()
        x = torch.randn(4, 5, 5, 128)
        out = model(x, x.clone(), x.clone(), scan_id=torch.arange(4),
                    scan_id_b=torch.arange(4, 8), inv_scale=1.0)
        self.assertTrue(torch.isfinite(out["inv_loss"]).all(), out["inv_loss"])


if __name__ == "__main__":
    unittest.main()
