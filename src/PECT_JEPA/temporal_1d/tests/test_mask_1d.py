"""
Masking Tests for 1D Temporal PECT-JEPA (Stage A3, multi-strategy).
Verifies:
- shapes & counts constant across strategies
- context/target disjoint
- all three strategies sampled; per-strategy semantics
- reproducibility with seed; invalid probability config raises
"""

import torch
import unittest
from ..masking.temporal_mask import MultiStrategy1DMasker


class TestMask1D(unittest.TestCase):
    def test_mask_shapes_disjoint_and_strategies(self):
        m = MultiStrategy1DMasker(num_patches=16, num_visible=5,
                                  strategy_probs={"late_decay": 0.4,
                                                  "random_patch": 0.3,
                                                  "head_from_tail": 0.3})
        ctx, tgt, mask, sid = m.sample_mask(batch_size=300, seed=0)
        self.assertEqual(ctx.shape, (300, 5))
        self.assertEqual(tgt.shape, (300, 11))
        self.assertEqual(mask.shape, (300, 16))
        for b in range(300):
            self.assertEqual(len(set(ctx[b].tolist()) & set(tgt[b].tolist())), 0)
            self.assertEqual(int(mask[b][tgt[b]].sum()), 11)
        # all three strategies must appear
        self.assertEqual(set(sid.tolist()), {0, 1, 2})

        # strategy semantics: late_decay ctx=[0..4], head_from_tail ctx=[11..15]
        for b in range(300):
            if sid[b] == 0:
                self.assertEqual(ctx[b].tolist(), [0, 1, 2, 3, 4])
            elif sid[b] == 2:
                self.assertEqual(ctx[b].tolist(), [11, 12, 13, 14, 15])

        # reproducibility
        ctx2, tgt2, _, sid2 = m.sample_mask(batch_size=300, seed=0)
        self.assertTrue(torch.equal(ctx, ctx2))
        self.assertTrue(torch.equal(tgt, tgt2))
        self.assertTrue(torch.equal(sid, sid2))

        print("PASS: test_mask_1d.py (multi-strategy) | Disjoint: OK | 3 strategies: OK | Seeded: OK")

    def test_invalid_probs_raise(self):
        with self.assertRaises(ValueError):
            MultiStrategy1DMasker(strategy_probs={"late_decay": 0.5})


if __name__ == "__main__":
    unittest.main()
