"""
Mask Verification Tests for 1D Temporal PECT-JEPA (implement.md, Section 5.2).
Verifies:
1. Context Indices and Target Indices are strictly disjoint (Context ∩ Target = ∅).
2. 'late_decay' strategy preserves first 5 patches as context, masks last 11 patches as target.
3. 'random_patch' strategy dynamically samples random patches.
"""

import numpy as np
import torch
import unittest
from ..masking.temporal_mask import Dynamic1DBlockMasker


class TestMask1D(unittest.TestCase):
    def test_late_decay_mask(self):
        masker = Dynamic1DBlockMasker(
            num_patches=16,
            mask_strategy="late_decay",
            num_visible_early=5
        )
        B = 4
        ctx, tgt, mask_1d = masker.sample_mask(B)

        self.assertEqual(ctx.shape, (4, 5))
        self.assertEqual(tgt.shape, (4, 11))

        for b in range(B):
            c_set = set(ctx[b].tolist())
            t_set = set(tgt[b].tolist())
            self.assertEqual(len(c_set.intersection(t_set)), 0, "Context and Target overlap!")
            self.assertEqual(c_set, set(range(5)), "Context is not early 5 patches [0..4]")
            self.assertEqual(t_set, set(range(5, 16)), "Target is not late 11 patches [5..15]")

        print("PASS: test_mask_1d.py (late_decay) | Context: [0..4] (5), Target: [5..15] (11) | Disjoint: OK")

    def test_random_patch_mask(self):
        masker = Dynamic1DBlockMasker(
            num_patches=16,
            mask_strategy="random_patch",
            mask_ratio=0.70
        )
        B = 4
        ctx1, tgt1, mask1 = masker.sample_mask(B)
        ctx2, tgt2, mask2 = masker.sample_mask(B)

        for b in range(B):
            c_set = set(ctx1[b].tolist())
            t_set = set(tgt1[b].tolist())
            self.assertEqual(len(c_set.intersection(t_set)), 0, "Context and Target overlap!")
            self.assertEqual(len(c_set) + len(t_set), 16)

        # Dynamic check
        diff = torch.sum(torch.abs(mask1.float() - mask2.float())).item()
        self.assertGreater(diff, 0, "Random mask is static across sampling passes!")
        print(f"PASS: test_mask_1d.py (random_patch) | Disjoint: OK | Dynamic diff: {diff}")


if __name__ == "__main__":
    unittest.main()
