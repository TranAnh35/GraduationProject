"""
Mask and Information Flow Tests (Section 22.2).
Verifies dynamic mask sampling, context/target disjointness, and randomness across iterations.
"""

import torch
import unittest
from ..masking.spatiotemporal_mask import DynamicSpatioTemporalBlockMasker


class TestMask(unittest.TestCase):
    def test_dynamic_masking(self):
        """Test mask properties: randomness, disjointness, and non-empty visible context."""
        masker = DynamicSpatioTemporalBlockMasker(
            spatial_block_h=4,
            spatial_block_w=4,
            temporal_block_t=1,
            num_blocks=4,
            min_mask_ratio=0.15,
            max_mask_ratio=0.60
        )

        grid_shape = (8, 8, 7)  # H_t=8, W_t=8, K_t=7 -> 448 tokens
        B = 2

        ctx1, tgt1, mask_3d_1 = masker.sample_mask(B, grid_shape)
        ctx2, tgt2, mask_3d_2 = masker.sample_mask(B, grid_shape)

        # 1. Disjointness check
        for b in range(B):
            c_set = set(ctx1[b].tolist())
            t_set = set(tgt1[b].tolist())
            intersection = c_set.intersection(t_set)
            self.assertEqual(len(intersection), 0, f"Context and Target overlap: {intersection}")

        # 2. Non-empty check
        self.assertGreater(ctx1.shape[1], 0, "Context is empty")
        self.assertGreater(tgt1.shape[1], 0, "Target is empty")

        # 3. Mask ratio verification
        total_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
        ratio = tgt1.shape[1] / total_tokens
        self.assertGreaterEqual(ratio, 0.14, f"Mask ratio {ratio:.4f} is below min_mask_ratio (0.15)")
        self.assertLessEqual(ratio, 0.61, f"Mask ratio {ratio:.4f} is above max_mask_ratio (0.60)")

        # 4. Dynamic randomness check: two iterations should produce different masks
        diff = torch.sum(torch.abs(mask_3d_1.float() - mask_3d_2.float())).item()
        self.assertGreater(diff, 0, "Mask is static across iterations instead of dynamically sampled")

        print(f"PASS: Mask Test | Tokens: {grid_shape} ({total_tokens}) | N_ctx: {ctx1.shape[1]}, N_tgt: {tgt1.shape[1]} (Ratio: {ratio:.1%}) | Dynamic diff: {diff}")


if __name__ == "__main__":
    unittest.main()
