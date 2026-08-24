"""
Mask Verification Tests for PECT-JEPA v0.3 (implement.md, Section 5.2).
Verifies:
1. Context Indices and Target Indices are strictly disjoint (Context ∩ Target = ∅).
2. Target tokens at each selected frame t form an exact solid rectangle (B_h x B_w) without holes.
3. Two consecutive forward passes produce dynamic, distinct mask coordinates.
"""

import numpy as np
import torch
import unittest
from ..masking.spatiotemporal_mask import DynamicSpatioTemporalBlockMasker


class TestMask(unittest.TestCase):
    def test_frame_by_frame_mask_properties(self):
        B_h, B_w = 4, 4
        K_frames = 8
        masker = DynamicSpatioTemporalBlockMasker(
            spatial_block_h=B_h,
            spatial_block_w=B_w,
            num_masked_frames=K_frames
        )

        grid_shape = (8, 8, 16)  # Standard crop grid (64x64 raw pixels)
        B = 2

        ctx1, tgt1, mask_3d_1 = masker.sample_mask(B, grid_shape)
        ctx2, tgt2, mask_3d_2 = masker.sample_mask(B, grid_shape)

        # 1. Disjointness check: Context ∩ Target = ∅
        for b in range(B):
            c_set = set(ctx1[b].tolist())
            t_set = set(tgt1[b].tolist())
            intersection = c_set.intersection(t_set)
            self.assertEqual(len(intersection), 0, f"Context and Target overlap: {len(intersection)} tokens!")
            # Union must cover all tokens
            total_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
            self.assertEqual(len(c_set) + len(t_set), total_tokens, "Context + Target count mismatch")

        # 2. Strict solid rectangle check at each masked frame
        mask_np = mask_3d_1[0].cpu().numpy()  # [H_t, W_t, T_c]
        masked_frame_count = 0
        for t in range(grid_shape[2]):
            frame_mask = mask_np[:, :, t]
            masked_count = np.sum(frame_mask)
            if masked_count > 0:
                masked_frame_count += 1
                # Must have exactly B_h * B_w tokens
                self.assertEqual(masked_count, B_h * B_w, f"Frame {t} masked count {masked_count} != {B_h * B_w}")
                
                # Check that the mask forms a contiguous solid bounding rectangle
                rows = np.any(frame_mask, axis=1)
                cols = np.any(frame_mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                self.assertEqual(rmax - rmin + 1, B_h, f"Height of masked block is not {B_h}")
                self.assertEqual(cmax - cmin + 1, B_w, f"Width of masked block is not {B_w}")
                
                # Check solid rectangle without holes
                block_area = frame_mask[rmin : rmax + 1, cmin : cmax + 1]
                self.assertTrue(np.all(block_area), f"Holes detected in masked block on frame {t}!")

        self.assertEqual(masked_frame_count, K_frames, f"Expected {K_frames} masked frames, got {masked_frame_count}")

        # 3. Dynamic randomness check: consecutive iterations produce different coordinates
        diff = torch.sum(torch.abs(mask_3d_1.float() - mask_3d_2.float())).item()
        self.assertGreater(diff, 0, "Mask is static across iterations instead of dynamically sampled")

        print(f"PASS: test_mask.py | Disjointness: OK | Solid Rectangle ({B_h}x{B_w} x {K_frames} frames): OK | Dynamic diff: {diff}")


if __name__ == "__main__":
    unittest.main()
