"""
Unit test for Contiguous Cluster Masker on 5x5 Grid with Hole-Filling.
"""

import unittest
from collections import deque
import torch

from ..masking.cluster_mask import ContiguousClusterMasker5x5


class TestContiguousClusterMasker5x5(unittest.TestCase):

    def setUp(self):
        self.masker = ContiguousClusterMasker5x5(min_masked=10, max_masked=15, grid_size=5)

    def test_mask_ratios(self):
        """Test that the number of masked points strictly falls in [10, 15] (40% - 60%)."""
        for seed in range(50):
            ctx, tgt, mask_bool = self.masker.sample_mask(batch_size=4, seed=seed)
            B, N_tgt = tgt.shape
            self.assertTrue(10 <= N_tgt <= 15, f"Expected 10-15 target tokens, got {N_tgt}")
            self.assertEqual(ctx.shape[1] + N_tgt, 25)
            self.assertEqual(mask_bool.shape, (4, 25))

    def test_connectivity(self):
        """Test that every target cluster forms a single connected component on the 5x5 grid."""
        for seed in range(50):
            _, tgt, _ = self.masker.sample_mask(batch_size=4, seed=seed)
            for b in range(4):
                tgt_pts = set(tgt[b].tolist())
                coords = [(idx // 5, idx % 5) for idx in tgt_pts]
                coord_set = set(coords)

                start = coords[0]
                visited = {start}
                queue = deque([start])

                while queue:
                    r, c = queue.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in coord_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

                self.assertEqual(len(visited), len(coord_set), "Masked cluster must be a single connected component")

    def test_no_isolated_context(self):
        """Test that Hole-Filling eliminates isolated context points (0 context neighbors)."""
        for seed in range(100):
            ctx, _, _ = self.masker.sample_mask(batch_size=4, seed=seed)
            for b in range(4):
                ctx_pts = set(ctx[b].tolist())
                coords = [(idx // 5, idx % 5) for idx in ctx_pts]
                coord_set = set(coords)

                for (r, c) in coords:
                    nb_ctx = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in coord_set:
                            nb_ctx.append((nr, nc))
                    self.assertGreaterEqual(
                        len(nb_ctx), 1,
                        f"Found isolated context point at ({r}, {c}) with 0 context neighbors!"
                    )


if __name__ == "__main__":
    unittest.main()
