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


class TestComplementarySpatiotemporalMasker5x5(unittest.TestCase):

    def setUp(self):
        from ..masking.cluster_mask import ComplementarySpatiotemporalMasker5x5, build_masker_5x5
        from ..configs.config import Spatiotemporal5x5Config
        self.masker = ComplementarySpatiotemporalMasker5x5(
            grid_size=5,
            num_temporal_stages=4,
            num_spatial_cluster=8,
            mode="causal",
        )
        self.config = Spatiotemporal5x5Config()
        self.build_masker = build_masker_5x5

    def test_default_factory(self):
        """Test that build_masker_5x5 defaults to ComplementarySpatiotemporalMasker5x5."""
        from ..masking.cluster_mask import ComplementarySpatiotemporalMasker5x5
        masker = self.build_masker(self.config)
        self.assertIsInstance(masker, ComplementarySpatiotemporalMasker5x5)

    def test_cst_mask_shapes_and_disjointness(self):
        """Test tensor dimensions, total token counts, and complete disjointness."""
        for seed in range(20):
            ctx, tgt, mask_bool = self.masker.sample_mask(batch_size=8, seed=seed)
            self.assertEqual(ctx.shape, (8, 34))
            self.assertEqual(tgt.shape, (8, 16))
            self.assertEqual(mask_bool.shape, (8, 100))

            for b in range(8):
                ctx_set = set(ctx[b].tolist())
                tgt_set = set(tgt[b].tolist())
                self.assertEqual(len(ctx_set & tgt_set), 0, "Context and target tokens must be strictly disjoint!")

    def test_causal_diffusion_stages(self):
        """Test that context tokens contain ONLY early stages (0, 1) and target tokens contain ONLY late stages (2, 3)."""
        for seed in range(20):
            ctx, tgt, _ = self.masker.sample_mask(batch_size=4, seed=seed)
            for b in range(4):
                for tok in ctx[b].tolist():
                    stage = tok % 4
                    self.assertIn(stage, (0, 1), f"Context token {tok} has invalid late stage {stage}!")
                for tok in tgt[b].tolist():
                    stage = tok % 4
                    self.assertIn(stage, (2, 3), f"Target token {tok} has invalid early stage {stage}!")

    def test_spatial_probes_partition(self):
        """Test that target probes form 8 unique spatial locations and context probes form 17 unique spatial locations."""
        for seed in range(20):
            ctx, tgt, _ = self.masker.sample_mask(batch_size=4, seed=seed)
            for b in range(4):
                tgt_probes = set(tok // 4 for tok in tgt[b].tolist())
                ctx_probes = set(tok // 4 for tok in ctx[b].tolist())
                self.assertEqual(len(tgt_probes), 8, f"Expected 8 spatial target probes, got {len(tgt_probes)}")
                self.assertEqual(len(ctx_probes), 17, f"Expected 17 spatial context probes, got {len(ctx_probes)}")
                self.assertEqual(len(tgt_probes & ctx_probes), 0, "Target and context spatial probes must be disjoint!")

    def test_mask_bank_sampling(self):
        """Test that mask bank sampling (seed=None) produces disjoint, valid tensor shapes."""
        ctx, tgt, mask_bool = self.masker.sample_mask(batch_size=16)
        self.assertEqual(ctx.shape, (16, 34))
        self.assertEqual(tgt.shape, (16, 16))
        self.assertEqual(mask_bool.shape, (16, 100))
        for b in range(16):
            ctx_set = set(ctx[b].tolist())
            tgt_set = set(tgt[b].tolist())
            self.assertEqual(len(ctx_set & tgt_set), 0, "Bank mask context and target must be strictly disjoint!")


if __name__ == "__main__":
    unittest.main()
