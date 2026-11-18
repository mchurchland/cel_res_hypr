import unittest

import numpy as np
from scipy.linalg import block_diag

from util.spectral_intervention import (
    attenuate_blocks, real_schur_blocks, secondary_blocks, selected_norm,
)


class TestSpectralIntervention(unittest.TestCase):
    def setUp(self):
        # Nonnormal, real matrix with a complex pair and two real eigenvalues.
        self.a = block_diag([[.95]], [[.5, -.3], [.3, .5]], [[.1]])
        self.a[0, 1:] = [1.2, -.7, .4]
        self.a[1, 3] = .6
        self.t, self.z, self.blocks = real_schur_blocks(self.a)

    def test_identity_is_exact(self):
        chosen = secondary_blocks(self.blocks, 2)
        np.testing.assert_array_equal(attenuate_blocks(self.a, self.t, self.z, self.blocks, chosen, 1), self.a)

    def test_shrink_pair_keeps_leader_and_other_eigenvalue(self):
        chosen = secondary_blocks(self.blocks, 2)
        self.assertEqual(sum(self.blocks[j].size for j in chosen), 2)
        changed = attenuate_blocks(self.a, self.t, self.z, self.blocks, chosen, .5)
        expected = np.array([.95, .25 + .15j, .25 - .15j, .1])
        np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(changed)), np.sort_complex(expected), atol=1e-12)
        np.testing.assert_allclose(self.z.T @ changed @ self.z - self.t,
                                   block_diag([[0]], [[-.25, .15], [-.15, -.25]], [[0]]), atol=1e-12)

    def test_zeroing_and_exact_frobenius_budget(self):
        chosen = secondary_blocks(self.blocks, 2)
        for multiplier in (0, .25, .75):
            changed = attenuate_blocks(self.a, self.t, self.z, self.blocks, chosen, multiplier)
            self.assertAlmostEqual(np.linalg.norm(changed-self.a, "fro"),
                                   (1-multiplier)*selected_norm(self.blocks, chosen))
            self.assertAlmostEqual(np.max(np.abs(np.linalg.eigvals(changed))), .95)

    def test_random_and_smallest_match_block_dimensions_and_exclude_leader(self):
        a = block_diag([[.95]], [[.6, -.2], [.2, .6]], [[.3, -.1], [.1, .3]], [[.2]], [[.1]])
        _, _, blocks = real_schur_blocks(a)
        target = secondary_blocks(blocks, 3)
        for strategy in ("random", "smallest"):
            control = secondary_blocks(blocks, 3, strategy, np.random.default_rng(10), target)
            self.assertEqual(sorted(blocks[j].size for j in target), sorted(blocks[j].size for j in control))
            self.assertNotIn(max(range(len(blocks)), key=lambda j: blocks[j].modulus), control)

    def test_orthogonal_coordinate_invariance(self):
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))
        a = q @ self.a @ q.T
        t, z, blocks = real_schur_blocks(a)
        target = secondary_blocks(blocks, 2)
        changed = attenuate_blocks(a, t, z, blocks, target, .5)
        np.testing.assert_allclose(np.sort(np.abs(np.linalg.eigvals(changed))), [.1, np.hypot(.25,.15), np.hypot(.25,.15), .95], atol=1e-12)

    def test_invalid_inputs(self):
        for a in (np.array([[np.nan]]), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                real_schur_blocks(a)
        for multiplier in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                attenuate_blocks(self.a, self.t, self.z, self.blocks, [1], multiplier)


if __name__ == "__main__":
    unittest.main()
