import unittest
import numpy as np
import pandas as pd
from util.ei_win_fraction import PARAMETERS, pair_surfaces, bootstrap_counts, summarize_surface, surface_matrix


class WinFractionTests(unittest.TestCase):
    def test_strict_wins_ties_and_negative_scores(self):
        y = np.array([[2., 0., -1., .1], [2., 0., -3., .3]])
        counts = np.array([[2, 0], [0, 2], [1, 1]])
        result, means, _, _ = summarize_surface(y, counts)
        self.assertEqual(result["percent_ce_better"], 50)
        self.assertEqual(result["n_ties"], 1)
        self.assertEqual(result["n_er_better"], 1)
        self.assertEqual(result["ci_low"], 50)
        np.testing.assert_allclose(means, [2, 0, -2, .2])

    def test_mean_surface_differs_from_average_seed_wins(self):
        result, _, _, _ = summarize_surface(np.array([[10., 10.], [-1., -1.], [-1., -1.]]), np.array([[1, 1, 1]]))
        self.assertEqual(result["percent_ce_better"], 100)
        self.assertAlmostEqual(result["mean_seed_win_percent"], 100 / 3)

    def test_zero_variance_complete_ties(self):
        result, _, _, _ = summarize_surface(np.zeros((4, 6)), np.ones((3, 4), dtype=int))
        self.assertEqual(result["percent_ce_better"], 0)
        self.assertEqual(result["n_ties"], 6)
        self.assertEqual(result["ci_high"], 0)

    def test_nonfinite_and_bad_bootstrap_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaises(ValueError):
                summarize_surface([[value]], [[1]])
        with self.assertRaises(ValueError):
            summarize_surface([[1], [2]], [[1, 0]])

    def test_complete_seed_resampling(self):
        counts = bootstrap_counts(7, 20, np.random.default_rng(4))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(20, 7))

    def example(self):
        rows = []
        for seed in [1, 2]:
            for scale in [.1, .5]:
                rows.append(dict(normalization="own", negative_fraction=.2, seed=seed,
                                 rho_target=.95, leak=1, input_scale=scale, neuron_bias=0, MC=seed + scale))
        return pd.DataFrame(rows)

    def test_pairing_uses_grid_and_seed_not_row_order(self):
        ce = self.example()
        er = ce.iloc[::-1].copy()
        er["MC"] -= 1
        paired, audit = pair_surfaces(ce, er, ["MC"])
        np.testing.assert_allclose(paired.MC, 1)
        self.assertEqual(audit.n_rows.sum(), 4)
        matrix, grid = surface_matrix(paired, "MC", [1, 2])
        self.assertEqual(matrix.shape, (2, 2))

    def test_missing_and_duplicate_configuration_audited(self):
        ce = self.example()
        paired, audit = pair_surfaces(ce, ce.iloc[:-1], ["MC"])
        self.assertEqual(audit[audit._merge == "left_only"].n_rows.sum(), 1)
        with self.assertRaises(ValueError):
            surface_matrix(paired, "MC", [1, 2])
        with self.assertRaises(ValueError):
            pair_surfaces(pd.concat([ce, ce.iloc[:1]]), ce, ["MC"])


if __name__ == "__main__":
    unittest.main()
