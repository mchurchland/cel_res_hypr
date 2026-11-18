import unittest
import numpy as np

from util.crossover_analysis import bootstrap_panel, curve_crossings, summarize_crossing
from util.hyperparameter_variability import factorial_variability


class TestCrossoverAnalysis(unittest.TestCase):
    def test_bracketed_linear_root(self):
        root = curve_crossings([20, 24, 28], [3, 1, -1])[0]
        self.assertEqual(root["root"], 26)
        self.assertEqual(root["bracket_low"], 24)
        self.assertEqual(root["bracket_high"], 28)
        self.assertEqual(root["direction"], "CE_to_ER")

    def test_no_extrapolation_constant_touch_and_multiple_crossings(self):
        for curve in [[1, 2, 3], [-1, -2, -3], [0, 0, 0], [1, 0, 1]]:
            self.assertEqual(curve_crossings([20, 24, 28], curve), [])
        self.assertEqual(curve_crossings([20, 24, 28], [1, 0, -1])[0]["root"], 24)
        self.assertEqual(len(curve_crossings([20, 24, 28], [1, -1, 1])), 2)

    def test_reject_invalid_curves(self):
        for x, y in [([20, 20], [1, -1]), ([20, 24], [1, np.nan]), ([20], [1])]:
            with self.assertRaises(ValueError):
                curve_crossings(x, y)

    def test_bootstrap_keeps_pairs_and_entire_curves(self):
        x = np.arange(20).reshape(10, 2).astype(float)
        panel = np.stack([x, x + 7], axis=1)
        mean, draws = bootstrap_panel(panel, n_boot=100, seed=6)
        np.testing.assert_allclose(mean[1] - mean[0], 7)
        np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 7)
        _, again = bootstrap_panel(panel, n_boot=100, seed=6)
        np.testing.assert_array_equal(draws, again)

    def test_missing_and_multiple_roots_are_reported(self):
        result = summarize_crossing([20, 24, 28], [1, .5, -1],
                                    [[1, .5, -1], [1, 1, 1], [1, -1, 1]])
        self.assertAlmostEqual(result["bootstrap_unique_fraction"], 1 / 3)
        self.assertAlmostEqual(result["bootstrap_no_crossing_fraction"], 1 / 3)
        self.assertAlmostEqual(result["bootstrap_multiple_fraction"], 1 / 3)

    def test_three_parameter_factorial_conserves_variance(self):
        a, b, c = np.meshgrid([-1., 1.], [-1., 1.], [-1., 1.], indexing="ij")
        result = factorial_variability((a + 2*b + 3*c).ravel(), [2, 2, 2])
        np.testing.assert_allclose(result["allocated_variance"], [[1, 4, 9]])
        pure = factorial_variability((a*b).ravel(), [2, 2, 2])
        np.testing.assert_allclose(pure["allocated_fraction"], [[.5, .5, 0]])
        self.assertEqual(len(pure["subsets"]), 7)


if __name__ == "__main__":
    unittest.main()
