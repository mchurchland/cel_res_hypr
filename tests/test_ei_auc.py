import unittest
import numpy as np
from util.ei_auc import curve_areas, win_fraction_auc


class CurveAreaTests(unittest.TestCase):
    def test_constant_and_zero(self):
        signed, positive, negative = curve_areas([[2, 2, 2], [0, 0, 0]], [0, .2, 1])
        np.testing.assert_allclose(signed, [2, 0])
        np.testing.assert_allclose(positive, [2, 0])
        np.testing.assert_allclose(negative, [0, 0])

    def test_crossing_splits_segment(self):
        signed, positive, negative = curve_areas([1, -3], [0, 1])
        self.assertEqual(signed, -1)
        self.assertEqual(positive, .125)
        self.assertEqual(negative, 1.125)

    def test_signed_equals_positive_minus_negative(self):
        y = np.random.default_rng(12).normal(size=(100, 6))
        signed, positive, negative = curve_areas(y, np.linspace(0, 1, 6))
        np.testing.assert_allclose(signed, positive - negative, atol=1e-14)

    def test_bad_inputs(self):
        for values, x in [([0, np.nan], [0, 1]), ([0, 1], [1, 0]), ([0, 1], [0, 0]), ([1], [0, 1])]:
            with self.assertRaises(ValueError):
                curve_areas(values, x)

    def test_fraction_auc_and_ties_partition_area(self):
        y = np.array([[[1, 0, -1], [-1, 0, 1]], [[1, 0, -1], [-1, 0, 1]]])
        result = win_fraction_auc(y, [[2, 0], [0, 2]], [0, 1])
        for name in ["ce", "er", "tie"]:
            self.assertAlmostEqual(result[f"{name}_win_auc_percent"], 100 / 3)

    def test_fraction_auc_averages_seeds_before_indicator(self):
        y = np.array([[[10], [10]], [[-1], [-1]], [[-1], [-1]]])
        result = win_fraction_auc(y, [[1, 1, 1]], [0, 1])
        self.assertEqual(result["ce_win_auc_percent"], 100)
        self.assertEqual(result["er_win_auc_percent"], 0)
