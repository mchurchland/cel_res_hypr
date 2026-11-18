import unittest
import numpy as np

from tools.animate_ei_network_modes import activity_modes, relative_edge_widths


class TestNetworkVideoDiagnostics(unittest.TestCase):
    def test_widths_preserve_magnitude_ratios_and_ignore_sign(self):
        weights = np.array([[0., -2.], [4., 1.]])
        widths = relative_edge_widths(weights, max_width=2.)
        np.testing.assert_allclose(widths, [[0, 1], [2, .5]])

    def test_independent_width_normalization_cancels_global_gain(self):
        weights = np.array([[0., -2.], [4., 1.]])
        np.testing.assert_allclose(relative_edge_widths(weights), relative_edge_widths(weights * .007))
        np.testing.assert_allclose(relative_edge_widths(weights), relative_edge_widths(-weights))
        np.testing.assert_array_equal(relative_edge_widths(np.zeros((2, 2))), 0.)

    def test_bad_width_inputs_fail(self):
        for weights, maximum in [(np.array([np.nan]), 2.), (np.array([1.]), 0.)]:
            with self.assertRaises(ValueError):
                relative_edge_widths(weights, maximum)

    def test_single_collective_activity_mode(self):
        signal = np.sin(np.arange(100) * .1)
        states = signal[:, None] * np.array([[1., 2., -3.]]) + 7
        fractions, dimension = activity_modes(states)
        self.assertAlmostEqual(fractions[0], 1.)
        self.assertAlmostEqual(dimension, 1.)

    def test_three_equal_variance_activity_modes(self):
        states = np.array(list(np.ndindex(2, 2, 2)), dtype=float) * 2 - 1
        fractions, dimension = activity_modes(states)
        np.testing.assert_allclose(fractions, [1/3, 1/3, 1/3])
        self.assertAlmostEqual(dimension, 3.)
        fractions, dimension = activity_modes(np.ones((20, 3)))
        np.testing.assert_array_equal(fractions, 0.)
        self.assertEqual(dimension, 0.)


if __name__ == "__main__":
    unittest.main()
