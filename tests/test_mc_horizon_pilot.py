"""Synthetic checks for horizon diagnostics and their finite-sample null."""
import unittest
import numpy as np
from tools.run_mc_horizon_pilot import horizon_diagnostics,permutation_null


class HorizonDiagnosticsTests(unittest.TestCase):
    def test_recall_at_last_measured_delay_does_not_pass(self):
        rows=horizon_diagnostics(np.full(100,.8),np.full((30,100),.002),[30,60,100])
        self.assertFalse(any(row['practical_pass'] for row in rows))
        self.assertEqual(rows[-1]['remaining_excess'],0)
        self.assertFalse(rows[-1]['trailing_near_null'])

    def test_null_level_curve_passes(self):
        rows=horizon_diagnostics(np.full(100,.002),np.full((30,100),.002),[30,60,100])
        self.assertTrue(all(row['practical_pass'] for row in rows))
        np.testing.assert_allclose([row['remaining_excess'] for row in rows],0,atol=1e-14)

    def test_delayed_decay_requires_a_later_horizon(self):
        curve=np.r_[np.full(70,.8),np.full(130,.002)]
        rows=horizon_diagnostics(curve,np.full((30,200),.002),[60,80,100,150,200])
        self.assertFalse(rows[0]['practical_pass'])
        self.assertFalse(rows[1]['practical_pass'])
        self.assertTrue(rows[2]['practical_pass'])

    def test_raw_noise_area_is_not_reported_as_signal(self):
        row=horizon_diagnostics(np.full(400,.002),np.full((30,400),.002),[100])[0]
        self.assertAlmostEqual(row['remaining_raw'],.6)
        self.assertAlmostEqual(row['remaining_excess'],0)

    def test_invalid_horizon_and_nonfinite_scores(self):
        with self.assertRaises(ValueError):horizon_diagnostics(np.zeros(100),np.zeros((30,100)),[101])
        with self.assertRaises(ValueError):horizon_diagnostics(np.full(100,np.nan),np.zeros((30,100)),[100])

    def test_target_permutation_null_is_seeded_and_breaks_perfect_recall(self):
        y=np.random.default_rng(7).normal(size=(500,3))
        a=permutation_null(y,y,199,55);b=permutation_null(y,y,199,55)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(199,3))
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(((a>=0)&(a<=1)).all())
        self.assertLess(a.mean(),.004)
        self.assertGreater(a.mean(),.001)


if __name__=='__main__':unittest.main()
