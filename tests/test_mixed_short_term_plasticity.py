import unittest
import numpy as np
from network_stats.short_term_plasticity import PRESETS,run_stp_states
from network_stats.mixed_short_term_plasticity import mixed_assignment,run_mixed_stp_states


class TestMixedSTP(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.w=(rng.normal(size=(6,6))*.12).astype(np.float32)
        self.b=(rng.normal(size=(6,1))*.1).astype(np.float32)
        self.signal=rng.uniform(-1,1,(150,1)).astype(np.float32)

    def test_uniform_types_reproduce_original_backend_exactly(self):
        for mode,flag in (("facilitating",True),("depressing",False)):
            mask=np.full(6,flag,dtype=bool)
            mixed=run_mixed_stp_states(self.w,self.b,self.signal,mask,10)
            original=run_stp_states(self.w,self.b,self.signal,PRESETS[mode],10)
            np.testing.assert_array_equal(mixed[0],original[0])
            np.testing.assert_array_equal(mixed[2],original[2]);np.testing.assert_array_equal(mixed[3],original[3])
            for field,value in original[1].items():self.assertEqual(mixed[1][field],value)

    def test_balanced_assignment_is_reproducible(self):
        a=mixed_assignment(297,9)
        self.assertEqual(int(a.sum()),148)
        np.testing.assert_array_equal(a,mixed_assignment(297,9))
        self.assertFalse(np.array_equal(a,mixed_assignment(297,10)))

    def test_mixed_zero_input_and_no_mutation(self):
        original=self.w.copy();mask=mixed_assignment(6,10)
        a=run_mixed_stp_states(self.w,self.b,np.zeros_like(self.signal),mask,10)
        np.testing.assert_array_equal(a[0],0)
        self.assertEqual(a[1]['gain_mean'],1.)
        self.assertEqual(a[1]['gain_facilitating_mean'],1.)
        self.assertEqual(a[1]['gain_depressing_mean'],1.)
        np.testing.assert_array_equal(a[2],a[4]);np.testing.assert_array_equal(a[3],1.)
        np.testing.assert_array_equal(original,self.w)

    def test_mixed_states_bounded_and_reset(self):
        mask=mixed_assignment(6,10)
        a=run_mixed_stp_states(self.w,self.b,self.signal,mask,10)
        b=run_mixed_stp_states(self.w,self.b,self.signal,mask,10)
        np.testing.assert_array_equal(a[0],b[0])
        self.assertGreaterEqual(a[1]['stp_state_min'],0.)
        self.assertLessEqual(a[1]['stp_state_max'],1.)
        self.assertTrue(np.isfinite(a[0]).all())


if __name__=='__main__':unittest.main()
