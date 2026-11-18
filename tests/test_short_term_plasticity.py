import unittest
import numpy as np
from scipy.integrate import solve_ivp

from network_stats.short_term_plasticity import STPParameters,PRESETS,stp_constant_rate,run_stp_states


class TestShortTermPlasticity(unittest.TestCase):
    def test_silence_is_recovered_rest_with_unit_gain(self):
        for p in PRESETS.values():
            a=stp_constant_rate(0.,100,p)
            np.testing.assert_allclose(a,np.tile([p.U,1.,1.],(100,1)),atol=1e-12)

    def test_rate_integration_against_scipy(self):
        for p in PRESETS.values():
            for rate in (.01,.2,1.):
                def rhs(t,y):
                    u,x=y
                    return [(p.U-u)/p.tau_f+p.U*(1-u)*rate,(1-x)/p.tau_d-u*x*rate]
                ref=solve_ivp(rhs,[0,100],[p.U,1.],t_eval=np.arange(1,101),rtol=1e-11,atol=1e-12)
                actual=stp_constant_rate(rate,100,p)
                np.testing.assert_allclose(actual[:,:2],ref.y.T,atol=2e-5,rtol=2e-5)
                self.assertTrue(np.all((actual[:,:2]>=0)&(actual[:,:2]<=1)))

    def test_presets_facilitate_or_depress_under_moderate_drive(self):
        self.assertGreater(stp_constant_rate(.05,300,PRESETS['facilitating'])[-1,2],1.)
        self.assertLess(stp_constant_rate(.05,300,PRESETS['depressing'])[-1,2],1.)

    def test_zero_input_zero_network_and_reset(self):
        w=np.zeros((3,3),np.float32);b=np.array([[1.],[-1.],[.5]],np.float32)
        signal=np.sin(np.arange(50)*.2).astype(np.float32)[:,None]
        actual,_,_,_=run_stp_states(w,b,signal,PRESETS['facilitating'],5)
        np.testing.assert_allclose(actual,np.tanh(signal@b.T),atol=1e-7)
        np.testing.assert_array_equal(actual,run_stp_states(w,b,signal,PRESETS['facilitating'],5)[0])
        np.testing.assert_array_equal(run_stp_states(w,b,np.zeros_like(signal),PRESETS['depressing'],5)[0],0)

    def test_substep_convergence_and_no_mutation(self):
        rng=np.random.default_rng(1)
        w=(rng.normal(size=(5,5))*.1).astype(np.float32);original=w.copy()
        b=(rng.normal(size=(5,1))*.1).astype(np.float32)
        signal=rng.uniform(-1,1,size=(200,1)).astype(np.float32)
        a=run_stp_states(w,b,signal,STPParameters(.15,75,5,2),20)[0]
        ref=run_stp_states(w,b,signal,STPParameters(.15,75,5,4),20)[0]
        np.testing.assert_allclose(a,ref,atol=2e-6,rtol=2e-5)
        np.testing.assert_array_equal(w,original)

    def test_invalid_parameters(self):
        for values in ((0,5,75),(.2,0,75),(.2,5,-1),(.2,np.nan,75)):
            with self.assertRaises(ValueError):STPParameters(*values)


if __name__=='__main__':unittest.main()
