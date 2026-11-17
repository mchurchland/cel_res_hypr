import unittest
import numpy as np
from util.hyperparameter_variability import factorial_variability


class TestFactorialVariability(unittest.TestCase):
    def grid(self):
        return np.meshgrid(*[[-1.,1.]]*4,indexing="ij")

    def test_constant_and_invalid(self):
        for c in [0.,3.14,-2.]:
            r=factorial_variability(np.full(16,c),(2,2,2,2))
            self.assertEqual(r['total_variance'][0],0)
            self.assertTrue(np.isnan(r['allocated_fraction']).all())
            self.assertEqual(r['matched_sweep_rms_sd'].sum(),0)
        for x in [np.zeros(15),np.r_[np.zeros(15),np.nan]]:
            with self.assertRaises(ValueError):
                factorial_variability(x,(2,2,2,2))

    def test_one_parameter(self):
        a,b,c,d=self.grid()
        r=factorial_variability((5+3*a).ravel(),(2,2,2,2))
        np.testing.assert_allclose(r['main_fraction'],[[1,0,0,0]],atol=1e-12)
        np.testing.assert_allclose(r['total_fraction'],[[1,0,0,0]],atol=1e-12)
        np.testing.assert_allclose(r['matched_sweep_rms_sd'],[[3,0,0,0]],atol=1e-12)
        np.testing.assert_allclose(r['mean_matched_sweep_range'],[[6,0,0,0]])

    def test_additive(self):
        a,b,c,d=self.grid()
        r=factorial_variability((a+2*b+3*c+4*d).ravel(),(2,2,2,2))
        np.testing.assert_allclose(r['main_variance'],[[1,4,9,16]])
        np.testing.assert_allclose(r['total_fraction'],r['main_fraction'])
        self.assertAlmostEqual(r['interaction_fraction'][0],0)
        self.assertAlmostEqual(r['allocated_fraction'].sum(),1)

    def test_pure_interaction(self):
        a,b,c,d=self.grid()
        r=factorial_variability((2*a*b).ravel(),(2,2,2,2))
        np.testing.assert_allclose(r['main_variance'],0,atol=1e-12)
        np.testing.assert_allclose(r['total_fraction'],[[1,1,0,0]])
        np.testing.assert_allclose(r['allocated_fraction'],[[.5,.5,0,0]])
        self.assertEqual(r['interaction_fraction'][0],1)

    def test_mixed_and_unequal_levels(self):
        rng=np.random.default_rng(45)
        x=rng.normal(size=(7,96))
        r=factorial_variability(x,(4,3,4,2))
        np.testing.assert_allclose(r['component_variance'].sum(axis=1),x.var(axis=1))
        np.testing.assert_allclose(r['allocated_fraction'].sum(axis=1),1)
        self.assertTrue((r['total_fraction']>=r['main_fraction']-1e-12).all())
        self.assertTrue((r['total_fraction']<=1+1e-12).all())
        # E_other[Var_i(P|other)] equals the sum of interactions containing i.
        tensor=x.reshape(7,4,3,4,2)
        for i in range(4):
            conditional_var=tensor.var(axis=i+1)
            expected=conditional_var.mean(axis=(1,2,3))
            np.testing.assert_allclose(r['total_involving_variance'][:,i],expected)


if __name__=='__main__':
    unittest.main()
