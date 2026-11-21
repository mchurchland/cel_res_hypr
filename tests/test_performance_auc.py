import unittest
import numpy as np
from tools.plot_performance_auc import performance_aucs


class PerformanceAUC(unittest.TestCase):
    def fixture(self):
        c=dict(families=['ce_empirical'],fractions=[0.,.5,1.],normalizations=['own_radius'],rhos=[.6],leaks=[1.],inputs=[.1,1.5])
        cube=np.zeros((3,1,3,1,1,1,2,3));counts=np.array([[3,0,0],[0,3,0],[0,0,3],[1,1,1]])
        return c,cube,counts

    def test_performance_units_and_input_averaging(self):
        c,cube,counts=self.fixture();cube[...,0]=2.;cube[:,:,:,:,:,:,1,0]=6.;cube[...,1]=.25;cube[...,2]=-10.
        table,individual=performance_aucs(cube,c,counts);mc=table[table.metric=='MC'].set_index('input_scale')
        self.assertEqual(mc.loc['ALL','performance_auc'],4.);self.assertEqual(mc.loc['0.1','performance_auc'],2.)
        self.assertEqual(mc.loc['1.5','performance_auc'],6.)
        self.assertTrue((table[table.metric=='NARMA10_test_nmse'].performance_auc==.25).all())
        self.assertTrue((table[table.metric=='IPC'].performance_auc==-10).all())
        self.assertEqual(len(individual),27)

    def test_curve_integral_and_paired_bootstrap(self):
        c,cube,counts=self.fixture();cube[...,0]=np.array([0.,2.,0.])[None,None,:,None,None,None,None]
        table,_=performance_aucs(cube,c,counts)
        self.assertTrue((table[table.metric=='MC'].performance_auc==1).all())
        cube[...,0]=np.array([1.,2.,3.])[:,None,None,None,None,None,None]
        table,_=performance_aucs(cube,c,counts);row=table[(table.metric=='MC')&(table.input_scale=='ALL')].iloc[0]
        self.assertEqual(row.performance_auc,2.)
        np.testing.assert_allclose([row.ci_low,row.ci_high],np.quantile([1,2,3,2],[.025,.975]))

    def test_span_normalization_and_invalid_inputs(self):
        c,cube,counts=self.fixture();cube[:]=7.;c['fractions']=[.2,.4,.6]
        table,_=performance_aucs(cube,c,counts);np.testing.assert_allclose(table.performance_auc,7.)
        with self.assertRaises(ValueError):performance_aucs(cube,c,np.zeros((2,3)))
        cube[0,0,0,0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):performance_aucs(cube,c,counts)


if __name__=='__main__':unittest.main()
