import unittest
import numpy as np
from tools.plot_architecture_auc import compute_auc


class AUC(unittest.TestCase):
    def fixture(self):
        c=dict(families=['ce_empirical','ws_binary'],fractions=[0.,.5,1.],normalizations=['own_radius'],rhos=[.6],leaks=[1.],inputs=[.1,1.5])
        cube=np.zeros((3,2,3,1,1,1,2,3))
        cube[:,0,...,0]=2.;cube[:,0,:,:,:,:,1,0]=0.;cube[:,1,...,0]=1.
        cube[:,0,...,1]=.5;cube[:,1,...,1]=1.
        counts=np.array([[3,0,0],[0,3,0],[0,0,3],[1,1,1]])
        return c,cube,counts

    def test_input_subsets_metric_direction_and_ties(self):
        c,cube,counts=self.fixture();auc,performance=compute_auc(cube,c,counts)
        mc=auc[auc.metric=='MC'].set_index('input_scale')
        self.assertEqual(mc.loc['ALL','ce_win_auc_percent'],50.)
        self.assertEqual(mc.loc['0.1','ce_win_auc_percent'],100.)
        self.assertEqual(mc.loc['1.5','ce_win_auc_percent'],0.)
        self.assertTrue((auc[auc.metric=='NARMA10_test_nmse'].ce_win_auc_percent==100).all())
        self.assertTrue((auc[auc.metric=='IPC'].tie_win_auc_percent==100).all())
        self.assertTrue((performance[performance.metric=='MC'].mean_performance_auc==1).all())

    def test_mean_before_win_indicator_and_nonzero_ci(self):
        c,cube,counts=self.fixture();cube[:,0,...,0]=np.array([10,1,1])[:,None,None,None,None,None];cube[:,1,...,0]=2.
        auc,_=compute_auc(cube,c,counts);row=auc[(auc.metric=='MC')&(auc.input_scale=='ALL')].iloc[0]
        self.assertEqual(row.ce_win_auc_percent,100.)
        self.assertEqual(row.ce_ci_low,0.);self.assertEqual(row.ce_ci_high,100.)

    def test_observed_span_normalization_and_missing_fraction(self):
        c,cube,counts=self.fixture();c['fractions']=[.2,.4,.6]
        auc,_=compute_auc(cube,c,counts)
        self.assertTrue(np.allclose(auc.ce_win_auc_percent+auc.control_win_auc_percent+auc.tie_win_auc_percent,100))
        with self.assertRaises(ValueError):compute_auc(cube[:,:,0:1],dict(c,fractions=[.2]),counts)


if __name__=='__main__':unittest.main()
