"""Check paired-grid reconstruction and interaction allocation against known cases."""
import unittest
import itertools
import numpy as np
import pandas as pd
from tools.analyze_mc200 import anova, validate_cube, interval
from tools.analyze_architecture_mc_narma import variance_components
from util.ei_auc import win_fraction_auc


class AnalysisTests(unittest.TestCase):
    def test_anova_agrees_with_existing_reference(self):
        rng=np.random.default_rng(4);values=rng.normal(size=(3,2,4))
        subsets,components,total=anova(values)
        rows=[dict(rho_target=a,leak=b,input_scale=c,score=values[a,b,c]) for a,b,c in itertools.product(range(3),range(2),range(4))]
        reference,ref_total=variance_components(pd.DataFrame(rows))
        np.testing.assert_allclose(components,[r['variance'] for r in reference],rtol=1e-12)
        self.assertAlmostEqual(total,ref_total)

    def test_constant_and_single_parameter_surfaces(self):
        _,parts,total=anova(np.ones((7,3,2,4)))
        np.testing.assert_array_equal(parts,0);np.testing.assert_array_equal(total,0)
        x=np.broadcast_to(np.arange(4),(3,2,4)).copy()
        subsets,parts,total=anova(x)
        self.assertAlmostEqual(parts[subsets.index((2,))],total)
        np.testing.assert_allclose(np.delete(parts,subsets.index((2,))),0,atol=1e-12)

    def test_interaction_is_retained(self):
        a=np.array([-1,1])[:,None,None];b=np.array([-1,1])[None,:,None]
        values=np.broadcast_to(a*b,(2,2,3))
        subsets,parts,total=anova(values)
        self.assertEqual(total,1);self.assertEqual(parts[subsets.index((0,1))],1)
        self.assertAlmostEqual(parts.sum(),1)

    def test_grid_reordering_and_missing_or_duplicate_points(self):
        c=dict(repeats=2,families=['a'],normalizations=['own'],fractions=[0.,1.],rhos=[.6,.9],leaks=[1.],inputs=[.1,.5])
        rows=[]
        for r,f,n,x,a,b,z in itertools.product(range(2),c['families'],c['normalizations'],c['fractions'],c['rhos'],c['leaks'],c['inputs']):
            rows.append(dict(repeat_id=r,family=f,normalization=n,negative_fraction=x,rho_target=a,leak=b,input_scale=z,score=r+10*x+a+z))
        d=pd.DataFrame(rows);cube=validate_cube(d.sample(frac=1,random_state=2),c,['score'])
        self.assertAlmostEqual(cube[1,0,0,1,3,0],1+10+.9+.5)
        with self.assertRaises(ValueError):validate_cube(d.iloc[:-1],c,['score'])
        with self.assertRaises(ValueError):validate_cube(pd.concat([d.iloc[:-1],d.iloc[:1]]),c,['score'])

    def test_wins_follow_mean_then_indicator_and_ties(self):
        difference=np.array([10.,-1.,-1.])[:,None,None]*np.ones((3,3,2))
        difference[:,:,1]=0
        counts=np.array([[3,0,0],[0,3,0],[0,0,3],[1,1,1]])
        result=win_fraction_auc(difference,counts,[0,.5,1])
        self.assertEqual(result['ce_win_auc_percent'],50)
        self.assertEqual(result['tie_win_auc_percent'],50)

    def test_paired_intervals_and_nonfinite_anova(self):
        d=np.ones((5,3))*2;weights=np.ones((20,5))/5
        mean,lo,hi=interval(d,weights);np.testing.assert_allclose(mean,lo);np.testing.assert_allclose(mean,hi)
        with self.assertRaises(ValueError):anova(np.full((3,2,4),np.nan))


if __name__=='__main__':unittest.main()
