"""Meaningful checks of MC200 checkpoint identity, pairing, and scoring protocol."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mc200_recompute as mc


class MC200Tests(unittest.TestCase):
    def test_aliases_keep_distinct_distributions_and_scaling(self):
        self.assertEqual(mc.canonical_spec(('er','empirical','own_radius')), ('architecture','er_signed_empirical','own_radius'))
        self.assertEqual(mc.canonical_spec(('er','gamma','frobenius_match')), ('er','gamma','shared_ce_scale'))
        self.assertEqual(mc.canonical_spec(('er','cap90','frobenius_match')), ('er','cap90','frobenius_match'))
        self.assertEqual(mc.canonical_spec(('er','binary','frobenius_match')), ('er','binary','frobenius_match'))

    def test_array_digest_covers_values_shapes_and_dtype(self):
        a={'x':np.ones((2,3))}
        self.assertNotEqual(mc.array_digest(a),mc.array_digest({'x':a['x'].reshape(3,2)}))
        self.assertNotEqual(mc.array_digest(a),mc.array_digest({'x':a['x'].astype('float32')}))
        self.assertNotEqual(mc.array_digest(a),mc.array_digest({'x':a['x']*2}))

    def test_partition_covers_each_group_once(self):
        m={'parents':{'architecture':{'config':{'fractions':[0,.1,.2]}}},'unique_specs':[['a','b','c'],['d','e','f']]}
        groups=7*6
        decoded=[mc.decode_group(m,i) for rank in range(40) for i in range(rank,groups,40)]
        self.assertEqual(len(decoded),len(set(decoded)))
        self.assertEqual(set(decoded),{(r,tuple(s),fi) for r in range(7) for s in m['unique_specs'] for fi in range(3)})

    def test_checkpoint_roundtrip_and_wrong_condition_rejected(self):
        m={'experiment_id':'test','parents':{'architecture':{'config':{'rhos':[.95],'leaks':[1.],'inputs':[.1]}}},'tasks':{'mc_max_delay':200}}
        arrays={'curves':np.full((1,200),.3),'alpha_index':np.zeros((1,200),dtype=np.uint8)}
        meta=dict(experiment_id='test',repeat=0,spec=['a','b','c'],fraction_index=2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.npz';mc.write_checkpoint(p,arrays,meta)
            got,_=mc.read_checkpoint(p,m,0,('a','b','c'),2)
            np.testing.assert_array_equal(got['curves'],arrays['curves'])
            with self.assertRaises(ValueError):mc.read_checkpoint(p,m,1,('a','b','c'),2)
            with self.assertRaises(FileExistsError):mc.write_checkpoint(p,arrays,meta)

    def test_later_targets_do_not_alter_early_readouts(self):
        rng=np.random.default_rng(22);x=rng.normal(size=(180,8));y=rng.normal(size=(180,12))
        a=mc.validated_ridge(x[:100],y[:100],x[100:140],y[100:140],x[140:],[1e-6,.1,1.])[0]
        b=mc.validated_ridge(x[:100],y[:100,:3],x[100:140],y[100:140,:3],x[140:],[1e-6,.1,1.])[0]
        np.testing.assert_allclose(a[:,:3],b,atol=1e-12,rtol=1e-12)

    def test_perfect_recall_constant_prediction_and_negative_correlation(self):
        y=np.random.default_rng(3).normal(size=(5000,3));p=y.copy();p[:,1]=1;p[:,2]*=-1
        np.testing.assert_allclose(mc.squared_correlations(y,p),[1,0,1],atol=1e-12)


if __name__=='__main__':unittest.main()
