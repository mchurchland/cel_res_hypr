"""Synthetic checks for the scientific estimands and experimental design."""
import itertools
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from util.robustness import (PARAMETERS, performance_metrics, threshold_attainment,
    grid_edges, local_sensitivity, variance_decomposition, feasible_variance,
    spearman, cluster_spearman, bootstrap_mean)


class TestRobustness(unittest.TestCase):
    def test_constant_positive(self):
        r=performance_metrics(np.full(96,10.),[("threshold",10)])
        for name in ("sd","cv","iqr","mad","range","variance"):
            self.assertEqual(r[name],0)
        for name in ("mean","minimum","maximum","p10","bottom10"):
            self.assertEqual(r[name],10)
        self.assertEqual(r["retention80"],1)
        self.assertEqual(r["retention90"],1)
        self.assertEqual(r["threshold"],1)

    def test_all_metric_definitions(self):
        x=np.arange(1,101,dtype=float)
        r=performance_metrics(x)
        expected=dict(mean=50.5,sd=np.sqrt(833.25),cv=np.sqrt(833.25)/50.5,
            iqr=49.5,mad=25,minimum=1,p10=10.9,bottom10=5.5,maximum=100,
            retention80=.21,retention90=.11,range=99,variance=833.25)
        for k,v in expected.items():
            self.assertAlmostEqual(r[k],v,places=10,msg=k)
        # Noninteger decile convention: explicitly take 10/96 configurations.
        self.assertEqual(performance_metrics(np.arange(96.))["bottom10"],4.5)

    def test_zero_mean_and_negative_values(self):
        r=performance_metrics(np.zeros(96))
        self.assertTrue(np.isnan(r["cv"]))
        self.assertTrue(np.isnan(r["retention90"]))
        self.assertEqual(r["sd"],0)
        self.assertEqual(threshold_attainment([0,0],[0])[0],1)
        r=performance_metrics([-1,1])
        self.assertTrue(np.isnan(r["cv"]))
        self.assertEqual(r["sd"],1)
        self.assertEqual(r["n_negative"],1)
        r=performance_metrics([-3,-2,-1])
        self.assertTrue(np.isnan(r["retention80"]))
        self.assertEqual(r["minimum"],-3)

    def test_missing_and_nonfinite_values(self):
        r=performance_metrics([1,2,np.nan,np.inf,-np.inf])
        self.assertEqual(r["n_valid"],2)
        self.assertEqual(r["n_nonfinite"],3)
        self.assertEqual(r["mean"],1.5)
        r=performance_metrics([np.nan])
        self.assertEqual(r["n_valid"],0)
        self.assertTrue(np.isnan(r["mean"]))
        self.assertTrue(np.isnan(threshold_attainment([], [1])[0]))
        with self.assertRaises(ValueError):
            performance_metrics([1],[("bad",np.nan)])

    def test_floor_does_not_imply_low_cv(self):
        floor=performance_metrics([0,0,0,1])
        high=performance_metrics([9,9,9,10])
        self.assertEqual(floor["sd"],high["sd"])
        self.assertGreater(floor["cv"],high["cv"])
        self.assertEqual(floor["bottom10"],0)

    def test_dispersion_and_reliable_capability_can_disagree(self):
        poor=performance_metrics([1,1,1,1],[("threshold",5)])
        good=performance_metrics([8,9,10,11],[("threshold",5)])
        self.assertLess(poor["sd"],good["sd"])
        self.assertEqual(poor["retention90"],1)
        self.assertGreater(good["bottom10"],poor["bottom10"])
        self.assertGreater(good["threshold"],poor["threshold"])
        wild=performance_metrics([0,0,100,100])
        self.assertEqual(wild["sd"],50)
        self.assertEqual(wild["mad"],50)
        self.assertEqual(wild["iqr"],100)

    def test_threshold_curve(self):
        np.testing.assert_allclose(threshold_attainment([0,1,2,3,np.nan],[-1,0,1,2,3,4]),[1,1,.75,.5,.25,0])

    def test_threshold_endpoints_survive_csv_and_cache(self):
        from tools.reevaluate_robustness import threshold_settings, write_csv
        frame=pd.DataFrame({metric:[1.,2.,3.] for metric in ("MC","IPC","KR")})
        thresholds=threshold_settings({"main":frame})["MC"]["curve"]
        expected=pd.DataFrame(dict(threshold=thresholds,attainment=threshold_attainment([1,2,3],thresholds)))
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("curve.csv","curve.csv.gz"):
                path=Path(tmp)/name
                write_csv(expected,path)
                actual=pd.read_csv(path)
                self.assertFalse(actual.threshold.duplicated().any())
                self.assertEqual(actual.attainment.iloc[-1],0)
                self.assertGreater(actual.threshold.iloc[-1],3)

    def grid(self):
        return pd.DataFrame(itertools.product([1.,2.,4.],[.5,1.],[.1,1.],[0.,.1]),columns=PARAMETERS)

    def test_neighbor_sensitivity_and_coordinates(self):
        grid=self.grid()
        # Linear in normalized log-radius: slope exactly 10 in that dimension.
        values=10*np.log(grid.rho_target)/np.log(4)
        edges=grid_edges(grid)
        r=local_sensitivity(values.to_numpy(),edges)
        self.assertAlmostEqual(r["sensitivity_rho_target_mean"],10)
        self.assertAlmostEqual(r["sensitivity_rho_target_p90"],10)
        self.assertEqual(r["edges_rho_target"],16)
        for p in PARAMETERS[1:]:
            self.assertEqual(r[f"sensitivity_{p}_mean"],0)
        values.iloc[0]=np.nan
        self.assertEqual(local_sensitivity(values.to_numpy(),edges)["edges_rho_target"],15)
        with self.assertRaises(ValueError):
            grid_edges(pd.concat([grid,grid.iloc[:1]]))
        grid.loc[0,"rho_target"]=0
        with self.assertRaises(ValueError):
            grid_edges(grid)

    def test_missing_configuration_is_not_bridged(self):
        full=self.grid()
        edges=grid_edges(full)
        values=np.log(full.rho_target.to_numpy())
        # A missing center point at radius 2 excludes both radius edges.
        values[8]=np.nan
        self.assertEqual(local_sensitivity(values,edges)["edges_rho_target"],14)

    def test_variance_decomposition(self):
        y=np.array([[0.,2.],[2.,4.]])
        r=variance_decomposition(y)
        self.assertEqual(r["total_variance"],2)
        self.assertEqual(r["configuration_variance"],1)
        self.assertEqual(r["seed_variance"],1)
        self.assertEqual(r["interaction_noise_variance"],0)
        y=np.array([[0.,2.],[2.,0.]])
        r=variance_decomposition(y)
        self.assertEqual(r["interaction_noise_variance"],1)
        self.assertEqual(r["configuration_variance"],0)
        self.assertEqual(r["seed_variance"],0)
        rng=np.random.default_rng(4)
        r=variance_decomposition(rng.normal(size=(20,96)))
        self.assertAlmostEqual(r["total_variance"],sum(r[k] for k in ("configuration_variance","seed_variance","interaction_noise_variance")))
        self.assertAlmostEqual(r["within_configuration_variance"],r["seed_variance"]+r["interaction_noise_variance"])
        with self.assertRaises(ValueError):
            variance_decomposition([[0,np.nan],[2,3]])

    def test_feasible_variance(self):
        self.assertEqual(feasible_variance(0,0,10),0)
        self.assertEqual(feasible_variance(10,0,10),0)
        self.assertEqual(feasible_variance(5,0,10),25)
        x=np.r_[np.zeros(25),np.ones(75)*10]
        self.assertEqual(x.var(),feasible_variance(x.mean(),0,10))
        self.assertTrue(np.isnan(feasible_variance(-1,0,10)))

    def test_seed_cluster_bootstrap(self):
        d=pd.DataFrame(dict(seed=np.repeat(np.arange(8),2),x=np.repeat(np.arange(8),2),y=np.repeat(np.arange(8),2)))
        r=cluster_spearman(d,"x","y",n_boot=100)
        self.assertEqual(r["n_clusters"],8)
        self.assertEqual(r["n"],16)
        self.assertAlmostEqual(r["rho"],1)
        self.assertAlmostEqual(r["ci_low"],1)
        self.assertTrue(np.isnan(spearman([0,0,0],[1,2,3])))
        # Correlated columns stay paired during resampling.
        mean,ci=bootstrap_mean(np.c_[np.arange(8),2*np.arange(8)],100)
        np.testing.assert_allclose(ci[:,1],2*ci[:,0])
        np.testing.assert_allclose(mean,[3.5,7])


if __name__ == "__main__":
    unittest.main()
