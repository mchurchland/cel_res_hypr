"""Protocol, pairing, validation isolation and resumable-run integration tests."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from reservoirpy.datasets import narma

import mc_narma_crossover as runner
from network_stats.temporal_tasks import (
    TemporalTaskSettings, evaluate_temporal_tasks, fit_narma_readout, make_task_data,
)


def fixture_connectome(directory):
    known = np.zeros((6, 6), dtype=np.float32)
    unknown = np.zeros_like(known)
    for i in range(6):
        known[i, (i + 1) % 6] = -2 if i % 3 == 0 else 2
        unknown[i, (i + 2) % 6] = 3
    a, b = Path(directory) / "known.npy", Path(directory) / "unknown.npy"
    np.save(a, known)
    np.save(b, unknown)
    return a, b


class TaskProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.settings = TemporalTaskSettings(washout=20, train=50, validation=20,
                                             test=20, mc_max_delay=5, narma_burnin=100)

    def test_uses_package_targets_and_one_step_alignment(self):
        seed = 27
        d = make_task_data(self.settings, seed)
        length = 110
        u, y = narma(length + 100, order=10, a1=.3, a2=.05, b=1.5, c=.1,
                     seed=seed + 37_000_003)
        np.testing.assert_array_equal(d["narma_input"], u[9:-1][100:].astype(np.float32))
        np.testing.assert_array_equal(d["narma_target"], y[100:])
        self.assertEqual(tuple(d["narma_input"].shape), (length, 1))
        self.assertTrue(torch.all((d["narma_input"] >= 0) & (d["narma_input"] <= .5)))

    def test_reproducible_signals_without_global_rng_mutation(self):
        torch.manual_seed(5)
        state = torch.random.get_rng_state().clone()
        a = make_task_data(self.settings, 11)
        b = make_task_data(self.settings, 11)
        c = make_task_data(self.settings, 12)
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(a["mc_input"], c["mc_input"]))
        self.assertFalse(torch.equal(a["narma_input"], c["narma_input"]))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_settings_reject_invalid_lengths_and_penalties(self):
        for fields in [dict(train=30), dict(validation=0), dict(narma_burnin=-1),
                       dict(narma_ridge_alphas=()), dict(narma_ridge_alphas=(float("nan"),))]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                TemporalTaskSettings(**fields)

    def test_readout_learns_intercept_and_test_labels_cannot_select_alpha(self):
        gen = torch.Generator().manual_seed(17)
        x = torch.randn(150, 4, generator=gen, dtype=torch.float64)
        y = 3 + x @ torch.tensor([[1.], [-2.], [.5], [0.]], dtype=torch.float64)
        pieces = (x[:80], y[:80], x[80:110], y[80:110], x[110:], y[110:])
        a = fit_narma_readout(*pieces, (1e-8, .01, 1.))
        b = fit_narma_readout(*pieces[:-1], -y[110:], (1e-8, .01, 1.))
        self.assertLess(a["NARMA10_test_nmse"], 1e-12)
        self.assertEqual(a["NARMA10_ridge_alpha"], b["NARMA10_ridge_alpha"])
        self.assertEqual(a["NARMA10_val_nmse"], b["NARMA10_val_nmse"])
        self.assertGreater(b["NARMA10_test_nmse"], 1)
        self.assertAlmostEqual(a["NARMA10_test_nrmse"] ** 2, a["NARMA10_test_nmse"])

    def test_readout_rejects_constant_target_and_nonfinite_states(self):
        x = torch.eye(5)
        y = torch.ones(5, 1)
        with self.assertRaises(ValueError):
            fit_narma_readout(x, y, x, y, x, y, (.1,))
        bad = x.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            fit_narma_readout(bad, y, x, y, x, y, (.1,))

    def test_small_end_to_end_tasks_are_finite_and_skip_rank_metrics(self):
        data = make_task_data(self.settings, 123)
        w = torch.eye(6) * .6
        win = torch.arange(1, 7, dtype=torch.float32)[:, None] * .1
        with patch("network_stats.stats.compute_KR", side_effect=AssertionError("KR called")), \
             patch("network_stats.stats.compute_IPC", side_effect=AssertionError("IPC called")):
            scores = evaluate_temporal_tasks(w, win, .8, data, self.settings)
        self.assertTrue(all(np.isfinite(v) for v in scores.values()))
        self.assertTrue(0 <= scores["MC"] <= 5.001)
        self.assertNotIn("IPC", scores)
        self.assertNotIn("KR", scores)
        self.assertNotIn("GR", scores)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.known, self.unknown = fixture_connectome(self.root)

    def args(self, chunk=0):
        return ["--ce-adj", str(self.known), "--ce-unknown-sign-weights", str(self.unknown),
                "--out-dir", str(self.root / "run" / f"chunk_{chunk}"),
                "--n-repeats", "2", "--repeat-split", "2", "--repeat-rank", str(chunk),
                "--sign-fracs", ".2", ".24", "--rho-values", ".8", "--leak-values", ".8",
                "--input-scales", ".5", "--washout", "20", "--train-length", "50",
                "--validation-length", "20", "--test-length", "20", "--mc-max-delay", "5",
                "--narma-burnin", "100", "--narma-ridge-alphas", ".0001", ".01"]

    def run_quietly(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args)

    def test_repeat_partition_is_disjoint_and_complete(self):
        parts = [runner.partition_repeats(7, 3, r) for r in range(3)]
        self.assertEqual(sorted(sum(parts, [])), list(range(7)))
        with self.assertRaises(ValueError):
            runner.partition_repeats(7, 3, 3)

    def test_unknown_edges_restored_at_known_fraction(self):
        completed, fraction = runner.load_completed_connectome(self.known, self.unknown, 5)
        self.assertAlmostEqual(fraction, 1 / 3)
        known, unknown = np.load(self.known), np.load(self.unknown)
        np.testing.assert_array_equal(completed[known != 0], known[known != 0])
        np.testing.assert_array_equal(abs(completed[unknown != 0]), unknown[unknown != 0])
        self.assertEqual(np.count_nonzero(completed[unknown != 0] < 0), 2)

    def test_overlapping_input_matrices_fail_without_filtering(self):
        np.save(self.unknown, np.abs(np.load(self.known)))
        with self.assertRaises(ValueError):
            runner.load_completed_connectome(self.known, self.unknown, 5)

    def test_both_models_preserve_edges_and_nested_sign_pairing(self):
        completed, _ = runner.load_completed_connectome(self.known, self.unknown, 5)
        for family in runner.FAMILIES:
            a, win_a, info_a = runner.build_condition(completed, family, .2, "spectral_radius", 5, torch.device("cpu"))
            b, win_b, info_b = runner.build_condition(completed, family, .4, "spectral_radius", 5, torch.device("cpu"))
            fixed, win_fixed, fixed_info = runner.build_condition(completed, family, .2, "original_radius", 5, torch.device("cpu"))
            self.assertEqual(int(torch.count_nonzero(a)), np.count_nonzero(completed))
            self.assertTrue(torch.all((a < 0) <= (b < 0)))
            self.assertTrue(torch.equal(win_a, win_b))
            self.assertTrue(torch.equal(win_a, win_fixed))
            self.assertTrue(torch.equal(torch.sign(a), torch.sign(fixed)))
            self.assertAlmostEqual(info_a["raw_rho"], fixed_info["raw_rho"])
            self.assertAlmostEqual(info_a["ref_rho"], info_b["ref_rho"], places=5)
            self.assertAlmostEqual(info_a["post_rho"], 1., places=4)
            if family == "matched_er":
                self.assertAlmostEqual(float(torch.abs(a[a != 0]).std()), 0., places=6)

    def test_dry_run_counts_without_output(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            runner.main(["--dry-run"])
        self.assertIn('"rows_full_run": 115200', buffer.getvalue())
        self.assertIn('"configurations_per_condition": 24', buffer.getvalue())

    def test_real_small_run_resume_and_complete_merge(self):
        self.run_quietly(self.args(0))
        shards = list((self.root / "run/chunk_0").glob("repeat_*__*.csv"))
        self.assertEqual(len(shards), 4)
        with shards[0].open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["normalization"] for r in rows}, set(runner.NORMALIZATIONS))
        self.assertTrue(all(float(r["neuron_bias"]) == 0 for r in rows))
        self.assertTrue(all("IPC" not in r and "KR" not in r and "GR" not in r for r in rows))
        before = {p: p.read_bytes() for p in shards}
        with patch.object(runner, "evaluate_temporal_tasks", side_effect=AssertionError("resume reran a completed shard")):
            self.run_quietly(self.args(0) + ["--resume"])
        self.assertEqual(before, {p: p.read_bytes() for p in shards})
        with self.assertRaises(ValueError):
            runner.merge_run(self.root / "run")
        self.run_quietly(self.args(1))
        self.run_quietly(["--merge", "--out-dir", str(self.root / "run")])
        with (self.root / "run/mc_narma_crossover.ALL.csv").open() as f:
            merged = list(csv.DictReader(f))
        self.assertEqual(len(merged), 16)
        with self.assertRaises(FileExistsError):
            self.run_quietly(["--merge", "--out-dir", str(self.root / "run")])

    def test_changed_settings_cannot_resume(self):
        self.run_quietly(self.args())
        with self.assertRaises(ValueError):
            self.run_quietly(self.args() + ["--resume", "--seed", "123"])

    def test_atomic_writer_refuses_overwrite(self):
        p = self.root / "result.csv"
        runner.write_shard(p, [{"x": 1}])
        before = p.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.write_shard(p, [{"x": 2}])
        self.assertEqual(before, p.read_bytes())

    def test_slurm_launcher_forwards_the_requested_design(self):
        fake = self.root / "fake_python"
        fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        environment = {**os.environ, "PYTHON_BIN": str(fake), "SLURM_SUBMIT_DIR": str(runner.ROOT),
                       "SLURM_ARRAY_TASK_ID": "3", "SLURM_ARRAY_TASK_COUNT": "20",
                       "SLURM_ARRAY_TASK_MIN": "0", "SLURM_ARRAY_TASK_STEP": "1",
                       "SLURM_ARRAY_JOB_ID": "test-job", "SLURM_CPUS_PER_TASK": "8"}
        for key in ("PROJECT_ROOT", "OUTROOT", "RUN_ID", "TOTAL_REPEATS", "BASE_SEED"):
            environment.pop(key, None)
        result = subprocess.run(["bash", str(runner.ROOT / "r_mc_narma_crossover.sbatch")],
                                cwd=runner.ROOT, env=environment, check=True, capture_output=True, text=True)
        forwarded = json.loads(result.stdout)
        parsed = runner.parse_args(forwarded[1:])
        self.assertEqual(tuple(parsed.sign_fracs), runner.FRACTIONS)
        self.assertEqual(tuple(parsed.normalization_modes), runner.NORMALIZATIONS)
        self.assertEqual(tuple(parsed.rho_values), runner.RHOS)
        self.assertEqual(tuple(parsed.leak_values), runner.LEAKS)
        self.assertEqual(parsed.repeat_rank, 3)
        self.assertEqual(parsed.repeat_split, 20)
        self.assertTrue(parsed.resume)
        self.assertEqual(parsed.out_dir, Path("results/crossover_mc_narma10/test-job/chunk_3"))


if __name__ == "__main__":
    unittest.main()
