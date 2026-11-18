"""Weight matching, pairing, STP compatibility, leakage and cluster restart tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import stp_weight_controls as runner
from network_stats.short_term_plasticity import PRESETS, run_stp_states
from network_stats.stp_weight_controls import (
    FAMILIES, PARAMETER_SETS, condition_grid, run_states, scale_matrix,
    spectral_radius, stp_vectors, unsigned_controls, match_signed_weights,
)
from tools.analyze_stp_weight_controls import paired_tables, estimate


class ControlModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.w = (rng.normal(size=(6, 6)) * .12).astype(np.float32)
        self.b = (rng.normal(size=(6, 1)) * .1).astype(np.float32)
        self.signal = rng.uniform(-1, 1, (150, 1)).astype(np.float32)

    def test_reference_presets_match_existing_pilot_bitwise(self):
        for mode, fraction in (("facilitating", 1.), ("depressing", 0.)):
            vectors = stp_vectors(6, "reference", fraction, np.arange(6))
            actual = run_states(self.w, self.b, self.signal, vectors, 10, 80)
            expected = run_stp_states(self.w, self.b, self.signal, PRESETS[mode], 10)
            np.testing.assert_array_equal(actual[0], expected[0])
            for field in ("gain_mean", "gain_sd", "gain_min", "gain_max"):
                self.assertEqual(actual[1][field], expected[1][field])

    def test_all_parameter_sets_bounded_reproducible_and_zero_input_rest(self):
        for preset in PARAMETER_SETS:
            for fraction in (0., .5, 1.):
                v = stp_vectors(6, preset, fraction, np.arange(6))
                a = run_states(self.w, self.b, self.signal, v, 10, 80)
                b = run_states(self.w, self.b, self.signal, v, 10, 80)
                np.testing.assert_array_equal(a[0], b[0])
                zero = run_states(self.w, self.b, np.zeros_like(self.signal), v, 10, 80)
                np.testing.assert_array_equal(zero[0], 0.)
                np.testing.assert_array_equal(zero[2], 1.)

    def test_frozen_gain_calibration_never_uses_validation_or_test(self):
        v = stp_vectors(6, "slow", .5, np.arange(6))
        a = run_states(self.w, self.b, self.signal, v, 10, 80)
        modified = self.signal.copy()
        modified[80:] = 100
        b = run_states(self.w, self.b, modified, v, 10, 80)
        np.testing.assert_array_equal(a[2], b[2])
        self.assertFalse(np.array_equal(a[0][80:], b[0][80:]))

    def test_constant_gain_is_equivalent_to_static_source_weight_scaling(self):
        gains = np.linspace(.5, 1.5, 6)
        a = run_states(self.w, self.b, self.signal, washout=10, train_end=80, fixed_gain=gains)
        b = run_states(self.w * gains[:, None], self.b, self.signal, washout=10, train_end=80)
        np.testing.assert_allclose(a[0], b[0], atol=2e-7, rtol=2e-6)
        self.assertAlmostEqual(a[1]["gain_sd"], np.std(gains), places=12)

    def test_exact_weight_multiset_and_masks_before_and_after_shared_scaling(self):
        ce = np.zeros((6, 6), np.float32)
        ce[np.arange(6), (np.arange(6) + 1) % 6] = [1, 2, 3, 5, 8, 13]
        mask = np.roll(ce != 0, 1, axis=0)
        np.fill_diagonal(mask, False)
        # Use the opposite ring to retain exactly six loop-free ER-fixture edges.
        mask = np.zeros_like(ce, dtype=bool)
        mask[np.arange(6), (np.arange(6) - 1) % 6] = True
        controls = unsigned_controls(ce, mask, 4)
        for family in ("ce_shuffled", "er_empirical"):
            np.testing.assert_array_equal(np.sort(controls[family][controls[family] != 0]),
                                          np.sort(controls["ce_empirical"][ce != 0]))
        for family, w in controls.items():
            np.testing.assert_array_equal(w != 0, ce != 0 if family.startswith("ce_") else mask)
            self.assertAlmostEqual(float(w[w != 0].mean()), 1., places=6)
        ref = spectral_radius(controls["ce_empirical"])
        left, _ = scale_matrix(controls["ce_empirical"], "shared_ce_scale", .95, ref)
        right, _ = scale_matrix(controls["er_empirical"], "shared_ce_scale", .95, ref)
        np.testing.assert_array_equal(np.sort(left[left != 0]), np.sort(right[right != 0]))
        own, _ = scale_matrix(controls["er_halfnormal"], "own_radius", .95, ref)
        self.assertAlmostEqual(spectral_radius(own), .95, places=6)

    def test_signed_weight_control_matches_excitation_and_inhibition_separately(self):
        ce = np.array([[0, -2, 5], [3, 0, -7], [0, 1, 0]], dtype=np.float32)
        signs = np.array([[0, 1, -1], [-1, 0, 1], [1, 0, 0]], dtype=np.int8)
        actual = match_signed_weights(ce, signs, 9)
        np.testing.assert_array_equal(np.sort(actual[actual != 0]), np.sort(ce[ce != 0]))
        np.testing.assert_array_equal(np.sign(actual), signs)
        self.assertEqual(float(abs(actual[actual < 0]).sum()), 9.)
        with self.assertRaises(ValueError):
            match_signed_weights(ce, abs(signs), 9)

    def test_mixed_assignments_nested_and_validation(self):
        order = np.random.default_rng(8).permutation(297)
        a = stp_vectors(297, "fast", .25, order)[3]
        b = stp_vectors(297, "reference", .5, order)[3]
        c = stp_vectors(297, "slow", .75, order)[3]
        self.assertTrue(np.all(a <= b) and np.all(b <= c))
        self.assertEqual(int(b.sum()), 148)
        self.assertEqual(len(condition_grid(list(PARAMETER_SETS), [.5])), 10)
        with self.assertRaises(ValueError):
            condition_grid(["reference"], [0.])
        with self.assertRaises(ValueError):
            stp_vectors(6, "reference", .5, np.zeros(6, dtype=int))
        with self.assertRaises(ValueError):
            run_states(self.w, self.b, self.signal, fixed_gain=np.full(6, -1.), washout=10, train_end=80)
        with self.assertRaises(ValueError):
            scale_matrix(np.zeros((3, 3)), "own_radius", .95, 1.)


class ClusterIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        ce = np.zeros((8, 8), np.float32)
        unknown = ce.copy()
        for i in range(8):
            ce[i, (i + 1) % 8] = -2 if i % 3 == 0 else i + 1
            unknown[i, (i + 2) % 8] = 3
        np.save(self.root / "known.npy", ce)
        np.save(self.root / "unknown.npy", unknown)
        (self.root / "nodes.txt").write_text("\n".join(f"n{i}" for i in range(8)) + "\n")
        pd.DataFrame(dict(node_index=[0, 3], neuron=["n0", "n3"])).to_csv(self.root / "sensory.csv", index=False)

    def args(self, rank):
        return ["--out-dir", str(self.root / "run"), "--ce-adj", str(self.root / "known.npy"),
                "--ce-unknown-sign-weights", str(self.root / "unknown.npy"),
                "--node-names", str(self.root / "nodes.txt"), "--sensory-labels", str(self.root / "sensory.csv"),
                "--n-repeats", "2", "--repeat-split", "2", "--repeat-rank", str(rank),
                "--ei-percent", "0", "50", "--parameter-sets", "reference", "--input-scales", ".1",
                "--washout", "10", "--train-length", "40", "--validation-length", "20", "--test-length", "20",
                "--mc-max-delay", "4", "--narma-burnin", "100", "--gain-control", "--resume"]

    def run_quietly(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args)

    def test_small_run_resume_strict_merge_pairing_and_analysis(self):
        self.run_quietly(self.args(0))
        out = self.root / "run"
        files = sorted(out.glob("repeats/repeat_0000/*.csv"))
        self.assertEqual(len(files), 28)
        before = {p: p.read_bytes() for p in files}
        with patch.object(runner, "evaluate", side_effect=AssertionError("Resume reran scores")):
            self.run_quietly(self.args(0))
        self.assertEqual(before, {p: p.read_bytes() for p in files})
        with self.assertRaises(ValueError):
            runner.merge_run(out)
        self.run_quietly(self.args(1))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.merge_run(out)
            runner.merge_run(out)
        frame = pd.read_csv(out / "stp_weight_controls.ALL.csv")
        self.assertEqual(len(frame), 784)
        self.assertTrue(np.isfinite(frame[runner.SCORES]).all().all())
        self.assertEqual(set(frame.n_driven), {2, 8})
        for _, g in frame.groupby(["repeat_id", "ei_percent", "normalization", "input_scale", "input_scheme", "stp_condition", "gain_control"]):
            self.assertAlmostEqual(float(g.input_l2_norm.min()), float(g.input_l2_norm.max()), places=7)
        contrasts, changes, frozen = paired_tables(frame, 100)
        self.assertEqual(set(contrasts.n_realizations), {2})
        self.assertFalse(changes.empty or frozen.empty)
        own = frame[frame.normalization == "own_radius"]
        np.testing.assert_allclose(own.initial_post_rho, .95, atol=1e-6)
        for _, g in frame[frame.normalization == "shared_ce_scale"].groupby(["repeat_id", "ei_percent"]):
            self.assertEqual(g.initial_scale.nunique(), 1)
        manifest = json.loads((out / "manifest.json").read_text())
        with np.load(out / "repeats/repeat_0000/inputs_and_networks.npz") as cache:
            for source in ("ce", "er"):
                a, b = cache[f"{source}_signs_ei0"], cache[f"{source}_signs_ei50"]
                self.assertTrue(np.all((a < 0) <= (b < 0)))
                self.assertEqual(np.count_nonzero(b < 0), 8)
        self.assertFalse(manifest["spec"]["dynamic_renormalization"])
        # Missing rows and foreign settings cannot be silently merged/resumed.
        damaged = frame.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            runner.validate_frame(damaged, manifest["spec"], manifest["experiment_id"], runner.expected_keys(manifest["spec"]))
        with self.assertRaises(ValueError):
            self.run_quietly(self.args(0) + ["--seed", "9"])
        self.assertTrue(np.isnan(estimate([1.], 100)["ci_low"]))


if __name__ == "__main__":
    unittest.main()
