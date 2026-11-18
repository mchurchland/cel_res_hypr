"""CUDA/reference agreement; NUMBA_ENABLE_CUDASIM=1 also runs these without a GPU."""
import unittest

import numpy as np
from numba import cuda

from network_stats.stp_weight_controls import run_states, stp_vectors
from network_stats.stp_weight_controls_cuda import check_cuda, run_states_batch


@unittest.skipUnless(cuda.is_available(), "Requires CUDA or NUMBA_ENABLE_CUDASIM=1")
class CUDATrajectoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = (rng.normal(size=(4, 4)) * .13).astype(np.float32)
        self.b = (rng.normal(size=(5, 4)) * .2).astype(np.float32)
        self.signal = rng.uniform(-1, 1, (20, 1)).astype(np.float32)
        self.vectors = [None, stp_vectors(4, "reference", 1., np.arange(4)),
                        stp_vectors(4, "reference", 0., np.arange(4)),
                        stp_vectors(4, "fast", .5, np.arange(4)),
                        stp_vectors(4, "slow", .5, np.arange(4))]

    def test_batch_matches_cpu_reference_all_stp_types(self):
        actual, details, means = run_states_batch(self.w, self.b, self.signal, self.vectors, 4, 14)
        for i, vectors in enumerate(self.vectors):
            expected = run_states(self.w, self.b[i, :, None], self.signal, vectors, 4, 14)
            np.testing.assert_allclose(actual[i], expected[0], atol=2e-7, rtol=3e-6)
            np.testing.assert_allclose(means[i], expected[2], atol=2e-7, rtol=3e-6)
            for field in details[i]:
                self.assertAlmostEqual(details[i][field], expected[1][field], places=6)

    def test_frozen_coefficients_are_training_only_and_match_cpu(self):
        _, _, means = run_states_batch(self.w, self.b, self.signal, self.vectors, 4, 14)
        modified = self.signal.copy()
        modified[14:] = -5
        _, _, changed = run_states_batch(self.w, self.b, modified, self.vectors, 4, 14)
        np.testing.assert_array_equal(means, changed)
        states, _, _ = run_states_batch(self.w, self.b, self.signal, [None] * 5, 4, 14, fixed_gains=means)
        for i in range(5):
            expected = run_states(self.w, self.b[i, :, None], self.signal, washout=4, train_end=14, fixed_gain=means[i])
            np.testing.assert_allclose(states[i], expected[0], atol=2e-7, rtol=3e-6)

    def test_silent_reset_no_cross_batch_coupling_and_startup_probe(self):
        a = run_states_batch(self.w, self.b, np.zeros_like(self.signal), self.vectors, 4, 14)
        np.testing.assert_array_equal(a[0], 0.)
        np.testing.assert_array_equal(a[2], 1.)
        full = run_states_batch(self.w, self.b, self.signal, self.vectors, 4, 14)[0]
        one = run_states_batch(self.w, self.b[3:4], self.signal, self.vectors[3:4], 4, 14)[0]
        np.testing.assert_array_equal(full[3], one[0])
        self.assertIn("name", check_cuda())

    def test_runner_gpu_batch_readout_and_optional_control_match_cpu(self):
        import torch
        from network_stats.temporal_tasks import TemporalTaskSettings
        from stp_weight_controls import evaluate, evaluate_cuda

        task = TemporalTaskSettings(washout=4, train=10, validation=3, test=6, mc_max_delay=2)
        rng = np.random.default_rng(19)
        data = dict(mc_input=torch.from_numpy(rng.uniform(-1, 1, (20, 1)).astype(np.float32)),
                    narma_input=torch.from_numpy(rng.uniform(0, .5, (23, 1)).astype(np.float32)),
                    narma_target=torch.from_numpy(rng.uniform(.1, .5, (23, 1))))
        conditions = [(self.b[i, :, None], self.vectors[i]) for i in (0, 3)]
        actual = evaluate_cuda(self.w, conditions, data, task, True, 0, 1)
        for result, (b, v) in zip(actual, conditions):
            expected = evaluate(self.w, b, data, task, v, True)
            self.assertEqual(set(result), set(expected))
            for control in expected:
                for field in expected[control]:
                    self.assertAlmostEqual(result[control][field], expected[control][field], delta=3e-5)


class CUDAValidationTests(unittest.TestCase):
    def test_bad_batches_rejected_before_device_access(self):
        w = np.eye(3, dtype=np.float32)
        b = np.ones((1, 3), np.float32)
        signal = np.ones((5, 1), np.float32)
        with self.assertRaises(ValueError):
            run_states_batch(w, b, signal, [], 1, 3)
        with self.assertRaises(ValueError):
            run_states_batch(w, b, signal, [None], 1, 3, fixed_gains=-b)


if __name__ == "__main__":
    unittest.main()
