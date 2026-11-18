"""Batched CUDA recurrence for the unchanged continuous-STP equations.

One CUDA block owns one independent trajectory. Threads update presynaptic STP
states and postsynaptic neurons, synchronizing the shared transmission vector at
each timestep. Conditions sharing a matrix are batched to use the GPU together.
Neural states/weights remain float32 and utilization/resources remain float64,
as in the CPU reference. No fast-math approximation or activity clipping.
"""
import math

import numpy as np
from numba import cuda, float32

MAX_NODES = 1024


@cuda.jit(device=True)
def _rhs(u, x, rate, U, tf, td):
    return (U - u) / tf + U * (1 - u) * rate, (1 - x) / td - u * x * rate


@cuda.jit(device=True)
def _rk4(u, x, rate, U, tf, td, dt):
    ku1, kx1 = _rhs(u, x, rate, U, tf, td)
    ku2, kx2 = _rhs(u + dt * ku1 / 2, x + dt * kx1 / 2, rate, U, tf, td)
    ku3, kx3 = _rhs(u + dt * ku2 / 2, x + dt * kx2 / 2, rate, U, tf, td)
    ku4, kx4 = _rhs(u + dt * ku3, x + dt * kx3, rate, U, tf, td)
    return u + dt * (ku1 + 2 * ku2 + 2 * ku3 + ku4) / 6, x + dt * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6


@cuda.jit(fastmath=False)
def _trajectories(weights, inputs, signal, U, tf, td, dynamic, fixed_gain,
                  washout, train_end, substeps, states, statistics, training_gain):
    condition = cuda.blockIdx.x
    j = cuda.threadIdx.x
    n = weights.shape[0]
    transmitted = cuda.shared.array(MAX_NODES, float32)
    h = float32(0.)
    u = 0.
    x = 1.
    gsum = 0.
    gsquare = 0.
    gmin = 1e30
    gmax = -1e30
    train_sum = 0.
    invalid = 0.
    if j < n:
        u = U[condition, j]
    for t in range(len(signal)):
        if j < n:
            gain = fixed_gain[condition, j]
            if dynamic[condition]:
                for _ in range(substeps):
                    u, x = _rk4(u, x, abs(h), U[condition, j], tf[condition, j], td[condition, j], 1. / substeps)
                gain = u * x / U[condition, j]
                if not math.isfinite(u) or not math.isfinite(x) or u < -1e-8 or u > 1 + 1e-8 or x < -1e-8 or x > 1 + 1e-8:
                    invalid = 1.
            transmitted[j] = float32(gain * h)
            if washout <= t < train_end:
                train_sum += gain
            if t >= washout:
                gsum += gain
                gsquare += gain * gain
                gmin = min(gmin, gain)
                gmax = max(gmax, gain)
        cuda.syncthreads()
        if j < n:
            recurrent = float32(0.)
            for source in range(n):
                recurrent += weights[source, j] * transmitted[source]
            h = float32(math.tanh(recurrent + inputs[condition, j] * signal[t]))
            states[condition, t, j] = h
        # Every reader must finish before a thread writes the next transmission.
        cuda.syncthreads()
    if j < n:
        statistics[condition, j, 0] = gsum
        statistics[condition, j, 1] = gsquare
        statistics[condition, j, 2] = gmin
        statistics[condition, j, 3] = gmax
        statistics[condition, j, 4] = invalid
        training_gain[condition, j] = train_sum / (train_end - washout)


def require_cuda(device=0):
    if device < 0 or not cuda.is_available():
        raise RuntimeError("CUDA backend requested but unavailable. Run in an Orca GPU allocation; --backend cpu is for local reference checks only.")
    cuda.select_device(device)


def run_states_batch(weights, input_weights, signal, vectors, washout, train_end,
                     fixed_gains=None, substeps=4, device=0):
    """GPU simulation; return host states/diagnostics for the existing readouts.

    input_weights is B×N; vectors is a list of B optional per-neuron STP tuples.
    Each condition starts from the zero neuronal state and recovered synapses.
    """
    w = np.ascontiguousarray(weights, dtype=np.float32)
    b = np.ascontiguousarray(input_weights, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[1] != 1 or not len(signal):
        raise ValueError("Require a nonempty T×1 signal.")
    signal = np.ascontiguousarray(signal[:, 0])
    n = len(w)
    if w.shape != (n, n) or not 0 < n <= MAX_NODES or b.ndim != 2 or b.shape[1] != n or not len(b):
        raise ValueError("Require square W (1–1024 nodes) and nonempty B×N input weights.")
    if len(vectors) != len(b) or not all(np.isfinite(v).all() for v in (w, b, signal)):
        raise ValueError("Invalid STP batch or nonfinite simulation input.")
    if not 0 <= washout < train_end <= len(signal) or substeps < 1 or int(substeps) != substeps:
        raise ValueError("Invalid segments or substeps.")
    size = len(b)
    U, tf, td = np.full((size, n), .15), np.ones((size, n)), np.ones((size, n))
    dynamic = np.array([v is not None for v in vectors], dtype=bool)
    for i, v in enumerate(vectors):
        if v is None:
            continue
        if any(np.asarray(a).shape != (n,) or not np.isfinite(a).all() for a in v[:3]):
            raise ValueError("Invalid STP parameter vectors.")
        U[i], tf[i], td[i] = v[:3]
    if np.any(U <= 0) or np.any(U > 1) or np.any(tf <= 0) or np.any(td <= 0):
        raise ValueError("STP requires 0<U<=1 and positive time constants.")
    gain = np.ones((size, n)) if fixed_gains is None else np.asarray(fixed_gains, dtype=float)
    if gain.shape != (size, n) or not np.isfinite(gain).all() or np.any(gain < 0) or (fixed_gains is not None and dynamic.any()):
        raise ValueError("Frozen gains must be finite/nonnegative and cannot be combined with dynamic STP.")
    require_cuda(device)
    d_states = cuda.device_array((size, len(signal), n), dtype=np.float32)
    d_stats = cuda.device_array((size, n, 5), dtype=np.float64)
    d_training = cuda.device_array((size, n), dtype=np.float64)
    threads = 1 << (n - 1).bit_length()
    arrays = [cuda.to_device(a) for a in (w, b, signal, U, tf, td, dynamic, np.ascontiguousarray(gain))]
    _trajectories[size, threads](*arrays, washout, train_end, substeps, d_states, d_stats, d_training)
    cuda.synchronize()
    states, stats, training = d_states.copy_to_host(), d_stats.copy_to_host(), d_training.copy_to_host()
    if not np.isfinite(states).all() or np.any(stats[:, :, 4] != 0) or not np.isfinite(stats).all():
        raise ValueError("Nonfinite CUDA trajectory or STP state outside bounds; no clipping.")
    details = []
    samples = (len(signal) - washout) * n
    for i in range(size):
        mean = stats[i, :, 0].sum() / samples
        details.append(dict(gain_mean=float(mean),
                            gain_sd=float(np.sqrt(max(0., stats[i, :, 1].sum() / samples - mean * mean))),
                            gain_min=float(stats[i, :, 2].min()), gain_max=float(stats[i, :, 3].max()),
                            saturation_fraction=float(np.mean(abs(states[i, washout:]) > .9)),
                            mean_state_variance=float(np.var(states[i, washout:].astype(float), axis=0).mean())))
    return states, details, training


def check_cuda(device=0):
    """Compile and execute a tiny kernel before creating output or starting a run."""
    states, _, _ = run_states_batch(np.eye(3, dtype=np.float32) * .2,
                                    np.full((2, 3), .1, np.float32),
                                    np.ones((4, 1), np.float32), [None, None], 0, 2, device=device)
    if not np.isfinite(states).all():
        raise RuntimeError("CUDA startup check failed.")
    if cuda.config.ENABLE_CUDASIM:
        return dict(name="CUDA simulator (validation only)", simulated=True)
    gpu = cuda.get_current_device()
    name = gpu.name.decode() if isinstance(gpu.name, bytes) else str(gpu.name)
    return dict(name=name, compute_capability=list(gpu.compute_capability), simulated=False)
