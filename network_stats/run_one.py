from dataclasses import dataclass
import math

import torch
from torch import Tensor

# ---- repo helpers (reuse your utils/stats) ----
from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC
from util.util import spectral_radius_power


@dataclass(frozen=True)
class STPParams:
    """Neuron-wise rate adaptation of Tsodyks–Markram synapses (time in steps)."""

    utilization: float = 0.1
    tau_recovery: float = 50.0
    tau_facilitation: float = 10.0
    gain_mode: str = "raw"
    update_rate: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.utilization) or not 0.0 < self.utilization <= 1.0:
            raise ValueError("STP utilization must be in (0, 1].")
        if (not math.isfinite(self.tau_recovery) or not math.isfinite(self.tau_facilitation)
                or self.tau_recovery < 1.0 or self.tau_facilitation < 1.0):
            raise ValueError("STP time constants must be at least one simulation step.")
        if self.gain_mode not in {"raw", "baseline"}:
            raise ValueError("STP gain_mode must be 'raw' or 'baseline'.")
        if not math.isfinite(self.update_rate) or not 0.0 < self.update_rate <= 1.0:
            raise ValueError("STP update_rate must be in (0, 1].")


def _stp_initial_state(z: Tensor, params: STPParams) -> tuple[Tensor, Tensor]:
    return torch.ones_like(z), torch.full_like(z, params.utilization)


def _stp_next(
    z: Tensor, resources: Tensor, utilization: Tensor, params: STPParams
) -> tuple[Tensor, Tensor]:
    # The reservoir state is signed; map it to a bounded, nonnegative rate.
    activity = ((z + 1.0) * 0.5).clamp(0.0, 1.0)
    resource_change = (
        (1.0 - resources) / params.tau_recovery
        - activity * resources * utilization
    )
    utilization_change = (
        (params.utilization - utilization) / params.tau_facilitation
        + params.utilization * (1.0 - utilization) * activity
    )
    next_resources = resources + params.update_rate * resource_change
    next_utilization = utilization + params.update_rate * utilization_change
    return next_resources, next_utilization


def _stp_gain(resources: Tensor, utilization: Tensor, params: STPParams) -> Tensor:
    gain = resources * utilization
    if params.gain_mode == "baseline":
        gain = gain / params.utilization
    return gain


def _effective_weights(
    W: Tensor, resources: Tensor, utilization: Tensor, params: STPParams
) -> Tensor:
    # W[i, j] runs from presynaptic i to postsynaptic j.
    return _stp_gain(resources, utilization, params).unsqueeze(-1) * W


def _weight_snapshot(
    W: Tensor, step: int, n_edges: int, saturation_fraction: float
) -> tuple[int, float, float, float, float]:
    """Radius and signed/absolute mean over the original nonzero edge set."""
    return (
        step,
        spectral_radius_power(W),
        float((W.sum() / n_edges).item()),
        float((W.abs().sum() / n_edges).item()),
        saturation_fraction,
    )


@torch.no_grad()
def run_reservoir_with_pre(
    W: Tensor,
    Win: Tensor,
    u: Tensor,
    leak: float,
    bias: Tensor | None = None,
    stp_params: STPParams | None = None,
    rho_trace_interval: int = 100,
    rho_trace: list[tuple[int, float, float, float, float]] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Basic ESN-style update loop with pre-activation tracking.
    See: https://github.com/cknd/pyESN/blob/master/pyESN.py for a similar state update.
    """
    N = W.shape[0]
    T = u.shape[0]
    z = torch.zeros(N, device=W.device)
    X = torch.zeros(T, N, device=W.device)
    Pre = torch.zeros(T, N, device=W.device)
    if bias is None:
        bias_vec = torch.zeros(N, device=W.device, dtype=W.dtype)
    else:
        bias_vec = bias.to(device=W.device, dtype=W.dtype).view(N)
    if stp_params is not None:
        resources, utilization = _stp_initial_state(z, stp_params)
        if rho_trace is not None:
            if rho_trace_interval <= 0:
                raise ValueError("rho_trace_interval must be positive.")
            n_edges = max(int(torch.count_nonzero(W).item()), 1)
            rho_trace.append(_weight_snapshot(
                _effective_weights(W, resources, utilization, stp_params),
                0, n_edges, float("nan"),
            ))
    for t in range(T):
        if stp_params is None:
            recurrent = W.T @ z
        else:
            recurrent = (z * _stp_gain(resources, utilization, stp_params)) @ W
        pre = recurrent + (Win @ u[t:t+1, :].T).squeeze() + bias_vec
        h = torch.tanh(pre)
        if stp_params is not None:
            resources, utilization = _stp_next(z, resources, utilization, stp_params)
        z = (1 - leak) * z + leak * h
        X[t] = z
        Pre[t] = pre
        if stp_params is not None and rho_trace is not None and (t + 1) % rho_trace_interval == 0:
            window = Pre[t + 1 - rho_trace_interval:t + 1]
            saturation_fraction = float((window.abs() >= 2.0).float().mean().item())
            rho_trace.append(_weight_snapshot(
                _effective_weights(W, resources, utilization, stp_params),
                t + 1, n_edges, saturation_fraction,
            ))
    return X, Pre


@torch.no_grad()
def run_reservoir_states(
    W: Tensor,
    Win: Tensor,
    u: Tensor,
    leak: float,
    bias: Tensor | None = None,
) -> Tensor:
    """Run one input trajectory from the zero state and return every state."""
    n_nodes = W.shape[0]
    z = torch.zeros(n_nodes, device=W.device, dtype=W.dtype)
    states = torch.zeros(u.shape[0], n_nodes, device=W.device, dtype=W.dtype)
    if bias is None:
        bias_vec = torch.zeros(n_nodes, device=W.device, dtype=W.dtype)
    else:
        bias_vec = bias.to(device=W.device, dtype=W.dtype).view(n_nodes)

    for t in range(u.shape[0]):
        pre = W.T @ z + (Win @ u[t:t+1, :].T).squeeze() + bias_vec
        z = (1 - leak) * z + leak * torch.tanh(pre)
        states[t] = z
    return states


@torch.no_grad()
def run_reservoir_stream_batch(
    W: Tensor,
    Win: Tensor,
    input_streams: Tensor,
    leak: float,
    initial_state: Tensor,
    bias: Tensor | None = None,
    stp_params: STPParams | None = None,
) -> Tensor:
    """Return one final reservoir state for every input stream in a batch."""
    if input_streams.dim() != 3:
        raise ValueError("input_streams must have shape [streams, time, inputs].")

    n_streams = input_streams.shape[0]
    n_nodes = W.shape[0]
    z = initial_state.to(device=W.device, dtype=W.dtype).view(1, n_nodes)
    z = z.expand(n_streams, n_nodes).clone()
    if bias is None:
        bias_vec = torch.zeros(n_nodes, device=W.device, dtype=W.dtype)
    else:
        bias_vec = bias.to(device=W.device, dtype=W.dtype).view(n_nodes)
    if stp_params is not None:
        resources, utilization = _stp_initial_state(z, stp_params)

    for t in range(input_streams.shape[1]):
        if stp_params is None:
            recurrent = z @ W
        else:
            recurrent = (z * _stp_gain(resources, utilization, stp_params)) @ W
        pre = recurrent + input_streams[:, t, :] @ Win.T + bias_vec
        if stp_params is not None:
            resources, utilization = _stp_next(z, resources, utilization, stp_params)
        z = (1 - leak) * z + leak * torch.tanh(pre)
    return z


def make_gr_input_streams(
    n_streams: int,
    stream_length: int,
    common_tail_length: int,
    n_inputs: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Construct the similar-input ensemble used by Vidamour/RCbench GR.

    Streams are independently sampled from U[-1, 1], except that the final
    ``common_tail_length`` input vectors are identical across every stream.
    The differing prefix represents task-irrelevant variation; the common
    tail defines the recent signal over which the reservoir should generalize.
    """
    if n_streams <= 0:
        raise ValueError("n_streams must be positive.")
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    if not 0 <= common_tail_length <= stream_length:
        raise ValueError("common_tail_length must lie in [0, stream_length].")

    streams = torch.rand(
        n_streams,
        stream_length,
        n_inputs,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 2.0 - 1.0
    if common_tail_length:
        common_tail = torch.rand(
            common_tail_length,
            n_inputs,
            device=device,
            dtype=dtype,
            generator=generator,
        ) * 2.0 - 1.0
        streams[:, -common_tail_length:, :] = common_tail
    return streams


def make_gr_input_stream_sweep(
    n_streams: int,
    prefix_length: int,
    common_tail_lengths: list[int],
    reference_tail_length: int,
    n_inputs: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> dict[int, Tensor]:
    """Construct a fixed-prefix sweep over shared-continuation length.

    Every condition reuses the same independently sampled prefix for each
    stream and the first ``L`` vectors of one shared continuation. The total
    stream length is therefore ``prefix_length + L``.

    The reference condition is generated first through
    :func:`make_gr_input_streams`, so it is exactly identical to the production
    protocol with ``stream_length=prefix_length + reference_tail_length``.
    Longer conditions extend that same shared continuation, while shorter
    conditions stop it earlier.
    """
    lengths = sorted(set(int(length) for length in common_tail_lengths))
    if not lengths:
        raise ValueError("common_tail_lengths must contain at least one value.")
    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive.")
    if reference_tail_length < 0:
        raise ValueError("reference_tail_length must be non-negative.")
    if any(length < 0 for length in lengths):
        raise ValueError("common_tail_lengths must be non-negative.")

    reference_streams = make_gr_input_streams(
        n_streams=n_streams,
        stream_length=prefix_length + reference_tail_length,
        common_tail_length=reference_tail_length,
        n_inputs=n_inputs,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    fixed_prefix = reference_streams[:, :prefix_length, :]
    shared_continuation = reference_streams[0, prefix_length:, :]

    extra_length = max(lengths) - reference_tail_length
    if extra_length > 0:
        extra_shared = make_gr_input_streams(
            n_streams=1,
            stream_length=extra_length,
            common_tail_length=0,
            n_inputs=n_inputs,
            device=device,
            dtype=dtype,
            generator=generator,
        )[0]
        shared_continuation = torch.cat(
            (shared_continuation, extra_shared), dim=0
        )

    streams_by_length: dict[int, Tensor] = {}
    for length in lengths:
        shared_tail = shared_continuation[:length, :].unsqueeze(0).expand(
            n_streams, -1, -1
        )
        streams_by_length[length] = torch.cat(
            (fixed_prefix, shared_tail), dim=1
        )
    return streams_by_length


def make_kr_input_streams(
    n_streams: int,
    stream_length: int,
    n_inputs: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Construct independent U[-1, 1] sequences for kernel rank."""
    return make_gr_input_streams(
        n_streams=n_streams,
        stream_length=stream_length,
        common_tail_length=0,
        n_inputs=n_inputs,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def run_one(W: Tensor, Win: Tensor, leak: float, device: torch.device,WASHOUT: int,
            T_TRAIN: int, T_TEST: int,
            MC_MAX_DELAY: int, IPC_MAX_DELAY: int, IPC_ORDERS: list[int],
            RIDGE_ALPHA: float, output_idx: Tensor | None = None,
            bias: Tensor | None = None,
            kr_num_streams: int = 300,
            kr_stream_length: int = 10,
            kr_seed: int = 0,
            gr_num_streams: int = 300,
            gr_stream_length: int = 10,
            gr_common_tail_length: int = 3,
            gr_seed: int = 0,
            rank_only: bool = False,
            kr_only: bool = False,
            stp_params: STPParams | None = None,
            rho_trace_interval: int = 100) -> dict:
    """
    End-to-end reservoir evaluation computing MC/IPC/KR/GR and controllability metrics.
    Wraps the ESN update plus metrics pipeline; see reservoirpy/pyESN for similar evaluation flows:
    [1] Jaeger, H. (2001). Short term memory in echo state networks.    """
    if rank_only and kr_only:
        raise ValueError("rank_only and kr_only are mutually exclusive.")
    if stp_params is not None and (rank_only or kr_only):
        raise ValueError("STP requires a full MC/IPC/KR/GR run for its radius trace.")

    metrics_only = rank_only or kr_only
    rho_trace: list[tuple[int, float, float, float, float]] = []
    if not metrics_only:
        T_total = WASHOUT + T_TRAIN + T_TEST
        u = (torch.rand(T_total, 1, device=device) * 2.0 - 1.0) ## rescale to [-1, 1]
        if stp_params is None:
            X, _ = run_reservoir_with_pre(W, Win, u, leak, bias=bias)
        else:
            X, _ = run_reservoir_with_pre(
                W, Win, u, leak, bias=bias, stp_params=stp_params,
                rho_trace_interval=rho_trace_interval, rho_trace=rho_trace,
            )

    # Both rank metrics use one final state from each sequence, with every
    # sequence initialized from the same zero state. KR sequences are fully
    # independent; GR sequences share the configured recent input tail.
    rank_initial_state = torch.zeros(W.shape[0], device=device, dtype=W.dtype)
    kr_generator = torch.Generator(device=device)
    kr_generator.manual_seed(kr_seed)
    kr_streams = make_kr_input_streams(
        n_streams=kr_num_streams,
        stream_length=kr_stream_length,
        n_inputs=Win.shape[1],
        device=device,
        dtype=W.dtype,
        generator=kr_generator,
    )
    X_kr = run_reservoir_stream_batch(
        W,
        Win,
        kr_streams,
        leak,
        initial_state=rank_initial_state,
        bias=bias,
        **({"stp_params": stp_params} if stp_params is not None else {}),
    )

    X_gr = None
    if not kr_only:
        gr_generator = torch.Generator(device=device)
        gr_generator.manual_seed(gr_seed)
        gr_streams = make_gr_input_streams(
            n_streams=gr_num_streams,
            stream_length=gr_stream_length,
            common_tail_length=gr_common_tail_length,
            n_inputs=Win.shape[1],
            device=device,
            dtype=W.dtype,
            generator=gr_generator,
        )
        X_gr = run_reservoir_stream_batch(
            W,
            Win,
            gr_streams,
            leak,
            initial_state=rank_initial_state,
            bias=bias,
            **({"stp_params": stp_params} if stp_params is not None else {}),
        )
    if output_idx is not None:
        idx = torch.as_tensor(output_idx, device=device, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("output_idx must contain at least one node when provided.")
        if not metrics_only:
            X = X.index_select(1, idx)
        X_kr = X_kr.index_select(1, idx)
        if X_gr is not None:
            X_gr = X_gr.index_select(1, idx)

    if metrics_only:
        MC_total = float("nan")
        IPC_total = float("nan")
    else:
        Xtr = X[WASHOUT:WASHOUT+T_TRAIN] ## t_train
        Xte = X[WASHOUT+T_TRAIN:] ## t_test
        utr = u[WASHOUT:WASHOUT+T_TRAIN] ## u_train
        ute = u[WASHOUT+T_TRAIN:] ## u_test
        MC_total, _ = compute_MC(Xtr, Xte, utr, ute, MC_MAX_DELAY, RIDGE_ALPHA,device)
        IPC_total = compute_IPC(
            Xtr,
            Xte,
            utr,
            ute,
            IPC_MAX_DELAY,
            RIDGE_ALPHA,
            device,
            IPC_ORDERS,
        )
    KR_val = compute_KR(X_kr)
    GR_val = float("nan") if X_gr is None else compute_GR(X_gr)
    scores = dict(MC=MC_total, IPC=IPC_total, KR=KR_val, GR=GR_val)
    if stp_params is not None:
        scores["rho_trace"] = rho_trace
    return scores
