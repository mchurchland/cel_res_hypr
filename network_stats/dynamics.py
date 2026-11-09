"""Leaky tanh reservoir dynamics with source-row recurrent weights."""

import torch
from torch import Tensor


@torch.no_grad()
def run_reservoir_with_pre(
    W: Tensor,
    Win: Tensor,
    u: Tensor,
    leak: float,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return states and preactivations, both [time, neurons], from zero state.

    For W[source,target], Win [neurons,inputs], u [time,inputs], the update
    is pre = W.T @ z + Win @ u[t] + bias and
    z = (1-leak)*z + leak*tanh(pre). Bias is a [neurons] vector.
    Historical buffers use the default tensor dtype, so the production path
    expects float32 under the default float32 setting (see docs/AUDIT.md).
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
    for t in range(T):
        pre = W.T @ z + (Win @ u[t : t + 1, :].T).squeeze() + bias_vec
        h = torch.tanh(pre)
        z = (1 - leak) * z + leak * h
        X[t] = z
        Pre[t] = pre
    return X, Pre


@torch.no_grad()
def run_reservoir_states(
    W: Tensor,
    Win: Tensor,
    u: Tensor,
    leak: float,
    bias: Tensor | None = None,
) -> Tensor:
    """Return [time,neurons] states from zero with buffers matching W.dtype.

    Uses W[source,target], Win [neurons,inputs], u [time,inputs], and
    z = (1-leak)*z + leak*tanh(W.T @ z + Win @ u[t] + bias).
    """
    n_nodes = W.shape[0]
    z = torch.zeros(n_nodes, device=W.device, dtype=W.dtype)
    states = torch.zeros(u.shape[0], n_nodes, device=W.device, dtype=W.dtype)
    if bias is None:
        bias_vec = torch.zeros(n_nodes, device=W.device, dtype=W.dtype)
    else:
        bias_vec = bias.to(device=W.device, dtype=W.dtype).view(n_nodes)

    for t in range(u.shape[0]):
        pre = W.T @ z + (Win @ u[t : t + 1, :].T).squeeze() + bias_vec
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
) -> Tensor:
    """Return [streams,neurons] final states for [streams,time,inputs] input.

    Each stream starts at the supplied [neurons] initial_state. The batched
    preactivation is z @ W + u[t] @ Win.T + bias with W[source,target].
    """
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

    for t in range(input_streams.shape[1]):
        pre = z @ W + input_streams[:, t, :] @ Win.T + bias_vec
        z = (1 - leak) * z + leak * torch.tanh(pre)
    return z
