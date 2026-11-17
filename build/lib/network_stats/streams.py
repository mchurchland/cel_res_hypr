"""Independent-input KR and shared-tail GR stream protocols."""

import torch
from torch import Tensor


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

    streams = (
        torch.rand(
            n_streams,
            stream_length,
            n_inputs,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 2.0
        - 1.0
    )
    if common_tail_length:
        common_tail = (
            torch.rand(
                common_tail_length,
                n_inputs,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 2.0
            - 1.0
        )
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
        shared_continuation = torch.cat((shared_continuation, extra_shared), dim=0)

    streams_by_length: dict[int, Tensor] = {}
    for length in lengths:
        shared_tail = (
            shared_continuation[:length, :].unsqueeze(0).expand(n_streams, -1, -1)
        )
        streams_by_length[length] = torch.cat((fixed_prefix, shared_tail), dim=1)
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
