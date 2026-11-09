"""Production simulation defaults and per-repeat variant context.

Grid columns are (spectral radius, leak, input scale, bias half-range).
Dataclass field names/defaults are retained for legacy callers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class SimulationParams:
    """Container for the time-series/metric settings used by run_one."""

    washout: int = int(500)
    t_train: int = int(1500)
    t_test: int = int(500)
    mc_max_delay: int = 30
    ipc_max_delay: int = 30
    ipc_orders: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    ridge_alpha: float = 1e-4
    kr_num_streams: int = 300
    kr_stream_length: int = 10
    kr_seed_offset: int = 29_000_003
    gr_num_streams: int = 300
    gr_stream_length: int = 10
    gr_common_tail_length: int = 3
    gr_seed_offset: int = 23_000_003


DEFAULT_SIM_PARAMS = SimulationParams()


@dataclass(frozen=True)
class VariantContext:
    """Inputs that stay constant while sweeping hyperparameters."""

    ce_W_bio: np.ndarray
    ce_ei: np.ndarray | None
    ws_k: int
    col_params: Sequence[tuple[float, ...]]
    device: torch.device
    seed: int
    sid: int
    er_p: float = 0.1
    ws_p: float = 0.1
    per_neg: float | None = None
    alpha: float | None = None
    src_tag: str = "chunk_0"
    sim_params: SimulationParams = DEFAULT_SIM_PARAMS
    input_idx: np.ndarray | None = None
    output_idx: np.ndarray | None = None
    normalization_mode: str = "spectral_radius"
    label_normalization: bool = False
    rank_only: bool = False
    kr_only: bool = False
