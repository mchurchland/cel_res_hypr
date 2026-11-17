"""Compatibility exports for legacy result aggregation scripts."""

from cel_res_hypr.paths import _safe_path
from cel_res_hypr.analysis.aggregation import (
    _read_glob,
    _ensure_columns,
    _dispersion_cv,
    _dispersion_v,
    _unique_hparam_rows,
    _build_combined,
    _assign_group_ids,
    _aggregate_over_hparams,
    _compute_dispersion_table,
    _compute_mean_table,
)
