"""Compatibility exports for reservoir experiment scripts.

Implementations live in cel_res_hypr.experiments; legacy names remain valid.
"""

from cel_res_hypr.experiments.config import (
    SimulationParams,
    DEFAULT_SIM_PARAMS,
    VariantContext,
)

from cel_res_hypr.experiments.evaluation import (
    _kl_empirical_to_fitted_gaussian,
    evaluate_reservoir,
    _apply_input_subset,
    _unpack_col_param,
    _make_neuron_bias,
    _family_baseline_ref,
    _run_variant_row,
)

from cel_res_hypr.experiments.csv_io import (
    CSV_HEADER,
    RANK_ROW_KEY_COLUMNS,
    _rank_row_key,
    _update_rank_columns,
    save_rows,
)

from cel_res_hypr.experiments.variants import (
    VARIANT_LABELS,
    VARIANT_DESCRIPTIONS,
    VARIANT_ALIASES,
    VARIANT_KEYS,
    WEIGHT_TEST_FEATURES,
    _seed,
    _nnz_match_ce,
    _resolve_key,
    _weight_test_feature_conn,
    _require_ce,
    run_variant,
    list_variants,
)
