"""Deterministic synthetic result tables for analysis and plotting checks."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def plotting_results():
    rng = np.random.default_rng(1603)
    rows = []
    modes = [
        "real",
        "shuffle",
        "celW+connShuf",
        "conn_shuf_only",
        "local_sign+binary",
        "global_sign_pres",
        "binary+shuffle",
        "binary+conshuffle+wshuffle",
        "binary_base",
        "binary_base_topology_shuffle",
    ]
    for m, mode in enumerate(modes):
        for rep in range(4):
            offset = rng.uniform(0.5, 1.5)
            for i in range(8):
                rows.append(
                    dict(
                        mode=mode,
                        shuffle_id=rep,
                        src="fixture",
                        rho_target=[0.8, 1.05][i % 2],
                        leak=[0.6, 0.8][i // 2 % 2],
                        input_scale=[0.1, 0.5][i // 4],
                        neuron_bias=0.0,
                        MC=offset * (2 + 0.2 * m + 0.08 * i),
                        IPC=offset * (1 + 0.1 * m + 0.2 * i),
                        KR=offset * (3 + 0.3 * m + 0.1 * i),
                        GR=offset * (1 + 0.2 * m + 0.1 * i),
                        raw_rho=2 + 0.1 * m,
                        wt_mean=0.01 * m,
                        kl_to_gaussian=0.1 * m,
                        cosine_similarity=0.9,
                    )
                )
    return pd.DataFrame(rows)
