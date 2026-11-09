"""Compare the complete small experiment with pre-refactor scientific outputs."""
import json
from pathlib import Path

import numpy as np

from tools.smoke_experiment import small_experiment


def test_original_cpu_experiment_and_csv_schema(tmp_path):
    original = json.loads((Path(__file__).parents[1] / "docs/baseline/baseline.json").read_text())
    current = small_experiment(tmp_path)
    assert current["columns"] == original["columns"]
    assert current["simulation"] == original["simulation"]
    assert current["adjacency"] == original["adjacency"]
    np.testing.assert_allclose(current["trajectory"], original["trajectory"], rtol=2e-5, atol=2e-6)
    for actual, expected in zip(current["rows"], original["rows"], strict=True):
        for value, baseline in zip(actual, expected, strict=True):
            if isinstance(baseline, str):
                assert value == baseline
            else:
                np.testing.assert_allclose(value, baseline, rtol=2e-5, atol=2e-6)
    assert (tmp_path / "invariance_variants.csv").read_text().splitlines()[0] == ",".join(original["columns"])
