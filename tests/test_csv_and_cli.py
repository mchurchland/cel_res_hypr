"""CSV identity, selective updates, and the legacy command-line entry point."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cel_res_hypr.experiments.csv_io import CSV_HEADER, save_rows

ROOT = Path(__file__).parents[1]


def original_rows():
    return json.loads((ROOT / "docs/baseline/baseline.json").read_text())["rows"]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_rank_patch_matches_full_identity_and_preserves_unrelated_fields(tmp_path):
    path = tmp_path / "experiment.csv"
    first, other = original_rows()[:2]
    save_rows(str(path), [tuple(first), tuple(other)])
    before = read_rows(path)
    update = first.copy()
    update[CSV_HEADER.index("KR")] = 7.0
    update[CSV_HEADER.index("GR")] = 2.0
    save_rows(str(path), [tuple(update)], rank_only=True)
    after = read_rows(path)
    assert after[1] == before[1]
    assert after[0]["KR"] == "7.0" and after[0]["GR"] == "2.0"
    for column in set(CSV_HEADER) - {"KR", "GR"}:
        assert after[0][column] == before[0][column]


def test_new_rank_identity_is_appended_with_blank_uncomputed_metrics(tmp_path):
    path = tmp_path / "experiment.csv"
    row = original_rows()[0]
    save_rows(str(path), [tuple(row)])
    row[CSV_HEADER.index("seed")] += 1
    save_rows(str(path), [tuple(row)], kr_only=True)
    after = read_rows(path)
    assert len(after) == 2
    assert after[1]["MC"] == after[1]["IPC"] == after[1]["GR"] == ""


def test_rank_patch_rejects_schema_without_changing_file(tmp_path):
    path = tmp_path / "experiment.csv"
    path.write_text("mode,KR\nreal,2\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unexpected CSV schema"):
        save_rows(str(path), [tuple(original_rows()[0])], rank_only=True)
    assert path.read_bytes() == before


def test_full_append_keeps_single_header(tmp_path):
    path = tmp_path / "experiment.csv"
    row = tuple(original_rows()[0])
    save_rows(str(path), [row])
    save_rows(str(path), [row], append=True)
    assert len(read_rows(path)) == 2
    assert path.read_text().count(",".join(CSV_HEADER)) == 1


def test_legacy_cli_runs_small_cpu_rank_experiment(tmp_path):
    adjacency = tmp_path / "adj.npy"
    np.save(
        adjacency,
        np.array(
            [[0.0, 1.0, -0.5], [-0.4, 0.0, 0.3], [0.2, -0.6, 0.0]], dtype=np.float32
        ),
    )
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    output = tmp_path / "results"
    command = [
        sys.executable,
        "inv_arc_test.py",
        "--job",
        "real",
        "--ce-adj",
        str(adjacency),
        "--ce-ei",
        "",
        "--out-dir",
        str(output),
        "--rho-values",
        "0.8",
        "--neuron-biases",
        "0.0",
        "--split",
        "12",
        "--rank",
        "0",
        "--n-repeats",
        "1",
        "--seed",
        "7",
        "--rank-only",
        "--cuda",
        "-1",
    ]
    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = read_rows(output / "invariance_variants.csv")
    assert len(rows) == 1
    assert rows[0]["seed"] == "8" and rows[0]["mode"] == "real"
    assert rows[0]["MC"] == rows[0]["IPC"] == ""
    assert np.isfinite(float(rows[0]["KR"])) and np.isfinite(float(rows[0]["GR"]))
