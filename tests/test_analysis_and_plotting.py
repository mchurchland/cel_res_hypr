"""Grouping policies, imported interfaces, and real figure generation."""

from pathlib import Path
import os
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

import graph_hist as graph
from cel_res_hypr.analysis.aggregation import (
    _compute_dispersion_table,
    _compute_mean_table,
    _ensure_columns,
)
from cel_res_hypr.paths import _safe_path


@pytest.fixture
def low_resolution_saving(monkeypatch):
    save = Figure.savefig

    def small_save(self, *args, **kwargs):
        kwargs["dpi"] = 40
        return save(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", small_save)
    yield
    plt.close("all")


def test_grouping_averages_duplicate_hyperparameters_before_dispersion():
    frame = pd.DataFrame(
        dict(
            mode=["real"] * 4,
            src=["a"] * 4,
            shuffle_id=[0] * 4,
            rho_target=[0.8] * 4,
            leak=[0.6, 0.6, 0.8, 1.0],
            input_scale=[0.1] * 4,
            neuron_bias=[0.0] * 4,
            MC=[1.0, 3.0, 3.0, 5.0],
        )
    )
    mean = _compute_mean_table(frame, ["MC"])
    dispersion = _compute_dispersion_table(frame)
    expected = np.array([2.0, 3.0, 5.0])
    assert mean.iloc[0]["mean"] == pytest.approx(expected.mean())
    assert mean.iloc[0]["n_hparams"] == 3
    assert dispersion.iloc[0]["dispersion"] == pytest.approx(
        expected.std() / expected.mean()
    )


def test_plotting_and_merging_keep_their_distinct_column_policies():
    frame = pd.DataFrame(
        dict(
            mode=[" real "],
            rho_target=[0.8],
            leak=[0.6],
            input_scale=[0.1],
            wt_mag_cv=[0.5],
        )
    )
    merged = _ensure_columns(frame)
    plotted = graph._ensure_columns(frame)
    assert "wt_mag_cv" not in merged and "wt_mag_cv" in plotted
    assert merged.iloc[0]["mode"] == " real " and plotted.iloc[0]["mode"] == "real"
    assert plotted.iloc[0]["neuron_bias"] == 0.0
    assert np.isnan(plotted.iloc[0]["MC"])


def test_malformed_csv_fallback_and_versioned_paths(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("mode,MC\nreal,1\nshuffle,2,extra\n")
    frame = graph._read_combined_csv(str(path))
    assert frame["mode"].tolist() == ["real", "shuffle"]
    assert frame["MC"].tolist() == [1, 2]
    assert _safe_path(str(path)) == str(tmp_path / "malformed.v1.csv")
    (tmp_path / "malformed.v1.csv").touch()
    assert _safe_path(str(path)) == str(tmp_path / "malformed.v2.csv")


def test_legacy_global_mode_filter_still_controls_selection(
    plotting_results, monkeypatch
):
    monkeypatch.setattr(graph, "ANALYSIS_MODE_FILTER", ["real"])
    assert set(graph._filter_to_modes(plotting_results)["mode"]) == {"real"}


def test_contour_and_triptych_render_nonempty_images(
    plotting_results, tmp_path, low_resolution_saving
):
    dispersion = _compute_dispersion_table(plotting_results)
    mean = _compute_mean_table(plotting_results)
    contours = graph.plot_cv_performance_contours_2d(
        dispersion, mean, str(tmp_path), bins=12, show=False
    )
    triptych = graph.plot_cv_performance_contour_triptych(
        dispersion, mean, plotting_results, str(tmp_path), bins=12, show=False
    )
    for path in (contours, triptych):
        assert Path(path).is_file()
        image = plt.imread(path)
        assert image.shape[0] > 100 and image.shape[1] > 100
        assert image[..., :3].std() > 0.01


def test_weight_performance_plot_restores_missing_export(
    plotting_results, tmp_path, low_resolution_saving
):
    weights = plotting_results.copy()
    modes = weights["mode"].unique()
    weights["mode"] = weights["mode"].map(
        {mode: f"weight_test{i}" for i, mode in enumerate(modes)}
    )
    graph.plot_weight_gauss_mean_perf(
        _compute_dispersion_table(weights), weights, str(tmp_path), show=False
    )
    output = tmp_path / "weight_mean_perf_log3d.png"
    assert output.is_file() and plt.imread(output).std() > 0.01


def test_optional_mc_gr_plot_retains_documented_historical_background(
    plotting_results, tmp_path, monkeypatch
):
    calls = []
    scatter = plt.scatter

    def capture(x, y, *args, **kwargs):
        calls.append(np.asarray(y).copy())
        return scatter(x, y, *args, **kwargs)

    monkeypatch.setattr(plt, "scatter", capture)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    graph.plot_mc_vs_gr_all_arch(plotting_results, str(tmp_path), 0.5)
    np.testing.assert_array_equal(calls[0], plotting_results["IPC"].to_numpy())


def test_merge_script_and_module_commands_agree(plotting_results, tmp_path):
    source = tmp_path / "chunk.csv"
    plotting_results.to_csv(source, index=False)
    root = Path(__file__).parents[1]
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    for command, label in [
        (["util/merge.py"], "script"),
        (["-m", "util.merge"], "module"),
    ]:
        subprocess.run(
            [
                sys.executable,
                *command,
                "--glob-variants",
                str(source),
                "--glob-shuf",
                str(tmp_path / "absent*.csv"),
                "--out-dir",
                str(tmp_path / label),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    for name in (
        "combined.ALL.csv",
        "mean_by_group.ALL.csv",
        "dispersion_by_group.ALL.csv",
    ):
        assert (tmp_path / "script" / name).read_bytes() == (
            tmp_path / "module" / name
        ).read_bytes()
