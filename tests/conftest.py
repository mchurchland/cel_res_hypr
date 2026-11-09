"""Deterministic synthetic result tables for analysis and plotting checks."""

import pytest

from tools.plot_validation import plotting_fixture


@pytest.fixture
def plotting_results():
    return plotting_fixture(repeats=4)
