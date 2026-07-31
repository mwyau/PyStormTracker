from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Add tests directory to sys.path to allow importing utils
sys.path.insert(0, str(Path(__file__).parent.absolute()))


@pytest.fixture
def reduced_gaussian_data() -> xr.DataArray:
    """Create a small deterministic reduced Gaussian field."""
    pl = np.array([4, 8, 12, 16, 16, 12, 8, 4], dtype=np.int32)
    values = np.sin(np.arange(int(pl.sum()), dtype=np.float64) / 7.0)
    return xr.DataArray(
        values[np.newaxis, :],
        dims=("time", "values"),
        coords={"time": [np.datetime64("2000-01-01")]},
        name="msl",
        attrs={"GRIB_gridType": "reduced_gg", "GRIB_pl": pl.tolist()},
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run ONLY integration tests",
    )
    parser.addoption(
        "--run-all", action="store_true", default=False, help="run all tests"
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration "
        "(skipped by default, run with --run-integration)",
    )
    config.addinivalue_line("markers", "slow: marks tests excluded from normal runs")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_integration = config.getoption("--run-integration")
    run_all = config.getoption("--run-all")
    run_slow = config.getoption("--run-slow")

    if run_all:
        return

    for item in items:
        is_integration = "integration" in item.keywords
        is_slow = "slow" in item.keywords
        if run_integration:
            if not is_integration:
                item.add_marker(
                    pytest.mark.skip(reason="only integration tests requested")
                )
        elif is_integration:
            item.add_marker(pytest.mark.skip(reason="integration test skipped"))
        if is_slow and not run_slow:
            item.add_marker(pytest.mark.skip(reason="slow test skipped"))
