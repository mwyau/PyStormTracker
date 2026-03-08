from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-all", action="store_true", default=False, help="run all tests"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_slow = config.getoption("--run-slow")
    run_all = config.getoption("--run-all")

    if run_all:
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow or --run-all option to run")
    skip_fast = pytest.mark.skip(reason="skipped because --run-slow was given")

    for item in items:
        if "slow" in item.keywords:
            if not run_slow:
                item.add_marker(skip_slow)
        elif run_slow:
            # If only --run-slow is given, skip the fast tests
            item.add_marker(skip_fast)
