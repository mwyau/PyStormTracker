from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run ONLY slow tests"
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

    for item in items:
        is_slow = "slow" in item.keywords
        if run_slow:
            if not is_slow:
                item.add_marker(pytest.mark.skip(reason="only slow tests requested"))
        elif is_slow:
            item.add_marker(pytest.mark.skip(reason="slow test skipped by default"))
