from __future__ import annotations

import pytest


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration (skipped by default, run with --run-integration)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_integration = config.getoption("--run-integration")
    run_all = config.getoption("--run-all")

    if run_all:
        return

    for item in items:
        is_integration = "integration" in item.keywords
        if run_integration:
            if not is_integration:
                item.add_marker(
                    pytest.mark.skip(reason="only integration tests requested")
                )
        elif is_integration:
            item.add_marker(pytest.mark.skip(reason="integration test skipped"))
