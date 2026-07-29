from __future__ import annotations

from pathlib import Path

import pytest
from utils import fetch_era5_vo850

from pystormtracker.io.imilast import read_imilast
from pystormtracker.models.tracks import Tracks


def run_command_direct(cmd_args: list[str]) -> Tracks | None:
    """Utility to run the tracker directly and return results."""
    import argparse

    from pystormtracker import compare, convert, sample, track

    # Prepend 'track' if missing
    if cmd_args and cmd_args[0] not in ["track", "sample", "convert", "compare"]:
        cmd_args = ["track", *cmd_args]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    track.setup_parser(subparsers)
    sample.setup_parser(subparsers)
    convert.setup_parser(subparsers)
    compare.setup_parser(subparsers)

    from typing import cast

    args = parser.parse_args(cmd_args)
    if hasattr(args, "func"):
        return cast(Tracks, args.func(args))
    return None


@pytest.fixture(scope="module")
def test_data_vo() -> str:
    """Download VO test data once per module."""
    return str(fetch_era5_vo850(resolution="2.5x2.5"))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Custom parameterization for Hodges integration tests."""
    if "steps" in metafunc.fixturenames:
        # 60 steps for 'short', None for 'full'
        raw_params = [
            (60, "short"),
            (None, "full"),
        ]

        params = [pytest.param(p[0], id=p[1]) for p in raw_params]
        metafunc.parametrize("steps", params, scope="module")


@pytest.fixture(scope="module")
def hodges_config(
    steps: int | None,
) -> int | None:
    """Return the requested Hodges integration-test length."""
    return steps


@pytest.mark.integration
def test_hodges_serial_integration(
    test_data_vo: str, tmp_path: Path, hodges_config: int | None
) -> None:
    """Basic integration test for the Hodges tracker via CLI."""
    steps = hodges_config
    out_file = tmp_path / f"hodges_tracks_{steps or 'full'}.txt"

    args = [
        "track",
        "-i",
        test_data_vo,
        "-v",
        "vo",
        "-m",
        "max",
        "-t",
        "1.0e-4",
        "-o",
        str(out_file),
        "-a",
        "hodges",
        "--format",
        "imilast",
    ]

    if steps:
        args.extend(["-n", str(steps)])

    run_command_direct(args)

    assert out_file.exists()
    tracks = read_imilast(out_file)
    assert len(tracks) > 0
    assert any(len(tr) >= 2 for tr in tracks)
