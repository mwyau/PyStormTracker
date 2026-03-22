from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pystormtracker.cli import main
from pystormtracker.io.imilast import read_imilast
from pystormtracker.utils.data import fetch_era5_vo850


def run_command_direct(cmd_args: list[str]) -> None:
    """Utility to run the tracker directly via main."""
    print(f"\nRunning command: stormtracker {' '.join(cmd_args)}", flush=True)
    try:
        with patch.object(sys, "argv", ["stormtracker", *cmd_args]):
            main()
    except BaseException as e:
        print(f"Command failed with {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


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
    request: pytest.FixtureRequest,
    steps: int | None,
) -> int | None:
    """Skip full tests locally if not in CI."""
    if steps is None and not os.environ.get("GITHUB_ACTIONS"):
        pytest.skip("Full Hodges integration tests are temporarily disabled locally.")

    return steps


@pytest.mark.integration
def test_hodges_serial_integration(
    test_data_vo: str, tmp_path: Path, hodges_config: int | None
) -> None:
    """Basic integration test for the Hodges tracker via CLI."""
    steps = hodges_config
    out_file = tmp_path / f"hodges_tracks_{steps or 'full'}.txt"

    args = [
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


@pytest.mark.integration
def test_hodges_output_format(test_data_vo: str, tmp_path: Path) -> None:
    """Test the Hodges (TRACK) ASCII output format (Short)."""
    out_file = tmp_path / "hodges_native.txt"

    args = [
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
        "-n",
        "5",
        "--format",
        "hodges",
    ]

    run_command_direct(args)

    assert out_file.exists()
    with open(out_file) as f:
        content = f.read()
        assert "TRACK_NUM" in content
        assert "TRACK_ID" in content
        assert "POINT_NUM" in content
