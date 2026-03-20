from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pystormtracker.cli import main
from pystormtracker.io.imilast import read_imilast
from pystormtracker.utils.data_utils import fetch_era5_vo850


def run_command_direct(cmd_args: list[str]) -> None:
    """Utility to run the tracker directly via main."""
    with patch.object(sys, "argv", ["stormtracker", *cmd_args]):
        main()


@pytest.fixture(scope="module")
def test_data_vo() -> str:
    """Download VO test data once per module."""
    return str(fetch_era5_vo850(resolution="2.5x2.5"))


@pytest.mark.integration
def test_hodges_serial_integration(test_data_vo: str, tmp_path: Path) -> None:
    """Basic integration test for the Hodges tracker via CLI."""
    out_file = tmp_path / "hodges_tracks.txt"

    # Use vo which is more robust for tracking in short time slices
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
        "10",  # Just 10 steps for speed
        "--format",
        "imilast",
    ]

    run_command_direct(args)

    assert out_file.exists()
    tracks = read_imilast(out_file)
    assert len(tracks) > 0
    assert any(len(tr) >= 2 for tr in tracks)


@pytest.mark.integration
def test_hodges_output_format(test_data_vo: str, tmp_path: Path) -> None:
    """Test the Hodges (TRACK) ASCII output format."""
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
