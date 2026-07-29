from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from utils import fetch_era5_msl

from pystormtracker.cli import main
from pystormtracker.track import run_tracker


@pytest.fixture
def msl_data() -> str:
    return str(fetch_era5_msl())


def test_run_tracker_serial(msl_data: str, tmp_path: Path) -> None:
    output_file = tmp_path / "test_tracks.txt"
    run_tracker(
        infile=msl_data,
        varname="msl",
        outfile=str(output_file),
        mode="min",
        backend="serial",
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_main_track(msl_data: str, tmp_path: Path) -> None:
    output_file = tmp_path / "main_output.txt"
    test_args = [
        "stormtracker",
        "track",
        "-i",
        msl_data,
        "-v",
        "msl",
        "-o",
        str(output_file),
        "-n",
        "2",
        "-b",
        "serial",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_main_help() -> None:
    test_args = ["stormtracker", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
