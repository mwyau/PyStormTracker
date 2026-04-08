from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from utils import fetch_era5_msl

from pystormtracker.cli import main, parse_args, run_tracker


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


def test_run_tracker_dask(msl_data: str, tmp_path: Path) -> None:
    output_file = tmp_path / "test_tracks_dask.txt"
    run_tracker(
        infile=msl_data,
        varname="msl",
        outfile=str(output_file),
        mode="min",
        backend="dask",
        n_workers=2,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_parse_args() -> None:
    test_args = [
        "stormtracker",
        "-i",
        "input.nc",
        "-v",
        "msl",
        "-o",
        "output.txt",
        "-n",
        "10",
        "-m",
        "max",
        "-b",
        "serial",
        "-w",
        "4",
    ]
    with patch("sys.argv", test_args):
        args = parse_args()
        assert args.input == "input.nc"
        assert args.var == "msl"
        assert args.output == "output.txt"
        assert args.num == 10
        assert args.algorithm == "simple"
        assert args.mode == "max"
        assert args.backend == "serial"
        assert args.workers == 4


def test_main(msl_data: str, tmp_path: Path) -> None:
    output_file = tmp_path / "main_output.txt"
    test_args = [
        "stormtracker",
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
    with patch("sys.argv", test_args):
        main()

    assert output_file.exists()
    assert output_file.stat().st_size > 0
