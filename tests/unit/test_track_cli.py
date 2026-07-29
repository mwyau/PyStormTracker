from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from utils import fetch_era5_msl

from pystormtracker.cli import main
from pystormtracker.models.tracks import Tracks
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


def test_main_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(sys, "argv", ["stormtracker"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 0
    assert "commands:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--num", "0"),
        ("--workers", "0"),
        ("--chunk-size", "0"),
        ("--extent", "1,0,-1,1"),
        ("--filter-range", "42-5"),
    ],
)
def test_track_rejects_invalid_cli_values(option: str, value: str) -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "-v",
        "msl",
        "-o",
        "unused.txt",
        option,
        value,
    ]
    with (
        patch.object(sys, "argv", test_args),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 2


def test_filter_range_enables_simple_filter() -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "-v",
        "msl",
        "-o",
        "unused.txt",
        "--filter-range",
        "3-21",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.run_tracker", return_value=Tracks()) as mocked_run,
    ):
        main()

    assert mocked_run.call_args.kwargs["filter"] is True
    assert mocked_run.call_args.kwargs["lmin"] == 3
    assert mocked_run.call_args.kwargs["lmax"] == 21


def test_runtime_validation_reports_clean_cli_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "-v",
        "msl",
        "-o",
        "unused.txt",
        "--zones",
        "not-json",
    ]
    with (
        patch.object(sys, "argv", test_args),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "invalid zones JSON" in captured.err
    assert "Traceback" not in captured.err
