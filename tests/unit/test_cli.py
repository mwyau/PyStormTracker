from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from utils import fetch_era5_msl

from pystormtracker import HodgesTracker, SimpleTracker, __version__
from pystormtracker.cli import main
from pystormtracker.models.tracks import Tracks, TracksMetadata
from pystormtracker.track import setup_parser


def _empty_tracks() -> Tracks:
    return Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))


@pytest.fixture
def msl_data() -> str:
    return str(fetch_era5_msl())


def test_cli_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["stormtracker", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"stormtracker {__version__}\n"


def test_tracker_serial(msl_data: str, tmp_path: Path) -> None:
    output_file = tmp_path / "test_tracks.txt"
    tracker = SimpleTracker(backend="serial")
    tracks = tracker.track(data=msl_data, variable="msl", detection_mode="min")
    tracks.write(output_file)

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


def test_track_parser_uses_automatic_format_and_mode_defaults() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    args = parser.parse_args(
        ["track", "-i", "input.nc", "-v", "msl", "-o", "output.trackjson"]
    )
    assert args.format == "auto"
    assert args.detection_mode == "auto"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--num", "0"),
        ("--workers", "0"),
        ("--chunk-size", "0"),
        ("--extent", "1,0,-1,1"),
        ("--extent", "nan,1,-1,1"),
        ("--filter-lmin", "-1"),
        ("--filter-lmax", "-1"),
        ("--stereo-grid-spacing-km", "nan"),
        ("--intensity-threshold", "inf"),
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


def test_filter_bounds_are_forwarded_without_algorithm_defaults() -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "-v",
        "msl",
        "-o",
        "unused.txt",
        "--projection",
        "healpix",
        "--filter-lmin",
        "3",
        "--filter-lmax",
        "21",
        "--taper-points",
        "4",
        "--nside",
        "16",
        "--feature-point-method",
        "quadratic",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.HealpixTracker") as mock_healpix,
    ):
        instance = mock_healpix.return_value
        instance.track.return_value = _empty_tracks()
        main()

    assert mock_healpix.call_args.kwargs["filter_lmin"] == 3
    assert mock_healpix.call_args.kwargs["filter_lmax"] == 21
    assert mock_healpix.call_args.kwargs["taper_points"] == 4
    assert mock_healpix.call_args.kwargs["nside"] == 16
    assert mock_healpix.call_args.kwargs["feature_point_method"] == "quadratic"


def test_filter_bounds_must_be_supplied_together() -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "-v",
        "msl",
        "-o",
        "unused.txt",
        "--filter-lmin",
        "3",
    ]
    with (
        patch.object(sys, "argv", test_args),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 2


def test_simple_tracker_defaults() -> None:
    tracker = SimpleTracker()
    assert tracker.filter_lmin is None
    assert tracker.filter_lmax is None
    assert tracker.taper_points == 0
    assert tracker.feature_point_method == "grid"


def test_hodges_tracker_defaults() -> None:
    tracker = HodgesTracker()
    assert tracker.filter_lmin is None
    assert tracker.filter_lmax is None
    assert tracker.taper_points == 0
    assert tracker.feature_point_method == "quadratic"


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
        "--algorithm",
        "hodges",
        "--dmax-zones",
        "not-json",
    ]
    with (
        patch.object(sys, "argv", test_args),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "invalid dmax_zones JSON" in captured.err
    assert "Traceback" not in captured.err
