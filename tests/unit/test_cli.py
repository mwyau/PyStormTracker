from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pystormtracker import HodgesTracker, SimpleTracker, __version__
from pystormtracker import compare as compare_command
from pystormtracker import convert as convert_command
from pystormtracker import sample as sample_command
from pystormtracker.cli import main
from pystormtracker.models.tracks import Tracks, TracksMetadata
from pystormtracker.track import setup_parser


def _empty_tracks() -> Tracks:
    return Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))


@pytest.fixture
def msl_data(tmp_path: Path) -> str:
    """Create a tiny local MSL input for CLI smoke tests."""
    values = np.full((2, 3, 4), 100000.0, dtype=np.float32)
    values[:, 1, 2] = 98000.0
    data = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2025-12-01T00", "2025-12-01T06"], dtype="datetime64[h]"),
            "latitude": [90.0, 0.0, -90.0],
            "longitude": [0.0, 90.0, 180.0, 270.0],
        },
        name="msl",
    )
    path = tmp_path / "synthetic_msl.nc"
    data.to_dataset().to_netcdf(path, engine="h5netcdf")
    return str(path)


def test_cli_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["stormtracker", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"stormtracker {__version__}\n"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "stormtracker",
            "-vv",
            "track",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "out.json",
        ],
        [
            "stormtracker",
            "track",
            "-vv",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "out.json",
        ],
    ],
)
def test_cli_accepts_verbosity_before_and_after_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with patch("pystormtracker.cli.track.main") as track_main:
        main()

    assert track_main.call_args is not None
    assert track_main.call_args.args[0].verbose == 2


@pytest.mark.parametrize("command", ["track", "sample", "compare", "convert"])
def test_cli_version_is_available_after_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["stormtracker", command, "-V"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"stormtracker {command} {__version__}\n"


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
        "--variable",
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
        ["track", "-i", "input.nc", "--variable", "msl", "-o", "output.trackjson"]
    )
    assert args.format == "auto"
    assert args.detection_mode == "auto"
    assert args.no_progress is False


def test_track_parser_accepts_no_progress() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    args = parser.parse_args(
        [
            "track",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "output.trackjson",
            "--no-progress",
        ]
    )
    assert args.no_progress is True


def test_hodges_execution_options_parse_independently() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    args = parser.parse_args(
        [
            "track",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "output.trackjson",
            "--algorithm",
            "hodges",
            "--backend",
            "dask",
            "--frame-workers",
            "1",
            "--sht-threads",
            "16",
            "--mge-workers",
            "16",
        ]
    )

    assert args.frame_workers == 1
    assert args.sht_threads == 16
    assert args.mge_workers == 16
    assert args.workers is None


def test_hodges_cli_forwards_execution_options() -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "--variable",
        "msl",
        "-o",
        "unused.txt",
        "--algorithm",
        "hodges",
        "--backend",
        "dask",
        "--frame-workers",
        "1",
        "--sht-threads",
        "16",
        "--mge-workers",
        "16",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.HodgesTracker") as mock_hodges,
    ):
        instance = mock_hodges.return_value
        instance.track.return_value = _empty_tracks()
        main()

    assert mock_hodges.call_args is not None
    assert mock_hodges.call_args.kwargs["frame_workers"] == 1
    assert mock_hodges.call_args.kwargs["sht_threads"] == 16
    assert mock_hodges.call_args.kwargs["mge_workers"] == 16


def test_hodges_cli_rejects_generic_workers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "--variable",
        "msl",
        "-o",
        "unused.txt",
        "--algorithm",
        "hodges",
        "--workers",
        "4",
    ]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "--workers is not supported with Hodges" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("6h", np.timedelta64(6, "h")),
        ("30m", np.timedelta64(30, "m")),
        ("1D", np.timedelta64(1, "D")),
    ],
)
def test_track_parser_accepts_documented_time_step_syntax(
    text: str, expected: np.timedelta64
) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)

    args = parser.parse_args(
        [
            "track",
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "output.trackjson",
            "--time-step",
            text,
        ]
    )

    assert args.time_step == expected


@pytest.mark.parametrize("text", ["PT6H", "6 hours", "0h", "6", "6d"])
def test_track_parser_rejects_undocumented_time_step_syntax(text: str) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "track",
                "-i",
                "input.nc",
                "--variable",
                "msl",
                "-o",
                "output.trackjson",
                "--time-step",
                text,
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize("command", ["track", "sample", "compare", "convert"])
def test_subcommand_shared_cli_options_and_long_variable(command: str) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    sample_command.setup_parser(subparsers)
    convert_command.setup_parser(subparsers)
    compare_command.setup_parser(subparsers)

    required_args = {
        "track": ["-i", "input.nc", "--variable", "msl", "-o", "out.json"],
        "sample": [
            "-i",
            "tracks.json",
            "-d",
            "data.nc",
            "--variable",
            "msl",
            "-o",
            "out.json",
        ],
        "compare": [
            "-r",
            "reference.json",
            "-c",
            "candidate.json",
            "--variable",
            "msl",
        ],
        "convert": [
            "-i",
            "input.json",
            "-o",
            "out.json",
            "--variable",
            "msl",
        ],
    }
    args = parser.parse_args([command, "-vv", *required_args[command]])

    assert args.verbose == 2
    assert args.variable == "msl"


@pytest.mark.parametrize("command", ["track", "sample", "compare", "convert"])
def test_short_v_is_not_a_variable_alias(command: str) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)
    sample_command.setup_parser(subparsers)
    convert_command.setup_parser(subparsers)
    compare_command.setup_parser(subparsers)

    required_args = {
        "track": [
            "-i",
            "input.nc",
            "--variable",
            "msl",
            "-o",
            "out.json",
            "-v",
            "vo",
        ],
        "sample": [
            "-i",
            "tracks.json",
            "-d",
            "data.nc",
            "--variable",
            "msl",
            "-o",
            "out.json",
            "-v",
            "vo",
        ],
        "compare": [
            "-r",
            "reference.json",
            "-c",
            "candidate.json",
            "--variable",
            "msl",
            "-v",
            "vo",
        ],
        "convert": [
            "-i",
            "input.json",
            "-o",
            "out.json",
            "--variable",
            "msl",
            "-v",
            "vo",
        ],
    }

    with pytest.raises(SystemExit):
        parser.parse_args([command, *required_args[command]])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--n-frames", "0"),
        ("--workers", "0"),
        ("--segment-frames", "0"),
        ("--extent", "1,0,-1,1"),
        ("--extent", "nan,1,-1,1"),
        ("--lmin", "-1"),
        ("--lmax", "-1"),
        ("--stereo-grid-spacing-km", "nan"),
        ("--object-threshold", "inf"),
        ("--feature-threshold", "inf"),
    ],
)
def test_track_rejects_invalid_cli_values(option: str, value: str) -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "--variable",
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
        "--variable",
        "msl",
        "-o",
        "unused.txt",
        "--algorithm",
        "healpix",
        "--lmin",
        "3",
        "--lmax",
        "21",
        "--taper-points",
        "4",
        "--nside",
        "16",
        "--feature-refinement",
        "quadratic",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.HealpixTracker") as mock_healpix,
    ):
        instance = mock_healpix.return_value
        instance.track.return_value = _empty_tracks()
        main()

    assert mock_healpix.call_args.kwargs["lmin"] == 3
    assert mock_healpix.call_args.kwargs["lmax"] == 21
    assert mock_healpix.call_args.kwargs["taper_points"] == 4
    assert mock_healpix.call_args.kwargs["nside"] == 16
    assert mock_healpix.call_args.kwargs["feature_refinement"] == "quadratic"


def test_filter_bounds_must_be_supplied_together() -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "--variable",
        "msl",
        "-o",
        "unused.txt",
        "--lmin",
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
    assert tracker.lmin is None
    assert tracker.lmax is None
    assert tracker.taper_points == 0
    assert tracker.feature_refinement == "grid"


def test_hodges_tracker_defaults() -> None:
    tracker = HodgesTracker()
    assert tracker.lmin is None
    assert tracker.lmax is None
    assert tracker.taper_points == 0
    assert tracker.feature_refinement == "bspline"
    assert tracker.track_smoopy_optimization_scale == 1.0


def test_hodges_tracker_accepts_explicit_smoopy_scale() -> None:
    tracker = HodgesTracker(track_smoopy_optimization_scale=0.01)
    assert tracker.track_smoopy_optimization_scale == 0.01


@pytest.mark.parametrize("scale", [0.0, -0.01, float("inf"), float("nan")])
def test_hodges_tracker_rejects_invalid_smoopy_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="track_smoopy_optimization_scale"):
        HodgesTracker(track_smoopy_optimization_scale=scale)


def test_hodges_cli_uses_source_compatible_detection_defaults() -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "--variable",
        "msl",
        "-o",
        "unused.txt",
        "--algorithm",
        "hodges",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.HodgesTracker") as mock_hodges,
    ):
        instance = mock_hodges.return_value
        instance.track.return_value = _empty_tracks()
        main()

    assert mock_hodges.call_args.kwargs["spectral_taper"] == 1.0
    assert mock_hodges.call_args.kwargs["feature_refinement"] == "bspline"


def test_runtime_validation_reports_clean_cli_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_args = [
        "stormtracker",
        "track",
        "-i",
        "unused.nc",
        "--variable",
        "msl",
        "-o",
        "unused.txt",
        "--algorithm",
        "hodges",
        "--dmax-zones",
        "nonexistent_file_path.dat",
    ]
    with (
        patch.object(sys, "argv", test_args),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert (
        "No such file or directory" in captured.err
        or "nonexistent_file_path.dat" in captured.err
    )
    assert "Traceback" not in captured.err
