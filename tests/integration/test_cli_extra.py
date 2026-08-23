from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pytest

from pystormtracker import compare, convert, sample, track
from pystormtracker.io.trackjson import read_trackjson
from pystormtracker.models.tracks import Tracks
from tests.utils import get_integration_msl_path


def run_command_direct(cmd_args: list[str]) -> Tracks | None:
    """Utility to run the CLI subcommands directly."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    track.setup_parser(subparsers)
    sample.setup_parser(subparsers)
    convert.setup_parser(subparsers)
    compare.setup_parser(subparsers)

    args = parser.parse_args(cmd_args)
    if hasattr(args, "func"):
        return cast(Tracks, args.func(args))
    return None


@pytest.fixture(scope="module")
def sample_tracks_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a small track file for integration testing."""
    msl_data = get_integration_msl_path()
    out_dir = tmp_path_factory.mktemp("cli_extra")
    out_file = out_dir / "tracks.trackjson"

    args = [
        "track",
        "-i",
        str(msl_data),
        "--variable",
        "msl",
        "-o",
        str(out_file),
        "-n",
        "5",
        "-f",
        "json",
    ]
    run_command_direct(args)
    return out_file


@pytest.mark.integration
def test_cli_sample(sample_tracks_file: Path, tmp_path: Path) -> None:
    """Test 'stormtracker sample' command."""
    msl_data = get_integration_msl_path()
    out_file = tmp_path / "sampled.trackjson"

    args = [
        "sample",
        "-i",
        str(sample_tracks_file),
        "-d",
        str(msl_data),
        "--variable",
        "msl",
        "-o",
        str(out_file),
        "-m",
        "nearest",
    ]
    run_command_direct(args)

    assert out_file.exists()
    tracks = read_trackjson(out_file)
    assert "msl" in tracks.variables


@pytest.mark.integration
def test_cli_compare(sample_tracks_file: Path, tmp_path: Path) -> None:
    """Test the ``stormtracker compare`` command with option aliases."""
    out_file = tmp_path / "matched.trackjson"
    report_file = tmp_path / "report.json"

    args = [
        "compare",
        "-r",
        str(sample_tracks_file),
        "-c",
        str(sample_tracks_file),
        "-s",
        "2.0",
        "-l",
        "0.6",
        "--variable",
        "msl",
        "-m",
        "max",
        "-o",
        str(report_file),
        "-M",
        str(out_file),
    ]
    run_command_direct(args)

    assert out_file.exists()
    assert report_file.exists()
    tracks = read_trackjson(out_file)
    assert len(tracks) > 0
