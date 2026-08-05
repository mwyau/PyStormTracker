from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pystormtracker.cli import main
from pystormtracker.models.tracks import Tracks, TracksMetadata


def _empty_tracks() -> Tracks:
    return Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))


def test_cli_load_dat_files(tmp_path: Path) -> None:
    """Test parsing of legacy zone.dat and adapt.dat from CLI."""
    zone_file = tmp_path / "zone.dat"
    zone_content = (
        "3\n"
        "  0.0  360.0  -90.0  -20.0  6.5\n"
        "  0.0  360.0  -20.0   20.0  3.0\n"
        "  0.0  360.0   20.0   90.0  6.5"
    )
    zone_file.write_text(zone_content)

    adapt_file = tmp_path / "adapt.dat"
    adapt_content = "1.0 1.0\n2.0 0.3\n5.0 0.1\n8.0 0.0"
    adapt_file.write_text(adapt_content)

    test_args = [
        "stormtracker",
        "track",
        "-i",
        "dummy.nc",
        "-v",
        "msl",
        "-o",
        "output.txt",
        "--algorithm",
        "hodges",
        "--zone-file",
        str(zone_file),
        "--adapt-file",
        str(adapt_file),
    ]

    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.HodgesTracker") as mock_tracker_cls,
    ):
        mock_instance = mock_tracker_cls.return_value
        mock_instance.track.return_value = _empty_tracks()
        main()

    mock_tracker_cls.assert_called_once()
    kwargs = mock_tracker_cls.call_args.kwargs

    zones_parsed = kwargs["zones"]
    assert zones_parsed.shape == (3, 5)
    assert zones_parsed[1, 4] == 3.0

    adapt_parsed = kwargs["adapt_params"]
    assert adapt_parsed.shape == (2, 4)
    assert np.array_equal(adapt_parsed[0], [1.0, 2.0, 5.0, 8.0])
    assert np.array_equal(adapt_parsed[1], [1.0, 0.3, 0.1, 0.0])


def test_cli_load_dat_json(tmp_path: Path) -> None:
    """Test parsing of zones and adapt-params from JSON strings."""
    zones_json = "[[0.0, 360.0, -90.0, 90.0, 10.0]]"
    adapt_json = "[[1.0, 2.0, 3.0, 4.0], [1.0, 0.5, 0.2, 0.1]]"

    test_args = [
        "stormtracker",
        "track",
        "-i",
        "dummy.nc",
        "-v",
        "msl",
        "-o",
        "output.txt",
        "--algorithm",
        "hodges",
        "--zones",
        zones_json,
        "--adapt-params",
        adapt_json,
    ]

    with (
        patch.object(sys, "argv", test_args),
        patch("pystormtracker.track.HodgesTracker") as mock_tracker_cls,
    ):
        mock_instance = mock_tracker_cls.return_value
        mock_instance.track.return_value = _empty_tracks()
        main()

    mock_tracker_cls.assert_called_once()
    kwargs = mock_tracker_cls.call_args.kwargs

    assert np.array_equal(kwargs["zones"], np.array(json.loads(zones_json)))
    assert np.array_equal(kwargs["adapt_params"], np.array(json.loads(adapt_json)))
