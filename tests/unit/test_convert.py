from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pystormtracker.convert import main
from pystormtracker.io.format import load_tracks
from pystormtracker.io.imilast import write_imilast
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks


def _tracks() -> Tracks:
    tracks = Tracks(track_type="msl")
    tracks.add_track(
        [
            Center(
                time=np.datetime64("2020-01-01T00:00"),
                lat=50.0,
                lon=0.0,
                vars={"msl": 100_000.0},
            ),
            Center(
                time=np.datetime64("2020-01-01T06:00"),
                lat=51.0,
                lon=1.0,
                vars={"msl": 99_000.0},
            ),
        ]
    )
    return tracks


def test_convert_infers_trackjson_and_geojson_formats(tmp_path: Path) -> None:
    source = tmp_path / "tracks.json"
    target = tmp_path / "tracks.geojson"
    _tracks().write(source)

    main(
        argparse.Namespace(
            input=str(source),
            output=str(target),
            in_format=None,
            out_format=None,
            var=None,
        )
    )

    converted = load_tracks(target)
    assert converted.track_type == "msl"
    np.testing.assert_allclose(converted.vars["msl"], [100_000.0, 99_000.0])
<<<<<<< Updated upstream


def test_convert_var_override_renames_single_custom_input_variable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tracks.txt"
    target = tmp_path / "tracks.trackjson"
    source_tracks = Tracks(track_type="unknown")
    source_tracks.add_track(
        [
            Center(
                time=np.datetime64("2020-01-01T00:00"),
                lat=50.0,
                lon=0.0,
                vars={"PRESSURE": 100_000.0},
            ),
            Center(
                time=np.datetime64("2020-01-01T06:00"),
                lat=51.0,
                lon=1.0,
                vars={"PRESSURE": 99_000.0},
            ),
        ]
    )
    write_imilast(source_tracks, source)

    main(
        argparse.Namespace(
            input=str(source),
            output=str(target),
            in_format="imilast",
            out_format="json",
            var="msl",
        )
    )

    converted = load_tracks(target)
    assert converted.track_type == "msl"
    assert set(converted.vars) == {"msl"}
    np.testing.assert_allclose(converted.vars["msl"], [100_000.0, 99_000.0])
=======
>>>>>>> Stashed changes
