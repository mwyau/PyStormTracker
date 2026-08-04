from __future__ import annotations

from pathlib import Path

import numpy as np

from pystormtracker.io.imilast import read_imilast, write_imilast
from pystormtracker.models.tracks import Tracks, TracksBuilder, TracksMetadata


def _tracks() -> Tracks:
    builder = TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    builder.add_track(
        10,
        [np.datetime64("2025-12-01T00:00"), np.datetime64("2025-12-01T06:00")],
        [10.0, 11.0],
        [190.0, 20.0],
        {"msl": [100000.0, 99000.0]},
    )
    return builder.finish()


def test_imilast_round_trip_normalizes_longitudes(tmp_path: Path) -> None:
    original = _tracks()
    output = tmp_path / "tracks.txt"
    write_imilast(original, output, decimal_places=4)
    loaded = read_imilast(output)
    assert loaded.ids.tolist() == [10]
    np.testing.assert_allclose(loaded.lons, [-170.0, 20.0], atol=1e-6)
    assert loaded.primary_var == "MSL"
    np.testing.assert_allclose(loaded.variables["MSL"], original.variables["msl"])
    np.testing.assert_array_equal(loaded.times, original.times)


def test_imilast_empty_output(tmp_path: Path) -> None:
    builder = TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    output = tmp_path / "empty.txt"
    write_imilast(builder.finish(), output)
    assert output.exists()
