from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.linker import SimpleLinker


def test_simple_linker_init() -> None:
    linker = SimpleLinker(threshold=1000.0)
    assert linker.threshold == 1000.0


def test_simple_linker_append() -> None:
    linker = SimpleLinker()
    tracks = Tracks()

    t0 = np.datetime64("2025-12-01T00:00:00")
    lats_1: NDArray[np.float64] = np.array([0.0])
    lons_1: NDArray[np.float64] = np.array([0.0])
    vars_1: dict[str, NDArray[np.float64]] = {"msl": np.array([1000.0])}
    step_data_1 = (t0, lats_1, lons_1, vars_1)
    linker.append(tracks, step_data_1)

    assert len(tracks) == 1
    assert tracks.time_range is not None
    assert tracks.time_range.start == t0
    assert tracks.time_range.end == t0

    t6 = np.datetime64("2025-12-01T06:00:00")
    lats_2: NDArray[np.float64] = np.array([1.0])
    lons_2: NDArray[np.float64] = np.array([1.0])
    vars_2: dict[str, NDArray[np.float64]] = {"msl": np.array([990.0])}
    step_data_2 = (t6, lats_2, lons_2, vars_2)
    linker.append(tracks, step_data_2)

    assert len(tracks) == 1
    assert tracks.time_range.end == t6
    assert tracks.time_range.step == np.timedelta64(6, "h")
    assert len(tracks[0]) == 2
