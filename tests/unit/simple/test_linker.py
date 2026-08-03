from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.models.tracks import TracksBuilder
from pystormtracker.simple.linker import SimpleLinker, great_circle_distance_matrix


def test_simple_linker_init() -> None:
    linker = SimpleLinker(threshold=1000.0)
    assert linker.threshold == 1000.0


def test_simple_linker_append() -> None:
    linker = SimpleLinker()
    builder = TracksBuilder("msl", "min", {"msl": "Pa"})

    t0 = np.datetime64("2025-12-01T00:00:00")
    lats_1: NDArray[np.float64] = np.array([0.0])
    lons_1: NDArray[np.float64] = np.array([0.0])
    vars_1: dict[str, NDArray[np.float64]] = {"msl": np.array([1000.0])}
    step_data_1 = (t0, lats_1, lons_1, vars_1)
    linker.append(builder, step_data_1)

    t6 = np.datetime64("2025-12-01T06:00:00")
    lats_2: NDArray[np.float64] = np.array([1.0])
    lons_2: NDArray[np.float64] = np.array([1.0])
    vars_2: dict[str, NDArray[np.float64]] = {"msl": np.array([990.0])}
    step_data_2 = (t6, lats_2, lons_2, vars_2)
    linker.append(builder, step_data_2)
    tracks = builder.finish()

    assert len(tracks) == 1
    assert len(tracks[0]) == 2


def test_great_circle_distance_crosses_dateline() -> None:
    distances = great_circle_distance_matrix(
        np.array([0.0]),
        np.array([179.0]),
        np.array([0.0]),
        np.array([-179.0]),
    )

    assert distances.shape == (1, 1)
    assert distances[0, 0] == pytest.approx(222.39, rel=1e-3)


def test_great_circle_distance_clamps_identical_points() -> None:
    distances = great_circle_distance_matrix(
        np.array([90.0]),
        np.array([0.0]),
        np.array([90.0]),
        np.array([180.0]),
    )

    assert np.isfinite(distances).all()
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)
