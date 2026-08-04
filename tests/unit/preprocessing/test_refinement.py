from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.preprocessing.refinement import subgrid_refine


def test_subgrid_refine_preserves_negative_longitudes() -> None:
    frame = np.array([[0.0, 0.5, 0.0], [0.6, 1.0, 0.4], [0.0, 0.5, 0.0]])
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([-180.0, -179.0, -178.0])

    refined_lat, refined_lon, refined_value = subgrid_refine(
        frame, 1, 1, lat, lon, periodic_x=True
    )

    assert refined_lat == 11.0
    assert refined_lon == pytest.approx(-179.1)
    assert refined_value > frame[1, 1]


def test_subgrid_refine_keeps_projected_x_nonperiodic() -> None:
    frame = np.array([[0.0, 0.5, 0.0], [0.6, 1.0, 0.4], [0.0, 0.5, 0.0]])
    y = np.array([-100.0, 0.0, 100.0])
    x = np.array([-100.0, 0.0, 100.0])

    refined_y, refined_x, refined_value = subgrid_refine(
        frame, 1, 1, y, x, periodic_x=False
    )

    assert refined_y == 0.0
    assert refined_x == pytest.approx(-10.0)
    assert refined_value > frame[1, 1]


def test_subgrid_refine_does_not_wrap_projected_boundary() -> None:
    frame = np.ones((3, 3), dtype=np.float64)
    y = np.array([-100.0, 0.0, 100.0])
    x = np.array([-100.0, 0.0, 100.0])

    refined = subgrid_refine(frame, 1, 0, y, x, periodic_x=False)

    assert refined == (0.0, -100.0, 1.0)
