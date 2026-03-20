from __future__ import annotations

import numpy as np

from pystormtracker.simple.kernels import (
    _numba_extrema_filter,
    _numba_get_centers,
    _numba_laplace_masked,
    _numba_remove_dup,
)


def test_numba_extrema_filter() -> None:
    # Create a 10x10 data with a clear minimum
    data = np.ones((10, 10), dtype=np.float64) * 100.0
    data[5, 5] = 90.0

    # Test for local minimum with size 3, threshold 5
    out = _numba_extrema_filter(data, size=3, threshold=5.0, is_min=True)
    assert out[5, 5] == 1.0
    assert np.sum(out) == 1.0

    # Test for local maximum (should be empty as data[5,5] is a minimum)
    out_max = _numba_extrema_filter(data, size=3, threshold=5.0, is_min=False)
    assert np.sum(out_max) == 0.0


def test_numba_extrema_filter_plateau() -> None:
    # Plateaus should be handled (rank filtering)
    data = np.ones((10, 10), dtype=np.float64) * 100.0
    data[5, 5] = 90.0
    data[5, 6] = 90.0 # Plateau

    out = _numba_extrema_filter(data, size=3, threshold=5.0, is_min=True)
    # Both 90.0 points are flagged because for both,
    # the 4th smallest element in their 3x3 window is 100.0,
    # and 100.0 - 90.0 = 10.0 > 5.0.
    # Deduplication happens later in the pipeline via _numba_remove_dup.
    assert np.sum(out) == 2.0
    assert out[5, 5] == 1.0
    assert out[5, 6] == 1.0


def test_numba_laplace_masked() -> None:
    data = np.zeros((5, 5), dtype=np.float64)
    data[2, 2] = -1.0 # Minimum

    mask = np.zeros((5, 5), dtype=np.float64)
    mask[2, 2] = 1.0

    # Laplace: up + down + left + right - 4*center
    # neighbors are 0, center is -1 -> 0 + 0 + 0 + 0 - 4*(-1) = 4
    out = _numba_laplace_masked(data, mask, is_min=True)
    assert out[2, 2] == 4.0
    assert np.sum(out) == 4.0


def test_numba_remove_dup_tie_breaking() -> None:
    # Create two duplicate intensity points
    laplacian = np.zeros((10, 10), dtype=np.float64)
    laplacian[5, 5] = 10.0
    laplacian[5, 6] = 10.0

    # Lower index wins: (5,5) should win over (5,6)
    out = _numba_remove_dup(laplacian, size=3)
    assert out[5, 5] == 1.0
    assert out[5, 6] == 0.0
    assert np.sum(out) == 1.0


def test_numba_get_centers() -> None:
    extrema = np.zeros((10, 10), dtype=np.float64)
    extrema[2, 2] = 1.0
    extrema[8, 8] = 1.0

    frame = np.random.rand(10, 10)

    r, c, vals = _numba_get_centers(extrema, frame)
    assert len(r) == 2
    assert r[0] == 2
    assert c[0] == 2
    assert r[1] == 8
    assert c[1] == 8
    assert vals[0] == frame[2, 2]
    assert vals[1] == frame[8, 8]
