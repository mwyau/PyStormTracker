from __future__ import annotations

import numpy as np

from pystormtracker.hodges.linker import HodgesLinker


def test_hodges_linker_init() -> None:
    linker = HodgesLinker(w1=0.5, w2=0.5, dmax=10.0)
    assert linker.w1 == 0.5
    assert linker.w2 == 0.5
    assert linker.dmax == 10.0


def test_hodges_linker_link_straight() -> None:
    linker = HodgesLinker(
        zones=np.zeros((0, 5), dtype=np.float64),
        adapt_thresholds=np.zeros(0, dtype=np.float64),
        adapt_values=np.zeros(0, dtype=np.float64),
    )

    # Create two detections moving in a straight line
    # T0: (0,0) and (10,10)
    # T1: (0,1) and (10,11)
    # T2: (0,2) and (10,12)

    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")
    t2 = np.datetime64("2025-12-01T12:00:00")

    detections = [
        (
            t0,
            np.array([0.0, 10.0]),
            np.array([0.0, 10.0]),
            {"msl": np.array([1000.0, 1000.0])},
        ),
        (
            t1,
            np.array([0.0, 10.0]),
            np.array([1.0, 11.0]),
            {"msl": np.array([990.0, 990.0])},
        ),
        (
            t2,
            np.array([0.0, 10.0]),
            np.array([2.0, 12.0]),
            {"msl": np.array([980.0, 980.0])},
        ),
    ]

    tracks = linker.link(detections)

    assert len(tracks) == 2
    # Verify track 1
    tr1 = tracks[0]
    assert len(tr1) == 3
    assert tr1[0].lat == 0.0
    assert tr1[1].lat == 0.0
    assert tr1[2].lat == 0.0

    # Verify track 2
    tr2 = tracks[1]
    assert len(tr2) == 3
    assert tr2[0].lat == 10.0
    assert tr2[1].lat == 10.0
    assert tr2[2].lat == 10.0


def test_hodges_linker_link_crossing() -> None:
    """
    Test that MGE correctly resolves track crossing which
    nearest-neighbor might fail.
    """
    linker = HodgesLinker(
        dmax=15.0,
        zones=np.zeros((0, 5), dtype=np.float64),
        adapt_thresholds=np.zeros(0, dtype=np.float64),
        adapt_values=np.zeros(0, dtype=np.float64),
    )

    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")
    t2 = np.datetime64("2025-12-01T12:00:00")

    # Two tracks crossing at T1
    # Track A: (0,0) -> (5,5) -> (10,10)
    # Track B: (0,10) -> (5,5) -> (10,0)

    # Detections (sorted by lat for ambiguity)
    detections = [
        (
            t0,
            np.array([0.0, 0.0]),
            np.array([0.0, 10.0]),
            {"msl": np.array([1000.0, 1000.0])},
        ),
        (
            t1,
            np.array([5.0, 5.0001]),
            np.array([5.0, 5.0001]),
            {"msl": np.array([990.0, 990.0])},
        ),
        (
            t2,
            np.array([10.0, 10.0]),
            np.array([10.0, 0.0]),
            {"msl": np.array([980.0, 980.0])},
        ),
    ]

    tracks = linker.link(detections)

    assert len(tracks) == 2
    # One track should go from (0,0) to (10,10)
    found_a = False
    for tr in tracks:
        if tr[0].lat == 0.0 and tr[0].lon == 0.0 and tr[2].lat == 10.0:
            found_a = True
            break
    assert found_a
