from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.models.tracks import (
    Tracks,
    TracksBuilder,
    TracksMetadata,
    compute_track_summaries,
)


def _metadata(primary_var: str = "msl", mode: str = "min") -> TracksMetadata:
    return TracksMetadata(primary_var, mode, {primary_var: "Pa"})  # type: ignore[arg-type]


def _tracks() -> Tracks:
    return Tracks(
        ids=np.array([10, 20, 35], dtype=np.int64),
        offsets=np.array([0, 4, 7, 12], dtype=np.int64),
        times=np.array(
            [
                "2020-01-01T00:00",
                "2020-01-01T06:00",
                "2020-01-01T12:00",
                "2020-01-01T18:00",
                "2020-01-02T00:00",
                "2020-01-02T06:00",
                "2020-01-02T12:00",
                "2020-01-03T00:00",
                "2020-01-03T06:00",
                "2020-01-03T12:00",
                "2020-01-03T18:00",
                "2020-01-04T00:00",
            ],
            dtype="datetime64[ms]",
        ),
        lats=np.arange(12, dtype=np.float64),
        lons=np.arange(12, dtype=np.float64),
        variables={"msl": np.arange(12, dtype=np.float64)},
        metadata=_metadata(),
    )


def test_packed_shape_and_direct_views() -> None:
    tracks = _tracks()
    assert len(tracks) == 3
    assert tracks[0].track_id == 10
    assert tracks[0].point_slice == slice(0, 4)
    assert tracks[-1].point_slice == slice(7, 12)
    assert not hasattr(tracks, "track_ids")
    np.testing.assert_array_equal(
        tracks.point_track_ids(), [10] * 4 + [20] * 3 + [35] * 5
    )


def test_empty_layout_and_singletons() -> None:
    empty = Tracks(
        ids=[],
        offsets=[0],
        times=[],
        lats=[],
        lons=[],
        variables={"msl": []},
        metadata=_metadata(),
    )
    assert len(empty) == 0
    builder = TracksBuilder("msl", "min", {"msl": "Pa"})
    builder.add_track(35, [np.datetime64("2020-01-01")], [0], [0], {"msl": [1]})
    singleton = builder.finish()
    np.testing.assert_array_equal(singleton.offsets, [0, 1])
    assert len(singleton[0]) == 1


@pytest.mark.parametrize(
    ("ids", "offsets", "point_count", "message"),
    [
        ([1, 1], [0, 1, 2], 2, "unique"),
        ([1], [1, 2], 2, "start"),
        ([1, 2], [0, 1, 1], 1, "strictly increasing"),
        ([1], [0, 2], 1, "final offset"),
        ([1], [0, 0], 0, "strictly increasing"),
    ],
)
def test_structural_invariants(
    ids: list[int], offsets: list[int], point_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Tracks(
            ids=ids,
            offsets=offsets,
            times=[
                np.datetime64("2020-01-01") + np.timedelta64(i, "D")
                for i in range(point_count)
            ],
            lats=[0.0] * point_count,
            lons=[0.0] * point_count,
            variables={"msl": [1.0] * point_count},
            metadata=_metadata(),
        )


@pytest.mark.parametrize(
    ("lats", "lons", "message"),
    [([91.0], [0.0], "latitudes"), ([0.0], [np.inf], "infinity")],
)
def test_coordinate_validation(
    lats: list[float], lons: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Tracks(
            ids=[1],
            offsets=[0, 1],
            times=[np.datetime64("2020-01-01")],
            lats=lats,
            lons=lons,
            variables={"msl": [1.0]},
            metadata=_metadata(),
        )


def test_time_and_variable_validation() -> None:
    with pytest.raises(ValueError, match="NaT"):
        Tracks(
            ids=[1],
            offsets=[0, 1],
            times=[np.datetime64("NaT")],
            lats=[0.0],
            lons=[0.0],
            variables={"msl": [1.0]},
            metadata=_metadata(),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        Tracks(
            ids=[1],
            offsets=[0, 2],
            times=[np.datetime64("2020-01-01"), np.datetime64("2020-01-01")],
            lats=[0.0, 0.0],
            lons=[0.0, 0.0],
            variables={"msl": [1.0, 2.0]},
            metadata=_metadata(),
        )
    with pytest.raises(ValueError, match="explicit unit"):
        Tracks(
            ids=[1],
            offsets=[0, 1],
            times=[np.datetime64("2020-01-01")],
            lats=[0.0],
            lons=[0.0],
            variables={"custom": [1.0]},
            metadata=TracksMetadata("custom", "max", {}),
        )


def test_immutability_and_read_only_mapping() -> None:
    tracks = _tracks()
    assert not tracks.ids.flags.writeable
    assert not tracks.variables["msl"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        tracks.ids[0] = 99
    with pytest.raises(TypeError):
        tracks.variables["new"] = np.empty(12)  # type: ignore[index]
    with pytest.raises(AttributeError):
        tracks.primary_var = "vo"  # type: ignore[misc]


def test_builder_backfills_new_variables() -> None:
    builder = TracksBuilder("msl", "min", {"msl": "Pa", "vo": "s^-1"})
    handle = builder.new_track(10)
    handle.append(np.datetime64("2020-01-01"), 0, 0, {"msl": 100000})
    handle.append(np.datetime64("2020-01-02"), 1, 1, {"msl": 99000, "vo": 1e-4})
    tracks = builder.finish()
    np.testing.assert_allclose(tracks.variables["vo"], [np.nan, 1e-4], equal_nan=True)


def test_subset_filter_sort_concatenate_and_summaries() -> None:
    tracks = _tracks().with_summaries(compute_track_summaries(_tracks()))
    subset = tracks.subset([-1, 0])
    np.testing.assert_array_equal(subset.ids, [35, 10])
    assert subset.summaries is not None
    filtered = tracks.filter([True, False, True])
    np.testing.assert_array_equal(filtered.ids, [10, 35])
    sorted_tracks = tracks.sort()
    np.testing.assert_array_equal(sorted_tracks.ids, [10, 20, 35])
    combined = Tracks.concatenate([tracks.subset([0]), tracks.subset([1])])
    np.testing.assert_array_equal(combined.ids, [10, 20])


def test_longitude_normalization() -> None:
    tracks = Tracks(
        ids=[1],
        offsets=[0, 4],
        times=[
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-04"),
        ],
        lats=[0, 0, 0, 0],
        lons=[180, 360, -181, 540],
        variables={"msl": [1, 2, 3, 4]},
        metadata=_metadata(),
    )
    np.testing.assert_array_equal(tracks.lons, [-180, 0, 179, -180])
