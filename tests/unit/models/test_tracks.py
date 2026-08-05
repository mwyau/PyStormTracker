from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

import pystormtracker
from pystormtracker.models.geo import SpatialBounds
from pystormtracker.models.tracks import (
    ProcessingStep,
    Tracks,
    TracksMetadata,
    _TracksBuilder,
)
from pystormtracker.models.units import Mode


def _metadata(
    primary_var: str = "msl",
    mode: Mode = "min",
    *,
    bounds: SpatialBounds | None = None,
    processing: tuple[ProcessingStep, ...] = (),
) -> TracksMetadata:
    return TracksMetadata(
        primary_var,
        mode,
        {primary_var: "Pa"},
        bounds,
        processing,
    )


def _tracks() -> Tracks:
    return Tracks(
        ids=[10, 20, 35],
        offsets=[0, 4, 7, 12],
        times=[
            1577836800000,
            1577858400000,
            1577880000000,
            1577901600000,
            1577923200000,
            1577944800000,
            1577966400000,
            1578009600000,
            1578031200000,
            1578052800000,
            1578074400000,
            1578096000000,
        ],
        lats=np.arange(12, dtype=np.float64),
        lons=np.arange(12, dtype=np.float64),
        variables={"msl": np.arange(12, dtype=np.float64)},
        metadata=_metadata(),
    )


def test_packed_shape_and_complete_offsets() -> None:
    tracks = _tracks()
    assert len(tracks) == 3
    assert tracks[0].point_slice == slice(0, 4)
    assert tracks[-1].point_slice == slice(7, 12)
    np.testing.assert_array_equal(
        tracks.point_track_ids(), [10] * 4 + [20] * 3 + [35] * 5
    )


def test_empty_layout_and_singleton_track() -> None:
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
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    builder.add_track(35, [1577836800000], [0.0], [0.0], {"msl": [1.0]})
    singleton = builder.finish()
    np.testing.assert_array_equal(singleton.offsets, [0, 1])


def test_builder_rejects_created_empty_tracks() -> None:
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    builder.new_track(10)
    with pytest.raises(ValueError, match=r"created track IDs have no points: \[10\]"):
        builder.finish()


def test_builder_rejects_duplicate_ids_and_empty_points() -> None:
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    builder.new_track(10)
    with pytest.raises(ValueError, match="duplicate track ID 10"):
        builder.new_track(10)
    invalid = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    invalid.add_track(
        20,
        [1577836800000, 1577836800000],
        [0.0, 0.0],
        [0.0, 0.0],
        {"msl": [1.0, 2.0]},
    )
    with pytest.raises(ValueError, match="times must be strictly increasing"):
        invalid.finish()


def test_builder_uses_integer_ids_and_direct_candidate_operations() -> None:
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    track_id = builder.new_track()
    assert isinstance(track_id, int)
    builder.append(track_id, 1577836800000, 1.0, 2.0, {"msl": 100000.0})
    builder.extend(
        track_id,
        [1577840400000],
        [3.0],
        [4.0],
        {"msl": [99000.0]},
    )
    assert builder.last_point(track_id) == (3.0, 4.0)
    tracks = builder.finish()
    np.testing.assert_array_equal(tracks.ids, [1])
    np.testing.assert_array_equal(tracks.offsets, [0, 2])


def test_builder_and_wire_types_are_not_root_public_api() -> None:
    assert not hasattr(pystormtracker, "_TracksBuilder")
    assert not hasattr(pystormtracker, "TracksMetadata")
    assert not hasattr(pystormtracker, "TrackJSONDocument")
    assert not hasattr(pystormtracker, "ProcessingStep")


def test_direct_tracks_construction_requires_explicit_metadata() -> None:
    with pytest.raises(ValueError, match="metadata is required"):
        Tracks()


@pytest.mark.parametrize(
    "bounds",
    [
        SpatialBounds(0.0, 90.0, -180.0, 180.0),
        SpatialBounds(0.0, 70.0, 120.0, -100.0),
        SpatialBounds(-20.0, 20.0, -30.0, 40.0),
    ],
)
def test_spatial_bounds_are_immutable_and_valid(bounds: SpatialBounds) -> None:
    metadata = _metadata(bounds=bounds)
    assert metadata.bounds == bounds
    with pytest.raises(AttributeError):
        bounds.south = -1.0  # type: ignore[misc]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "bounds",
    [
        (-91.0, 0.0, -180.0, 180.0),
        (0.0, 91.0, -180.0, 180.0),
        (10.0, 0.0, -180.0, 180.0),
        (0.0, 90.0, -181.0, 180.0),
        (0.0, 90.0, -180.0, 181.0),
        (0.0, 90.0, 0.0, 0.0),
    ],
)
def test_spatial_bounds_reject_invalid_intervals(
    bounds: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="bounds"):
        SpatialBounds(*bounds)


def test_canonical_empty_msl_and_metadata_alignment() -> None:
    tracks = Tracks(metadata=_metadata())
    assert set(tracks.variables) == {"msl"}
    assert set(tracks.variables) == set(tracks.units)
    assert tracks.primary_var in tracks.variables

    with pytest.raises(ValueError, match="identical keys"):
        Tracks(
            ids=[1],
            offsets=[0, 1],
            times=[1577836800000],
            lats=[0.0],
            lons=[0.0],
            variables={"msl": [1.0]},
            metadata=TracksMetadata("msl", "min", {"msl": "Pa", "vo": "s^-1"}),
        )


def test_core_model_has_no_cached_statistics() -> None:
    tracks = _tracks()
    assert not hasattr(tracks, "stats")
    assert not hasattr(tracks, "summaries")
    assert tracks == tracks.subset([0, 1, 2])


def test_processing_metadata_is_immutable_and_preserved_by_subset() -> None:
    processing = (ProcessingStep("spectral_filter", True, {"lmin": 5, "lmax": 42}),)
    tracks = _tracks().with_metadata(replace(_tracks().metadata, processing=processing))
    subset = tracks.subset([2])
    assert subset.metadata.processing == processing
    parameters = cast(
        MutableMapping[str, object],
        processing[0].parameters,
    )

    with pytest.raises(TypeError):
        parameters["lmin"] = 10


def test_bounds_are_preserved_by_subset() -> None:
    bounds = SpatialBounds(0.0, 90.0, 120.0, -100.0)
    tracks = _tracks().with_metadata(replace(_tracks().metadata, bounds=bounds))
    assert tracks.subset([2]).metadata.bounds == bounds

    assert tracks.metadata.bounds == bounds


def test_longitudes_are_normalized_to_signed_range() -> None:
    tracks = Tracks(
        ids=[1],
        offsets=[0, 4],
        times=[1577836800000, 1577923200000, 1578009600000, 1578096000000],
        lats=[0.0] * 4,
        lons=[180.0, 360.0, -181.0, 540.0],
        variables={"msl": [1.0, 2.0, 3.0, 4.0]},
        metadata=_metadata(),
    )
    np.testing.assert_array_equal(tracks.lons, [-180.0, 0.0, 179.0, -180.0])
