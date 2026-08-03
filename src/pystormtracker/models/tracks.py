"""Immutable packed trajectory storage and its mutable construction helper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from .center import Center
from .geo import geod_dist_km, normalize_longitudes_signed
from .units import canonical_unit_for


@dataclass(slots=True)
class TimeRange:
    """Metadata used by detector orchestration to select an input interval."""

    start: np.datetime64
    end: np.datetime64
    step: np.timedelta64 | None = None


@dataclass(frozen=True, slots=True)
class TracksMetadata:
    """Explicit metadata required to interpret a packed trajectory set."""

    primary_var: str
    mode: Literal["min", "max"]
    units: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.primary_var.strip():
            raise ValueError("primary_var must be nonempty")
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        normalized: dict[str, str] = {}
        for name, unit in self.units.items():
            if not name or not unit:
                raise ValueError("variable names and units must be nonempty")
            normalized[name] = unit
        object.__setattr__(self, "units", MappingProxyType(normalized))


def _copy_array(values: object, dtype: np.dtype[np.generic]) -> np.ndarray:
    """Copy an array into native-endian, C-contiguous storage."""
    native_dtype = dtype.newbyteorder("=")
    return np.array(values, dtype=native_dtype, order="C", copy=True)


def _integer_array(values: object, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype.kind in ("b", "U", "S", "O"):
        raise ValueError(f"{name} must contain integer values")
    if raw.dtype.kind == "f" and np.any(~np.isfinite(raw) | (raw != np.floor(raw))):
        raise ValueError(f"{name} must contain integer values")
    result = _copy_array(raw, np.dtype(np.int64)).astype(np.int64, copy=False)
    return result


def _float_array(values: object, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind in ("b", "U", "S", "O"):
        raise ValueError(f"{name} must contain numeric values")
    result = _copy_array(raw, np.dtype(np.float64)).astype(np.float64, copy=False)
    if np.any(np.isinf(result)):
        raise ValueError(f"{name} must not contain infinity")
    return result


def _time_array(values: object) -> NDArray[np.datetime64]:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise ValueError("times must contain datetime values")
    try:
        result = _copy_array(raw, np.dtype("datetime64[ms]")).astype(
            "datetime64[ms]", copy=False
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must contain datetime64-compatible values") from exc
    if np.any(np.isnat(result)):
        raise ValueError("times must not contain NaT")
    return result


def _freeze_array(values: object, dtype: np.dtype[np.generic]) -> np.ndarray:
    result = _copy_array(values, dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TrackSummaryColumns:
    """Derived, columnar per-track metrics used by interactive clients."""

    summary_version: int
    point_count: NDArray[np.int64]
    start_time: NDArray[np.datetime64]
    end_time: NDArray[np.datetime64]
    duration_hours: NDArray[np.float64]
    start_lat: NDArray[np.float64]
    start_lon: NDArray[np.float64]
    end_lat: NDArray[np.float64]
    end_lon: NDArray[np.float64]
    min_lat: NDArray[np.float64]
    max_lat: NDArray[np.float64]
    longitude_arc_start: NDArray[np.float64]
    longitude_arc_end: NDArray[np.float64]
    crosses_antimeridian: NDArray[np.bool_]
    peak_time: NDArray[np.datetime64]
    peak_lat: NDArray[np.float64]
    peak_lon: NDArray[np.float64]
    peak_value: NDArray[np.float64]
    path_length_km: NDArray[np.float64]
    displacement_km: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.summary_version != 1:
            raise ValueError("summary_version must be 1")
        int_fields = ("point_count",)
        time_fields = ("start_time", "end_time", "peak_time")
        bool_fields = ("crosses_antimeridian",)
        float_fields = (
            "duration_hours",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "min_lat",
            "max_lat",
            "longitude_arc_start",
            "longitude_arc_end",
            "peak_lat",
            "peak_lon",
            "peak_value",
            "path_length_km",
            "displacement_km",
        )
        lengths: set[int] = set()
        for name in int_fields:
            value = _freeze_array(getattr(self, name), np.dtype(np.int64))
            object.__setattr__(self, name, value)
            lengths.add(len(value))
        for name in time_fields:
            value = _freeze_array(getattr(self, name), np.dtype("datetime64[ms]"))
            object.__setattr__(self, name, value)
            lengths.add(len(value))
        for name in bool_fields:
            value = _freeze_array(getattr(self, name), np.dtype(np.bool_))
            object.__setattr__(self, name, value)
            lengths.add(len(value))
        for name in float_fields:
            value = _freeze_array(getattr(self, name), np.dtype(np.float64))
            object.__setattr__(self, name, value)
            lengths.add(len(value))
        if len(lengths) > 1:
            raise ValueError("summary columns must have equal lengths")

    def take(self, indices: NDArray[np.int64]) -> TrackSummaryColumns:
        """Return the selected rows as a new immutable summary container."""
        values: dict[str, object] = {}
        for name in (
            "point_count",
            "start_time",
            "end_time",
            "duration_hours",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "min_lat",
            "max_lat",
            "longitude_arc_start",
            "longitude_arc_end",
            "crosses_antimeridian",
            "peak_time",
            "peak_lat",
            "peak_lon",
            "peak_value",
            "path_length_km",
            "displacement_km",
        ):
            values[name] = getattr(self, name)[indices]
        return TrackSummaryColumns(summary_version=self.summary_version, **values)  # type: ignore[arg-type]


class Track:
    """A lightweight view into one packed track in a :class:`Tracks` parent."""

    __slots__ = ("_index", "_parent")

    def __init__(self, parent: Tracks, index: int) -> None:
        self._parent = parent
        self._index = index

    @property
    def track_id(self) -> int:
        return int(self._parent.ids[self._index])

    @property
    def point_slice(self) -> slice:
        return slice(
            int(self._parent.offsets[self._index]),
            int(self._parent.offsets[self._index + 1]),
        )

    @property
    def times(self) -> NDArray[np.datetime64]:
        return self._parent.times[self.point_slice]

    @property
    def lats(self) -> NDArray[np.float64]:
        return self._parent.lats[self.point_slice]

    @property
    def lons(self) -> NDArray[np.float64]:
        return self._parent.lons[self.point_slice]

    @property
    def variables(self) -> Mapping[str, NDArray[np.float64]]:
        return MappingProxyType(
            {
                name: values[self.point_slice]
                for name, values in self._parent.variables.items()
            }
        )

    def __iter__(self) -> Iterator[Center]:
        point_slice = self.point_slice
        for point_index in range(point_slice.start or 0, point_slice.stop or 0):
            yield Center(
                self._parent.times[point_index],
                float(self._parent.lats[point_index]),
                float(self._parent.lons[point_index]),
                {
                    name: float(values[point_index])
                    for name, values in self._parent.variables.items()
                },
            )

    def __len__(self) -> int:
        start = self.point_slice.start or 0
        stop = self.point_slice.stop or 0
        return stop - start

    def __getitem__(self, index: int) -> Center:
        length = len(self)
        normalized = index if index >= 0 else length + index
        if normalized < 0 or normalized >= length:
            raise IndexError("track point index out of range")
        point_index = (self.point_slice.start or 0) + normalized
        return Center(
            self._parent.times[point_index],
            float(self._parent.lats[point_index]),
            float(self._parent.lons[point_index]),
            {
                name: float(values[point_index])
                for name, values in self._parent.variables.items()
            },
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Track):
            return False
        if self.track_id != other.track_id or len(self) != len(other):
            return False
        return (
            np.array_equal(self.times, other.times)
            and np.array_equal(self.lats, other.lats, equal_nan=True)
            and np.array_equal(self.lons, other.lons, equal_nan=True)
            and set(self.variables) == set(other.variables)
            and all(
                np.array_equal(
                    self.variables[name], other.variables[name], equal_nan=True
                )
                for name in self.variables
            )
        )

    def abs_dist(self, other: Track | Center) -> float:
        """Return the distance between this track's last point and another point."""
        first = self[-1]
        second = other[0] if isinstance(other, Track) else other
        return first.abs_dist(second)


class Tracks:
    """Immutable packed trajectories with aligned per-point columns."""

    __slots__ = (
        "_frozen",
        "_ids",
        "_lats",
        "_lons",
        "_metadata",
        "_offsets",
        "_summaries",
        "_times",
        "_variables",
    )

    _ids: NDArray[np.int64]
    _offsets: NDArray[np.int64]
    _times: NDArray[np.datetime64]
    _lats: NDArray[np.float64]
    _lons: NDArray[np.float64]
    _variables: Mapping[str, NDArray[np.float64]]
    _metadata: TracksMetadata
    _summaries: TrackSummaryColumns | None
    _frozen: bool

    def __init__(
        self,
        ids: object | None = None,
        offsets: object | None = None,
        times: object | None = None,
        lats: object | None = None,
        lons: object | None = None,
        variables: Mapping[str, object] | None = None,
        metadata: TracksMetadata | None = None,
        summaries: TrackSummaryColumns | None = None,
        *,
        # Boundary-only compatibility for callers of the pre-packed constructor.
        track_ids: object | None = None,
        vars_dict: Mapping[str, object] | None = None,
        track_type: str | None = None,
        mode: Literal["min", "max"] | None = None,
        units: Mapping[str, str] | None = None,
    ) -> None:
        if ids is not None and track_ids is not None:
            raise TypeError("provide ids or track_ids, not both")
        if variables is not None and vars_dict is not None:
            raise TypeError("provide variables or vars_dict, not both")
        if track_ids is not None:
            legacy_ids = _integer_array(track_ids, "track_ids")
            ids, offsets = self._pack_legacy_ids(legacy_ids)
        if variables is None:
            variables = vars_dict
        if ids is None:
            ids = np.empty(0, dtype=np.int64)
        if offsets is None:
            offsets = np.array([0], dtype=np.int64)
        if times is None:
            times = np.empty(0, dtype="datetime64[ms]")
        if lats is None:
            lats = np.empty(0, dtype=np.float64)
        if lons is None:
            lons = np.empty(0, dtype=np.float64)
        if variables is None:
            variables = {}
        if metadata is None:
            primary_var = track_type if track_type is not None else "intensity"
            effective_mode: Literal["min", "max"] = mode or "max"
            metadata = TracksMetadata(
                primary_var,
                effective_mode,
                units or {primary_var: "1"},
            )
        elif any(value is not None for value in (track_type, mode, units)):
            raise TypeError("metadata cannot be combined with legacy metadata fields")

        packed_ids = _integer_array(ids, "ids")
        packed_offsets = _integer_array(offsets, "offsets")
        packed_times = _time_array(times)
        packed_lats = _float_array(lats, "lats")
        packed_lons = _float_array(lons, "lons")
        packed_lons = _float_array(normalize_longitudes_signed(packed_lons), "lons")
        packed_variables: dict[str, NDArray[np.float64]] = {}
        for name, values in variables.items():
            packed_variables[name] = _float_array(values, f"variable {name!r}")

        self._validate(
            packed_ids,
            packed_offsets,
            packed_times,
            packed_lats,
            packed_lons,
            packed_variables,
            metadata,
            summaries,
        )
        object.__setattr__(self, "_ids", _freeze_array(packed_ids, np.dtype(np.int64)))
        object.__setattr__(
            self, "_offsets", _freeze_array(packed_offsets, np.dtype(np.int64))
        )
        object.__setattr__(
            self, "_times", _freeze_array(packed_times, np.dtype("datetime64[ms]"))
        )
        object.__setattr__(
            self, "_lats", _freeze_array(packed_lats, np.dtype(np.float64))
        )
        object.__setattr__(
            self, "_lons", _freeze_array(packed_lons, np.dtype(np.float64))
        )
        object.__setattr__(
            self,
            "_variables",
            MappingProxyType(
                {
                    name: _freeze_array(values, np.dtype(np.float64))
                    for name, values in packed_variables.items()
                }
            ),
        )
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_summaries", summaries)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Tracks is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _pack_legacy_ids(
        point_ids: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if point_ids.size == 0:
            return np.empty(0, dtype=np.int64), np.array([0], dtype=np.int64)
        boundaries = np.flatnonzero(point_ids[1:] != point_ids[:-1]) + 1
        ids = point_ids[np.concatenate((np.array([0]), boundaries))]
        offsets = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                boundaries.astype(np.int64),
                np.array([point_ids.size]),
            )
        )
        if len(ids) + 1 != len(offsets):
            raise ValueError("legacy point IDs must describe contiguous tracks")
        return ids.astype(np.int64), offsets

    @staticmethod
    def _validate(
        ids: NDArray[np.int64],
        offsets: NDArray[np.int64],
        times: NDArray[np.datetime64],
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
        variables: Mapping[str, NDArray[np.float64]],
        metadata: TracksMetadata,
        summaries: TrackSummaryColumns | None,
    ) -> None:
        n_tracks = len(ids)
        n_points = len(times)
        if ids.ndim != 1:
            raise ValueError("ids must be one-dimensional")
        if offsets.ndim != 1 or len(offsets) != n_tracks + 1:
            raise ValueError("offsets must have length len(ids) + 1")
        if times.ndim != 1 or lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("point columns must be one-dimensional")
        if offsets[0] != 0:
            raise ValueError("offsets must start at zero")
        if offsets[-1] != n_points:
            raise ValueError("final offset must equal the point count")
        if len(np.unique(ids)) != n_tracks:
            raise ValueError("track IDs must be unique")
        if n_tracks and np.any(np.diff(offsets) <= 0):
            raise ValueError("offsets must be strictly increasing for tracks")
        if len(lats) != n_points or len(lons) != n_points:
            raise ValueError("point coordinate columns must have equal lengths")
        for name, values in variables.items():
            if len(values) != n_points:
                raise ValueError(f"variable {name!r} must have length N")
            expected_unit = canonical_unit_for(name)
            actual_unit = metadata.units.get(name)
            if actual_unit is None:
                raise ValueError(f"variable {name!r} requires an explicit unit")
            if expected_unit is not None and actual_unit != expected_unit:
                raise ValueError(
                    f"variable {name!r} must use canonical units {expected_unit!r}"
                )
        if n_points and metadata.primary_var not in variables:
            raise ValueError("primary_var must exist in variables for nonempty data")
        if np.any(~np.isfinite(lats)) or np.any((lats < -90.0) | (lats > 90.0)):
            raise ValueError("latitudes must be finite and in [-90, 90]")
        if np.any(~np.isfinite(lons)) or np.any((lons < -180.0) | (lons >= 180.0)):
            raise ValueError("longitudes must be finite and in [-180, 180)")
        for track_index in range(n_tracks):
            start = int(offsets[track_index])
            stop = int(offsets[track_index + 1])
            if np.any(times[start + 1 : stop] <= times[start : stop - 1]):
                raise ValueError("times must be strictly increasing within each track")
        if summaries is not None and len(summaries.point_count) != n_tracks:
            raise ValueError("summary columns must have length T")

    @property
    def ids(self) -> NDArray[np.int64]:
        return self._ids

    @property
    def offsets(self) -> NDArray[np.int64]:
        return self._offsets

    @property
    def times(self) -> NDArray[np.datetime64]:
        return self._times

    @property
    def lats(self) -> NDArray[np.float64]:
        return self._lats

    @property
    def lons(self) -> NDArray[np.float64]:
        return self._lons

    @property
    def variables(self) -> Mapping[str, NDArray[np.float64]]:
        return self._variables

    @property
    def metadata(self) -> TracksMetadata:
        return self._metadata

    @property
    def summaries(self) -> TrackSummaryColumns | None:
        return self._summaries

    @property
    def primary_var(self) -> str:
        return self.metadata.primary_var

    @property
    def mode(self) -> Literal["min", "max"]:
        return self.metadata.mode

    @property
    def units(self) -> Mapping[str, str]:
        return self.metadata.units

    @property
    def vars(self) -> Mapping[str, NDArray[np.float64]]:
        """Read-only compatibility view; use :attr:`variables` in new code."""
        return self.variables

    @property
    def track_type(self) -> str:
        """Read-only compatibility view; use :attr:`primary_var` in new code."""
        return self.primary_var

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[Track]:
        for index in range(len(self)):
            yield Track(self, index)

    @overload
    def __getitem__(self, index: int) -> Track: ...

    @overload
    def __getitem__(self, index: slice) -> Tracks: ...

    def __getitem__(self, index: int | slice) -> Track | Tracks:
        if isinstance(index, slice):
            indices = np.arange(len(self), dtype=np.int64)[index]
            return self.subset(indices)
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("track index out of range")
        return Track(self, normalized)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tracks):
            return False
        return (
            self.metadata == other.metadata
            and np.array_equal(self.ids, other.ids)
            and np.array_equal(self.offsets, other.offsets)
            and np.array_equal(self.times, other.times)
            and np.array_equal(self.lats, other.lats, equal_nan=True)
            and np.array_equal(self.lons, other.lons, equal_nan=True)
            and set(self.variables) == set(other.variables)
            and all(
                np.array_equal(
                    self.variables[name], other.variables[name], equal_nan=True
                )
                for name in self.variables
            )
        )

    def point_track_ids(self) -> NDArray[np.int64]:
        """Materialize point-level IDs as an explicitly requested derived array."""
        return np.repeat(self.ids, np.diff(self.offsets))

    def subset(self, indices: Sequence[int] | NDArray[np.int64]) -> Tracks:
        """Select complete tracks by their packed position."""
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("track indices must be one-dimensional")
        normalized = np.where(selected < 0, selected + len(self), selected)
        if np.any((normalized < 0) | (normalized >= len(self))):
            raise IndexError("track index out of range")
        new_ids = self.ids[normalized]
        counts = np.diff(self.offsets)[normalized]
        new_offsets = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )
        point_indices = (
            np.concatenate(
                [
                    np.arange(int(self.offsets[i]), int(self.offsets[i + 1]))
                    for i in normalized
                ]
            )
            if len(normalized)
            else np.empty(0, dtype=np.int64)
        )
        new_summaries = self.summaries.take(normalized) if self.summaries else None
        return Tracks(
            ids=new_ids,
            offsets=new_offsets,
            times=self.times[point_indices],
            lats=self.lats[point_indices],
            lons=self.lons[point_indices],
            variables={
                name: values[point_indices] for name, values in self.variables.items()
            },
            metadata=self.metadata,
            summaries=new_summaries,
        )

    def filter(self, mask: Sequence[bool] | NDArray[np.bool_]) -> Tracks:
        """Return complete tracks selected by a boolean per-track mask."""
        selected_mask = np.asarray(mask, dtype=np.bool_)
        if selected_mask.shape != (len(self),):
            raise ValueError("track filter must have length T")
        return self.subset(np.flatnonzero(selected_mask).astype(np.int64))

    def sort(self) -> Tracks:
        """Return tracks sorted by first time, latitude, and longitude."""
        if not len(self):
            return self
        starts = self.offsets[:-1]
        order = np.lexsort((self.lons[starts], self.lats[starts], self.times[starts]))
        return self.subset(order.astype(np.int64))

    def with_variables(
        self,
        variables: Mapping[str, object],
        *,
        metadata: TracksMetadata | None = None,
        compute_summaries: bool = False,
    ) -> Tracks:
        """Return a copy with a replacement variable mapping."""
        result = Tracks(
            ids=self.ids,
            offsets=self.offsets,
            times=self.times,
            lats=self.lats,
            lons=self.lons,
            variables=variables,
            metadata=metadata or self.metadata,
            summaries=None,
        )
        return (
            result.with_summaries(compute_track_summaries(result))
            if compute_summaries
            else result
        )

    def with_summaries(self, summaries: TrackSummaryColumns | None) -> Tracks:
        """Return the same packed data with validated derived summaries."""
        return Tracks(
            ids=self.ids,
            offsets=self.offsets,
            times=self.times,
            lats=self.lats,
            lons=self.lons,
            variables=self.variables,
            metadata=self.metadata,
            summaries=summaries,
        )

    @classmethod
    def concatenate(cls, tracks: Iterable[Tracks]) -> Tracks:
        """Concatenate complete packed trajectory sets."""
        items = tuple(tracks)
        if not items:
            return cls()
        first = items[0]
        for item in items[1:]:
            if item.metadata != first.metadata:
                raise ValueError("concatenated tracks must have matching metadata")
            if set(item.variables) != set(first.variables):
                raise ValueError("concatenated tracks must have matching variables")
        ids = np.concatenate([item.ids for item in items])
        counts = np.concatenate([np.diff(item.offsets) for item in items])
        offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
        variables = {
            name: np.concatenate([item.variables[name] for item in items])
            for name in first.variables
        }
        summaries = None
        if all(item.summaries is not None for item in items):
            summaries = _concatenate_summaries(
                [item.summaries for item in items if item.summaries is not None]
            )
        return cls(
            ids=ids,
            offsets=offsets,
            times=np.concatenate([item.times for item in items]),
            lats=np.concatenate([item.lats for item in items]),
            lons=np.concatenate([item.lons for item in items]),
            variables=variables,
            metadata=first.metadata,
            summaries=summaries,
        )

    def write(self, outfile: str, format: str = "imilast") -> None:
        """Write this track set in one of the packed-branch text formats."""
        if format == "imilast":
            from ..io.imilast import write_imilast

            write_imilast(self, outfile)
        elif format == "hodges":
            from ..io.hodges import write_hodges

            write_hodges(self, outfile)
        else:
            raise ValueError("packed branch supports 'imilast' and 'hodges' output")


@dataclass(slots=True)
class _BuilderTrack:
    track_id: int
    times: list[np.datetime64]
    lats: list[float]
    lons: list[float]
    variables: dict[str, list[float]]


class TrackHandle:
    """Mutable handle for one trajectory owned by a :class:`TracksBuilder`."""

    __slots__ = ("_builder", "_track_id")

    def __init__(self, builder: TracksBuilder, track_id: int) -> None:
        self._builder = builder
        self._track_id = track_id

    @property
    def track_id(self) -> int:
        return self._track_id

    @property
    def last_point(self) -> tuple[float, float]:
        """Return the current tail coordinate for linker bookkeeping."""
        track = self._builder._get_track(self._track_id)
        if not track.lats:
            raise ValueError("track has no points")
        return track.lats[-1], track.lons[-1]

    def append(
        self,
        time: np.datetime64,
        lat: float,
        lon: float,
        variables: Mapping[str, float],
    ) -> None:
        track = self._builder._get_track(self._track_id)
        track.times.append(np.datetime64(str(time), "ms"))
        track.lats.append(float(lat))
        track.lons.append(float(lon))
        for name in self._builder._variable_names | set(variables):
            if name not in track.variables:
                track.variables[name] = [np.nan] * (len(track.times) - 1)
            track.variables[name].append(float(variables.get(name, np.nan)))
        self._builder._variable_names.update(variables)

    def extend(
        self,
        times: object,
        lats: object,
        lons: object,
        variables: Mapping[str, object],
    ) -> None:
        time_array = _time_array(times)
        lat_array = _float_array(lats, "lats")
        lon_array = _float_array(lons, "lons")
        if not (len(time_array) == len(lat_array) == len(lon_array)):
            raise ValueError("builder point columns must have equal lengths")
        value_arrays = {
            name: _float_array(values, f"variable {name!r}")
            for name, values in variables.items()
        }
        if any(len(values) != len(time_array) for values in value_arrays.values()):
            raise ValueError(
                "builder variable columns must have length equal to points"
            )
        for point_index in range(len(time_array)):
            self.append(
                time_array[point_index],
                float(lat_array[point_index]),
                float(lon_array[point_index]),
                {
                    name: float(values[point_index])
                    for name, values in value_arrays.items()
                },
            )


class TracksBuilder:
    """Mutable, list-backed construction helper for finalized packed tracks."""

    def __init__(
        self,
        primary_var: str,
        mode: Literal["min", "max"],
        units: Mapping[str, str],
    ) -> None:
        self.metadata = TracksMetadata(primary_var, mode, units)
        self._tracks: list[_BuilderTrack] = []
        self._by_id: dict[int, _BuilderTrack] = {}
        self._next_id = 1
        self._variable_names: set[str] = set()

    def new_track(self, track_id: int | None = None) -> TrackHandle:
        if track_id is None:
            while self._next_id in self._by_id:
                self._next_id += 1
            track_id = self._next_id
            self._next_id += 1
        if isinstance(track_id, bool):
            raise ValueError("track_id must be an integer")
        track_id = int(track_id)
        if track_id in self._by_id:
            raise ValueError(f"duplicate track ID {track_id}")
        track = _BuilderTrack(track_id, [], [], [], {})
        self._tracks.append(track)
        self._by_id[track_id] = track
        return TrackHandle(self, track_id)

    def _get_track(self, track_id: int) -> _BuilderTrack:
        try:
            return self._by_id[track_id]
        except KeyError as exc:
            raise ValueError(f"unknown track ID {track_id}") from exc

    def add_track(
        self,
        track_id: int,
        times: object,
        lats: object,
        lons: object,
        variables: Mapping[str, object],
    ) -> TrackHandle:
        handle = self.new_track(track_id)
        handle.extend(times, lats, lons, variables)
        return handle

    def finish(self, compute_summaries: bool = False) -> Tracks:
        ids = np.asarray([track.track_id for track in self._tracks], dtype=np.int64)
        counts = np.asarray(
            [len(track.times) for track in self._tracks], dtype=np.int64
        )
        offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
        all_times = np.asarray(
            [value for track in self._tracks for value in track.times],
            dtype="datetime64[ms]",
        )
        all_lats = np.asarray(
            [value for track in self._tracks for value in track.lats], dtype=np.float64
        )
        all_lons = np.asarray(
            [value for track in self._tracks for value in track.lons], dtype=np.float64
        )
        variables: dict[str, NDArray[np.float64]] = {}
        for name in self._variable_names:
            variables[name] = np.asarray(
                [
                    value
                    for track in self._tracks
                    for value in track.variables.get(name, [np.nan] * len(track.times))
                ],
                dtype=np.float64,
            )
        result = Tracks(
            ids=ids,
            offsets=offsets,
            times=all_times,
            lats=all_lats,
            lons=all_lons,
            variables=variables,
            metadata=self.metadata,
        )
        return (
            result.with_summaries(compute_track_summaries(result))
            if compute_summaries
            else result
        )


def _concatenate_summaries(
    summaries: Sequence[TrackSummaryColumns],
) -> TrackSummaryColumns:
    names = (
        "point_count",
        "start_time",
        "end_time",
        "duration_hours",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "min_lat",
        "max_lat",
        "longitude_arc_start",
        "longitude_arc_end",
        "crosses_antimeridian",
        "peak_time",
        "peak_lat",
        "peak_lon",
        "peak_value",
        "path_length_km",
        "displacement_km",
    )
    values = {
        name: np.concatenate([getattr(item, name) for item in summaries])
        for name in names
    }
    return TrackSummaryColumns(summary_version=1, **values)


def _signed_longitude(value: float) -> float:
    return float(np.remainder(value + 180.0, 360.0) - 180.0)


def _longitude_arc(values: NDArray[np.float64]) -> tuple[float, float, bool]:
    sorted_values = np.sort(np.remainder(values, 360.0))
    if len(sorted_values) == 1:
        point = _signed_longitude(float(sorted_values[0]))
        return point, point, False
    gaps = np.diff(np.concatenate((sorted_values, [sorted_values[0] + 360.0])))
    largest_gap_index = int(np.argmax(gaps))
    start = float(sorted_values[(largest_gap_index + 1) % len(sorted_values)])
    end = float(sorted_values[largest_gap_index])
    start_signed = _signed_longitude(start)
    end_signed = _signed_longitude(end)
    width = float(360.0 - gaps[largest_gap_index])
    crosses = width > 0.0 and end_signed < start_signed
    return start_signed, end_signed, crosses


def _great_circle_array(
    lat1: NDArray[np.float64],
    lon1: NDArray[np.float64],
    lat2: NDArray[np.float64],
    lon2: NDArray[np.float64],
) -> NDArray[np.float64]:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_lon = np.deg2rad(lon1 - lon2)
    dot = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lon)
    return np.asarray(np.arccos(np.clip(dot, -1.0, 1.0)) * 6371.0, dtype=np.float64)


def compute_track_summaries(tracks: Tracks) -> TrackSummaryColumns:
    """Compute all derived track columns in O(N + T) time."""
    n_tracks = len(tracks)
    point_count = np.diff(tracks.offsets).astype(np.int64)
    start_time = np.empty(n_tracks, dtype="datetime64[ms]")
    end_time = np.empty(n_tracks, dtype="datetime64[ms]")
    duration_hours = np.empty(n_tracks, dtype=np.float64)
    start_lat = np.empty(n_tracks, dtype=np.float64)
    start_lon = np.empty(n_tracks, dtype=np.float64)
    end_lat = np.empty(n_tracks, dtype=np.float64)
    end_lon = np.empty(n_tracks, dtype=np.float64)
    min_lat = np.empty(n_tracks, dtype=np.float64)
    max_lat = np.empty(n_tracks, dtype=np.float64)
    longitude_arc_start = np.empty(n_tracks, dtype=np.float64)
    longitude_arc_end = np.empty(n_tracks, dtype=np.float64)
    crosses_antimeridian = np.empty(n_tracks, dtype=np.bool_)
    peak_time = np.full(n_tracks, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    peak_lat = np.full(n_tracks, np.nan, dtype=np.float64)
    peak_lon = np.full(n_tracks, np.nan, dtype=np.float64)
    peak_value = np.full(n_tracks, np.nan, dtype=np.float64)
    path_length_km = np.empty(n_tracks, dtype=np.float64)
    displacement_km = np.empty(n_tracks, dtype=np.float64)
    primary_values = tracks.variables.get(tracks.primary_var)

    for index in range(n_tracks):
        start = int(tracks.offsets[index])
        stop = int(tracks.offsets[index + 1])
        track_times = tracks.times[start:stop]
        track_lats = tracks.lats[start:stop]
        track_lons = tracks.lons[start:stop]
        start_time[index] = track_times[0]
        end_time[index] = track_times[-1]
        duration_hours[index] = float(
            (track_times[-1] - track_times[0]) / np.timedelta64(1, "h")
        )
        start_lat[index] = track_lats[0]
        start_lon[index] = track_lons[0]
        end_lat[index] = track_lats[-1]
        end_lon[index] = track_lons[-1]
        min_lat[index] = np.min(track_lats)
        max_lat[index] = np.max(track_lats)
        (
            longitude_arc_start[index],
            longitude_arc_end[index],
            crosses_antimeridian[index],
        ) = _longitude_arc(track_lons)
        if len(track_lats) > 1:
            path_length_km[index] = float(
                np.sum(
                    _great_circle_array(
                        track_lats[:-1], track_lons[:-1], track_lats[1:], track_lons[1:]
                    )
                )
            )
        else:
            path_length_km[index] = 0.0
        displacement_km[index] = float(
            geod_dist_km(track_lats[0], track_lons[0], track_lats[-1], track_lons[-1])
        )
        if primary_values is not None:
            values = primary_values[start:stop]
            finite = np.isfinite(values)
            if np.any(finite):
                valid_indices = np.flatnonzero(finite)
                relative_index = (
                    int(valid_indices[np.argmin(values[valid_indices])])
                    if tracks.mode == "min"
                    else int(valid_indices[np.argmax(values[valid_indices])])
                )
                peak_index = start + relative_index
                peak_time[index] = tracks.times[peak_index]
                peak_lat[index] = tracks.lats[peak_index]
                peak_lon[index] = tracks.lons[peak_index]
                peak_value[index] = values[relative_index]

    return TrackSummaryColumns(
        summary_version=1,
        point_count=point_count,
        start_time=start_time,
        end_time=end_time,
        duration_hours=duration_hours,
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        longitude_arc_start=longitude_arc_start,
        longitude_arc_end=longitude_arc_end,
        crosses_antimeridian=crosses_antimeridian,
        peak_time=peak_time,
        peak_lat=peak_lat,
        peak_lon=peak_lon,
        peak_value=peak_value,
        path_length_km=path_length_km,
        displacement_km=displacement_km,
    )
