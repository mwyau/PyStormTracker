"""Immutable packed trajectory storage and its mutable construction helper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

from ..time import TimePoint, encode_time_values
from .center import Center
from .geo import SpatialBounds, normalize_longitudes_signed
from .units import Mode, canonical_unit_for

if TYPE_CHECKING:
    from ..io.format import SupportedFormat


@dataclass(slots=True)
class TimeRange:
    """Metadata used by detector orchestration to select an input interval."""

    start: TimePoint | None
    end: TimePoint | None
    step: np.timedelta64 | None = None


JSONScalar: TypeAlias = str | int | float | bool | None

SPECTRAL_FILTER_OPERATION = "spectral_filter"
SPATIAL_TAPER_OPERATION = "spatial_taper"
REGRID_OPERATION = "regrid"


@dataclass(frozen=True, slots=True)
class ProcessingStep:
    """One recorded preprocessing operation and its JSON-compatible settings."""

    operation: str
    enabled: bool
    parameters: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("processing operation must be nonempty")
        normalized: dict[str, JSONScalar] = {}
        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("processing parameter names must be nonempty strings")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("processing parameters must contain finite floats")
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise ValueError("processing parameters must be JSON scalars")
            normalized[name] = value
        object.__setattr__(self, "parameters", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class TracksMetadata:
    """Explicit metadata required to interpret a packed trajectory set."""

    primary_var: str
    mode: Mode
    units: Mapping[str, str]
    bounds: SpatialBounds | None = None
    processing: tuple[ProcessingStep, ...] = ()

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
        if self.primary_var not in normalized:
            raise ValueError(
                f"primary_var {self.primary_var!r} requires a corresponding unit"
            )
        object.__setattr__(self, "units", MappingProxyType(normalized))
        object.__setattr__(self, "processing", tuple(self.processing))


def _copy_array(values: object, dtype: np.dtype[np.generic]) -> np.ndarray:
    """Copy an array into native-endian, C-contiguous storage."""
    native_dtype = dtype.newbyteorder("=")
    return np.array(values, dtype=native_dtype, order="C", copy=True)


def _integer_array(values: object, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype.kind in ("b", "U", "S", "O") or raw.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must contain integer values")
    if raw.dtype.kind == "f" and np.any(~np.isfinite(raw) | (raw != np.floor(raw))):
        raise ValueError(f"{name} must contain integer values")
    if raw.size and (
        np.any(raw < np.iinfo(np.int64).min) or np.any(raw > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} values must fit signed int64")
    return _copy_array(raw, np.dtype(np.int64)).astype(np.int64, copy=False)


def _float_array(values: object, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind in ("b", "U", "S", "O") or raw.dtype.kind not in (
        "i",
        "u",
        "f",
    ):
        raise ValueError(f"{name} must contain numeric values")
    result = _copy_array(raw, np.dtype(np.float64)).astype(np.float64, copy=False)
    if np.any(np.isinf(result)):
        raise ValueError(f"{name} must not contain infinity")
    return result


def _time_array(values: object) -> NDArray[np.int64]:
    try:
        result = encode_time_values(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must contain valid CF millisecond values") from exc
    return _copy_array(result, np.dtype(np.int64)).astype(np.int64, copy=False)


def _freeze_array(values: object, dtype: np.dtype[np.generic], name: str) -> np.ndarray:
    result = _copy_array(values, dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result.setflags(write=False)
    return result


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
    def times(self) -> NDArray[np.int64]:
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
                int(self._parent.times[point_index]),
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
            int(self._parent.times[point_index]),
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
        return (
            self.track_id == other.track_id
            and len(self) == len(other)
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

    def abs_dist(self, other: Track | Center) -> float:
        """Return the distance between this track's last point and another point."""
        first = self[-1]
        second = other[0] if isinstance(other, Track) else other
        return first.abs_dist(second)


class Tracks:
    """Immutable packed trajectories with aligned per-point columns."""

    _ids: NDArray[np.int64]
    _offsets: NDArray[np.int64]
    _times: NDArray[np.int64]
    _lats: NDArray[np.float64]
    _lons: NDArray[np.float64]
    _variables: Mapping[str, NDArray[np.float64]]
    _metadata: TracksMetadata
    _frozen: bool

    __slots__ = (
        "_frozen",
        "_ids",
        "_lats",
        "_lons",
        "_metadata",
        "_offsets",
        "_times",
        "_variables",
    )

    def __init__(
        self,
        ids: object | None = None,
        offsets: object | None = None,
        times: object | None = None,
        lats: object | None = None,
        lons: object | None = None,
        variables: Mapping[str, object] | None = None,
        metadata: TracksMetadata | None = None,
    ) -> None:
        if metadata is None:
            raise ValueError("metadata is required to construct Tracks")
        if ids is None:
            ids = np.empty(0, dtype=np.int64)
        if offsets is None:
            offsets = np.array([0], dtype=np.int64)
        if times is None:
            times = np.empty(0, dtype=np.int64)
        if lats is None:
            lats = np.empty(0, dtype=np.float64)
        if lons is None:
            lons = np.empty(0, dtype=np.float64)
        if variables is None:
            variables = {name: np.empty(0, dtype=np.float64) for name in metadata.units}

        packed_ids = _integer_array(ids, "ids")
        packed_offsets = _integer_array(offsets, "offsets")
        packed_times = _time_array(times)
        packed_lats = _float_array(lats, "lats")
        packed_lons = _float_array(lons, "lons")
        packed_lons = _float_array(normalize_longitudes_signed(packed_lons), "lons")
        packed_variables = {
            name: _float_array(values, f"variable {name!r}")
            for name, values in variables.items()
        }
        self._validate(
            packed_ids,
            packed_offsets,
            packed_times,
            packed_lats,
            packed_lons,
            packed_variables,
            metadata,
        )
        object.__setattr__(
            self, "_ids", _freeze_array(packed_ids, np.dtype(np.int64), "ids")
        )
        object.__setattr__(
            self,
            "_offsets",
            _freeze_array(packed_offsets, np.dtype(np.int64), "offsets"),
        )
        object.__setattr__(
            self,
            "_times",
            _freeze_array(packed_times, np.dtype(np.int64), "times"),
        )
        object.__setattr__(
            self, "_lats", _freeze_array(packed_lats, np.dtype(np.float64), "lats")
        )
        object.__setattr__(
            self, "_lons", _freeze_array(packed_lons, np.dtype(np.float64), "lons")
        )
        object.__setattr__(
            self,
            "_variables",
            MappingProxyType(
                {
                    name: _freeze_array(
                        values, np.dtype(np.float64), f"variable {name!r}"
                    )
                    for name, values in packed_variables.items()
                }
            ),
        )
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_frozen", True)

    @classmethod
    def empty(cls, metadata: TracksMetadata) -> Tracks:
        """Construct an empty packed result with explicit metadata."""
        variables = {name: np.empty(0, dtype=np.float64) for name in metadata.units}
        return cls(
            ids=np.empty(0, dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
            times=np.empty(0, dtype=np.int64),
            lats=np.empty(0, dtype=np.float64),
            lons=np.empty(0, dtype=np.float64),
            variables=variables,
            metadata=metadata,
        )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Tracks is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _validate(
        ids: NDArray[np.int64],
        offsets: NDArray[np.int64],
        times: NDArray[np.int64],
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
        variables: Mapping[str, NDArray[np.float64]],
        metadata: TracksMetadata,
    ) -> None:
        n_tracks = len(ids)
        n_points = len(times)
        if ids.ndim != 1:
            raise ValueError("ids must be one-dimensional")
        if offsets.ndim != 1 or len(offsets) != n_tracks + 1:
            raise ValueError("offsets must have length len(ids) + 1")
        if len(offsets) == 0 or offsets[0] != 0:
            raise ValueError("offsets must start at zero")
        if np.any(offsets < 0):
            raise ValueError("offsets must be nonnegative")
        if offsets[-1] != n_points:
            raise ValueError("final offset must equal the point count")
        if len(np.unique(ids)) != n_tracks:
            raise ValueError("track IDs must be unique")
        if n_tracks and np.any(np.diff(offsets) <= 0):
            raise ValueError("offsets must be strictly increasing for tracks")
        if len(lats) != n_points or len(lons) != n_points:
            raise ValueError("point coordinate columns must have equal lengths")
        if set(variables) != set(metadata.units):
            missing = sorted(set(variables) - set(metadata.units))
            extra = sorted(set(metadata.units) - set(variables))
            raise ValueError(
                "variables and units must have identical keys; "
                f"missing units={missing}, "
                f"extra units={extra}"
            )
        if metadata.primary_var not in variables:
            raise ValueError("primary_var must exist in variables")
        for name, values in variables.items():
            if len(values) != n_points:
                raise ValueError(f"variable {name!r} must have length N")
            expected_unit = canonical_unit_for(name)
            if expected_unit is not None and metadata.units[name] != expected_unit:
                raise ValueError(
                    f"variable {name!r} must use canonical units {expected_unit!r}"
                )
        if np.any(~np.isfinite(lats)) or np.any((lats < -90.0) | (lats > 90.0)):
            raise ValueError("latitudes must be finite and in [-90, 90]")
        if np.any(~np.isfinite(lons)) or np.any((lons < -180.0) | (lons >= 180.0)):
            raise ValueError("longitudes must be finite and in [-180, 180)")
        for track_index in range(n_tracks):
            start = int(offsets[track_index])
            stop = int(offsets[track_index + 1])
            if np.any(times[start + 1 : stop] <= times[start : stop - 1]):
                raise ValueError("times must be strictly increasing within each track")

    @property
    def ids(self) -> NDArray[np.int64]:
        return self._ids

    @property
    def offsets(self) -> NDArray[np.int64]:
        return self._offsets

    @property
    def times(self) -> NDArray[np.int64]:
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
    def primary_var(self) -> str:
        return self.metadata.primary_var

    @property
    def mode(self) -> Mode:
        return self.metadata.mode

    @property
    def units(self) -> Mapping[str, str]:
        return self.metadata.units

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
        """Compare canonical trajectory data and metadata, excluding cached stats."""
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
        """Select complete tracks by their packed position and preserve metadata."""
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
    ) -> Tracks:
        """Return a copy with replacement canonical variables and metadata."""
        return Tracks(
            ids=self.ids,
            offsets=self.offsets,
            times=self.times,
            lats=self.lats,
            lons=self.lons,
            variables=variables,
            metadata=metadata or self.metadata,
        )

    def with_metadata(self, metadata: TracksMetadata) -> Tracks:
        """Return canonical data with replacement metadata."""
        return Tracks(
            ids=self.ids,
            offsets=self.offsets,
            times=self.times,
            lats=self.lats,
            lons=self.lons,
            variables=self.variables,
            metadata=metadata,
        )

    @classmethod
    def concatenate(cls, tracks: Iterable[Tracks]) -> Tracks:
        """Concatenate complete packed trajectory sets."""
        items = tuple(tracks)
        if not items:
            raise ValueError("concatenating tracks requires at least one Tracks object")
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
        return cls(
            ids=ids,
            offsets=offsets,
            times=np.concatenate([item.times for item in items]),
            lats=np.concatenate([item.lats for item in items]),
            lons=np.concatenate([item.lons for item in items]),
            variables=variables,
            metadata=first.metadata,
        )

    def write(self, outfile: str | Path, format: SupportedFormat | None = None) -> None:
        """Write this track set through the public format router."""
        from ..io.format import save_tracks

        save_tracks(self, outfile, format=format)


@dataclass(slots=True)
class _TrackCandidate:
    """Private mutable storage used only while packing a Tracks result."""

    track_id: int
    times: list[int]
    lats: list[float]
    lons: list[float]
    variables: dict[str, list[float]]


class TracksBuilder:
    """Mutable, list-backed construction helper for finalized packed tracks."""

    def __init__(self, metadata: TracksMetadata) -> None:
        self.metadata = metadata
        self._tracks: list[_TrackCandidate] = []
        self._by_id: dict[int, _TrackCandidate] = {}
        self._next_id = 1
        self._variable_names: set[str] = set(metadata.units)

    def new_track(self, track_id: int | None = None) -> int:
        if track_id is None:
            while self._next_id in self._by_id:
                self._next_id += 1
            track_id = self._next_id
            self._next_id += 1
        if isinstance(track_id, bool) or not isinstance(track_id, (int, np.integer)):
            raise ValueError("track_id must be an integer")
        if track_id < np.iinfo(np.int64).min or track_id > np.iinfo(np.int64).max:
            raise ValueError("track_id must fit signed int64")
        track_id = int(track_id)
        if track_id in self._by_id:
            raise ValueError(f"duplicate track ID {track_id}")
        track = _TrackCandidate(track_id, [], [], [], {})
        self._tracks.append(track)
        self._by_id[track_id] = track
        return track_id

    def _get_track(self, track_id: int) -> _TrackCandidate:
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
    ) -> int:
        self.new_track(track_id)
        self.extend(track_id, times, lats, lons, variables)
        return track_id

    def append(
        self,
        track_id: int,
        time: object,
        lat: float,
        lon: float,
        variables: Mapping[str, float],
    ) -> None:
        track = self._get_track(track_id)
        encoded_time = int(_time_array([time])[0])
        track.times.append(encoded_time)
        track.lats.append(float(lat))
        track.lons.append(float(lon))
        names = self._variable_names | set(variables)
        for name in names:
            if name not in track.variables:
                track.variables[name] = [np.nan] * (len(track.times) - 1)
            track.variables[name].append(float(variables.get(name, np.nan)))
        self._variable_names.update(variables)

    def extend(
        self,
        track_id: int,
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
                track_id,
                int(time_array[point_index]),
                float(lat_array[point_index]),
                float(lon_array[point_index]),
                {
                    name: float(values[point_index])
                    for name, values in value_arrays.items()
                },
            )

    def last_point(self, track_id: int) -> tuple[float, float]:
        track = self._get_track(track_id)
        if not track.lats:
            raise ValueError("track has no points")
        return track.lats[-1], track.lons[-1]

    def finish(self) -> Tracks:
        empty = [track.track_id for track in self._tracks if not track.times]
        if empty:
            raise ValueError(f"created track IDs have no points: {empty}")
        ids = np.asarray([track.track_id for track in self._tracks], dtype=np.int64)
        counts = np.asarray(
            [len(track.times) for track in self._tracks], dtype=np.int64
        )
        offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
        all_times = np.asarray(
            [value for track in self._tracks for value in track.times],
            dtype=np.int64,
        )
        all_lats = np.asarray(
            [value for track in self._tracks for value in track.lats], dtype=np.float64
        )
        all_lons = np.asarray(
            [value for track in self._tracks for value in track.lons], dtype=np.float64
        )
        variables = {
            name: np.asarray(
                [
                    value
                    for track in self._tracks
                    for value in track.variables.get(name, [np.nan] * len(track.times))
                ],
                dtype=np.float64,
            )
            for name in sorted(self._variable_names)
        }
        return Tracks(
            ids=ids,
            offsets=offsets,
            times=all_times,
            lats=all_lats,
            lons=all_lons,
            variables=variables,
            metadata=self.metadata,
        )
